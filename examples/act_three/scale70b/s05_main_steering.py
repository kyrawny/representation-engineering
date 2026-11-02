"""
Stage 5: the main experiment, i.e. the paper's experiment 08 at 70B.

For each (scenario, identity pair) trial it generates, with the paper's
prompts and greedy decoding:
    unsteered   identity preamble only
    repe_only   identity preamble + activation steering toward the ACT target
    pe_only     identity preamble + affective instruction (target descriptors)
    hybrid      affective instruction + activation steering
Optional conditions (--conditions):
    repe_const     steering toward one constant target (mean of all ACT targets)
    repe_only_s1   repe_only at the literal 8B strength (s = 1)
    hybrid_s1      hybrid at the literal 8B strength (s = 1)
and reads the EPA of every response with this model's own reader, as in the
paper.

Each trial is written to a JSONL file as soon as it finishes, so a crashed or
pre-empted job resumes where it stopped.  Use --shard-index/--num-shards to
split the trials across several jobs or nodes (each loads its own model copy).

Usage::

    python -m examples.act_three.scale70b.s05_main_steering --model meta-llama/Llama-3.1-70B-Instruct \
        --shard-index 0 --num-shards 2
"""

import argparse
import time
from datetime import datetime

import numpy as np

from . import settings as S
from .common import (JsonlLog, load_components, load_json, method_api, safe_read_batch_size, shard,
                     strength_context, subset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--conditions", nargs="*", default=S.CORE_CONDITIONS,
                    choices=S.CORE_CONDITIONS + S.OPTIONAL_CONDITIONS)
    ap.add_argument("--subset", default="all", choices=["all", "half", "calibration"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-trials", type=int, default=None, help="For smoke tests.")
    ap.add_argument("--read-batch-size", type=int, default=8)
    ap.add_argument("--tag", default="main", help="Output sub-folder name.")
    args = ap.parse_args()

    api = method_api()
    tpath = S.results_dir(args.model) / "targets.json"
    if not tpath.exists():
        raise SystemExit(f"{tpath} missing: run stage 2 first.")
    tg = load_json(tpath)
    cfg_path = S.model_dir(args.model) / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} missing: run stage 4 with --write-config first.")

    comp = load_components(args.model, need_steerer=True)
    comp.fast_reader.batch_size = safe_read_batch_size(args.model, args.read_batch_size)
    print(f"Strength s = {comp.strength}  (coefficients x s, clamp x s)")

    trials = subset(tg["trials"], args.subset)
    trials = shard(trials, args.shard_index, args.num_shards)
    if args.max_trials:
        trials = trials[:args.max_trials]
    out = S.results_dir(args.model) / args.tag / f"shard_{args.shard_index}_of_{args.num_shards}.jsonl"
    log = JsonlLog(out)
    todo = [t for t in trials if t["key"] not in log.done]
    print(f"{len(trials)} trials in shard, {len(todo)} to do -> {out}")

    const = tg["constant_target"]
    gen = api["GENERATION_DEFAULTS"]
    t_start = time.time()
    for i, t in enumerate(todo):
        target = t["target_epa"]
        neutral = api["format_chat_prompt"](comp.tokenizer,
                                            api["make_system_prompt"](t["agent_identity"], t["user_identity"]),
                                            t["user_statement"])
        affective = api["format_chat_prompt"](comp.tokenizer,
                                              api["affective_prompt"](t["agent_identity"], t["user_identity"], target),
                                              t["user_statement"])
        plans = {
            "unsteered": lambda: api["generate_unsteered"](comp.model, comp.tokenizer, neutral),
            "repe_only": lambda: comp.steerer.generate(prompt=neutral, target_epa=target, **gen),
            "pe_only": lambda: api["generate_unsteered"](comp.model, comp.tokenizer, affective),
            "hybrid": lambda: comp.steerer.generate(prompt=affective, target_epa=target, **gen),
            "repe_const": lambda: comp.steerer.generate(prompt=neutral, target_epa=const, **gen),
        }
        conds = {}
        for c in args.conditions:
            t0 = time.time()
            if c.endswith("_s1"):
                base = c[:-3]
                with strength_context(comp.steerer, 1.0, comp.strength):
                    text = plans[base]()
            else:
                text = plans[c]()
            text = api["clean_response"](text)
            conds[c] = {"text": text, "text500": text[:S.TEXT500], "gen_seconds": time.time() - t0,
                        "target_used": const if c == "repe_const" else target}
        epas = comp.fast_reader.read([conds[c]["text"] for c in args.conditions])
        for c, e in zip(args.conditions, epas):
            dist = {d: abs(e[d] - target[d]) for d in S.DIMS}
            conds[c].update({"epa": e, "distances": dist, "mean_distance": float(np.mean(list(dist.values())))})
        log.append({**{k: t[k] for k in ("key", "index", "scenario_id", "category", "pair",
                                         "agent_identity", "user_identity", "agent_identity_epa",
                                         "user_identity_epa", "user_statement", "user_statement_epa",
                                         "target_epa")},
                    "strength": comp.strength, "constant_target": const, "conditions": conds,
                    "finished_at": datetime.now().isoformat()})
        el = time.time() - t_start
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i + 1}/{len(todo)}] {t['key']}  "
              f"unst={conds.get('unsteered', {}).get('mean_distance', float('nan')):.2f} "
              f"hyb={conds.get('hybrid', {}).get('mean_distance', float('nan')):.2f}  "
              f"ETA {eta / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
