"""
Stage 4: choose the steering strength for this model with a pre-specified
degeneration rule.  This is the paper's coefficient calibration ("as strong as
possible while staying below the text-degeneration threshold"), written as
an explicit rule so that it can be applied identically to 8B and 70B.

Procedure (63 calibration trials: 3 scenarios per category x 3 identity pairs):
  1. Generate the unsteered response for each trial.
  2. For s in STRENGTH_GRID (ascending; s_match from stage 3 is added if
     available), generate the RepE-only response steered toward the trial's
     ACT target with coefficients and clamp limit scaled by s.
  3. s passes if, over the 63 trials:
       - fraction of degenerate outputs <= MAX_DEGENERATE_FRAC
         (degenerate: < 20 characters, or bigram repetition > 0.2);
       - geometric-mean self-perplexity ratio (steered / unsteered) <= MAX_PPL_RATIO;
       - mean length ratio within LENGTH_RATIO_BOUNDS.
  4. The chosen strength is the largest s such that s and every smaller grid
     value pass.  The sweep stops at the first failure.

The EPA shift toward the target is recorded for information only.  It is NOT
used to choose s, so the choice cannot be tuned toward the result.

Validate the rule on 8B first (the paper used s = 1)::

    python -m examples.act_three.scale70b.s04_strength_calibration --model meta-llama/Llama-3.1-8B-Instruct

Then::

    python -m examples.act_three.scale70b.s04_strength_calibration --model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse
import json
import math

import numpy as np

from . import settings as S
from .common import (JsonlLog, all_trials, apply_strength, coherence_api, geo_mean, load_components,
                     load_json, method_api, read_jsonl, save_json, subset)


def is_degenerate(text, text_metrics):
    if len(text.strip()) < S.DEGENERATE_MIN_CHARS:
        return True
    return text_metrics(text).get("bigram_repetition_ratio", 0.0) > S.DEGENERATE_MAX_BIGRAM_REPEAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--write-config", action="store_true",
                    help="Write models/<model>/config.json with the chosen strength "
                         "(refused for the paper's models).")
    args = ap.parse_args()

    api = method_api()
    comp = load_components(args.model, need_steerer=True, strength=1.0)
    ppl_fn, text_metrics, _ = coherence_api()
    out_dir = S.results_dir(args.model)
    log = JsonlLog(out_dir / "calibration.jsonl")

    grid = sorted(set(args.grid or S.STRENGTH_GRID))
    np_path = out_dir / "norm_probe.json"
    s_match = load_json(np_path).get("s_match") if np_path.exists() else None
    if s_match and not args.grid:
        grid = sorted(set(grid) | {round(float(s_match), 2)})

    trials = subset(all_trials(api), "calibration")
    targets = _targets(args.model)

    def prompt_for(t):
        return api["format_chat_prompt"](comp.tokenizer,
                                         api["make_system_prompt"](t["agent_identity"], t["user_identity"]),
                                         t["user_statement"])

    def run(cond_key, gen):
        for t in trials:
            key = f"{t['key']}|{cond_key}"
            if key in log.done:
                continue
            text = api["clean_response"](gen(t))
            epa = comp.fast_reader.read([text])[0]
            log.append({"key": key, "trial": t["key"], "cond": cond_key, "text": text, "epa": epa})

    run("unsteered", lambda t: api["generate_unsteered"](comp.model, comp.tokenizer, prompt_for(t)))
    summary, chosen = [], None
    for s in grid:
        apply_strength(comp.steerer, s)
        run(f"s={s}", lambda t: comp.steerer.generate(prompt=prompt_for(t), target_epa=targets[t["key"]],
                                                      **api["GENERATION_DEFAULTS"]))
        row = _evaluate(log, trials, targets, s, comp, ppl_fn, text_metrics)
        summary.append(row)
        print(json.dumps(row))
        if not row["pass"]:
            break
        chosen = s
    apply_strength(comp.steerer, 1.0)

    res = {"model": args.model, "grid": grid, "s_match": s_match, "rule": {
        "max_ppl_ratio": S.MAX_PPL_RATIO, "max_degenerate_frac": S.MAX_DEGENERATE_FRAC,
        "length_ratio_bounds": S.LENGTH_RATIO_BOUNDS, "degenerate_min_chars": S.DEGENERATE_MIN_CHARS,
        "degenerate_max_bigram_repeat": S.DEGENERATE_MAX_BIGRAM_REPEAT},
        "summary": summary, "chosen_strength": chosen}
    if chosen is None:
        res["warning"] = "No grid strength passed; the smallest grid value failed. Inspect calibration.jsonl."
    save_json(res, out_dir / "calibration.json")
    print(f"\nChosen strength: {chosen}")

    if args.write_config and chosen is not None:
        short = S.short_name(args.model)
        if short in S.PROTECTED_SHORT_NAMES:
            print(f"Refusing to write config for paper model '{short}'.")
        else:
            cfg = {"model_name": args.model, "short_name": short,
                   "strength": chosen, "clamp_limit": chosen * S.REFERENCE_CLAMP_LIMIT,
                   "per_dim_coefficients": {d: chosen * c for d, c in S.REFERENCE_COEFFICIENTS.items()},
                   "reference_coefficients": S.REFERENCE_COEFFICIENTS,
                   "reading_method": S.READING_METHOD,
                   "use_orthogonal_directions": S.USE_ORTHOGONAL_DIRECTIONS,
                   "notes": "Written by scale70b/s04_strength_calibration.py (pre-specified rule)."}
            save_json(cfg, S.model_dir(args.model) / "config.json")
            print(f"Wrote {S.model_dir(args.model) / 'config.json'}")


def _evaluate(log, trials, targets, s, comp, ppl_fn, text_metrics):
    recs = {r["key"]: r for r in read_jsonl([log.path])}
    keys = [t["key"] for t in trials]
    un = [recs[f"{k}|unsteered"] for k in keys]
    st = [recs[f"{k}|s={s}"] for k in keys]
    ppl_un = ppl_fn(comp.model, comp.tokenizer, [r["text"] for r in un])
    ppl_st = ppl_fn(comp.model, comp.tokenizer, [r["text"] for r in st])
    degenerate = [is_degenerate(r["text"], text_metrics) for r in st]
    len_ratio = float(np.mean([len(b["text"]) / max(len(a["text"]), 1) for a, b in zip(un, st)]))
    ppl_ratio = geo_mean(ppl_st) / geo_mean(ppl_un)
    shift = {d: float(np.mean([abs(a["epa"][d] - targets[k][d]) - abs(b["epa"][d] - targets[k][d])
                               for a, b, k in zip(un, st, keys)])) for d in S.DIMS}
    ok = (np.mean(degenerate) <= S.MAX_DEGENERATE_FRAC and ppl_ratio <= S.MAX_PPL_RATIO
          and S.LENGTH_RATIO_BOUNDS[0] <= len_ratio <= S.LENGTH_RATIO_BOUNDS[1]
          and math.isfinite(ppl_ratio))
    return {"strength": s, "degenerate_frac": float(np.mean(degenerate)), "ppl_ratio_geo": ppl_ratio,
            "length_ratio": len_ratio, "distance_reduction_info_only": shift, "pass": bool(ok),
            "example": st[0]["text"][:300]}


def _targets(model_name):
    from .s03_norm_probe import _targets_for
    api = method_api()
    return _targets_for(model_name, subset(all_trials(api), "calibration"))


if __name__ == "__main__":
    main()
