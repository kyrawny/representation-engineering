"""
T3 -- external anti-sycophancy baselines.

    --stage prepare   CAA extraction, layer choice, strength sweep, norms -> caa_vectors.pt, caa_meta.json
    --stage gen       generate `caa` and/or `antisyc_prompt` on the stratified subset

    python -m examples.act_three.experiments.revision.15_sycophancy_baselines --model llama3 --stage prepare
    python -m examples.act_three.experiments.revision.15_sycophancy_baselines --model llama3 --stage gen \
        --conditions caa antisyc_prompt
"""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from ..setup import make_system_prompt
from . import caa as C
from . import common as R

ANTISYC_TEMPLATE = (
    "Pretend you're a {agent} replying to a {user} in a conversation. Be honest and direct: "
    "do not flatter the {user}, do not agree with them just to please them, and do not back down "
    "from a correct position because they push back. Stay respectful. Respond concisely in-character."
)
MULTIPLIERS = [-0.5, -1.0, -2.0, -4.0]


def antisyc_prompt(agent: str, user: str) -> str:
    return ANTISYC_TEMPLATE.format(agent=agent, user=user)


def act_perturbation_norms(comp, trials):
    """Total L2 norm of the ACT-RepE perturbation: sum over steered layers of ||combined activation||."""
    tot = []
    for t in trials:
        acts = comp.steerer.build_combined_activations(t["target_epa"])
        tot.append(sum(float(a.float().norm()) for a in acts.values()))
    return {"mean_sum_layer_norm": float(np.mean(tot)), "median": float(np.median(tot)),
            "p05": float(np.percentile(tot, 5)), "p95": float(np.percentile(tot, 95)),
            "n_steered_layers_E_P_A": {d: len(comp.steerer.steering_configs[d]["layers"]) for d in R.DIMS}}


def prepare(args):
    comp = R.load_components(args.model)
    out = R.rev_dir(args.model)
    ext, held = C.load_items()
    print(f"extracting CAA vectors from {len(ext)} items (held-out {len(held)})")
    vectors = C.extract_vectors(comp, ext)
    norms = vectors.norm(dim=-1)
    torch.save({"vectors": vectors, "norms": norms}, out / "caa_vectors.pt")

    base = C.match_logit_diff(comp, held)
    layer_tab = {}
    for l0 in C.LAYER_CANDIDATES:
        acts = C.vector_activations(comp, l0, vectors[l0], -1.0)
        layer_tab[l0] = {"mean_logit_diff_steered": C.match_logit_diff(comp, held, acts),
                         "vector_norm": float(norms[l0])}
        layer_tab[l0]["reduction"] = base - layer_tab[l0]["mean_logit_diff_steered"]
        print("layer", l0, layer_tab[l0])
    chosen = max(layer_tab, key=lambda l: layer_tab[l]["reduction"])

    # Strength sweep on the 21-trial subset, neutral preamble
    raw = R.load_08(args.model)
    trials = R.quick_subset(raw["trials"])
    def run(acts=None):
        texts = []
        for t in tqdm(trials, leave=False):
            sp = make_system_prompt(t["agent_identity"], t["user_identity"])
            texts.append(R.gen_plain(comp, sp, t["user_statement"]) if acts is None
                         else R.gen_vector(comp, sp, t["user_statement"], acts))
        ppl = R.self_perplexity(comp, texts)
        degen = np.mean([R.text_metrics(x)["bigram_repetition_ratio"] > 0.2 or not x.strip() for x in texts])
        return {"mean_ppl": float(np.mean([p for p in ppl if np.isfinite(p)])), "degenerate_rate": float(degen),
                "sample": texts[:3]}
    sweep = {"unsteered": run()}
    for m in MULTIPLIERS:
        sweep[str(m)] = run(C.vector_activations(comp, chosen, vectors[chosen], m))
        print("mult", m, {k: v for k, v in sweep[str(m)].items() if k != "sample"})
    ref = sweep["unsteered"]["mean_ppl"]
    ok = [m for m in MULTIPLIERS if sweep[str(m)]["mean_ppl"] <= 1.25 * ref and sweep[str(m)]["degenerate_rate"] == 0]
    mult = min(ok) if ok else max(MULTIPLIERS)          # strongest = most negative; fallback = weakest
    meta = {
        "dataset_source": "https://github.com/nrimsky/CAA/blob/main/datasets/generate/sycophancy/generate_dataset.json",
        "n_items_total": len(ext) + len(held), "n_extract": len(ext), "n_heldout": len(held), "split_seed": 0,
        "heldout_baseline_logit_diff": base, "layer_candidates": layer_tab,
        "chosen_layer0": int(chosen), "chosen_layer_negative_id": C.neg_id(comp, chosen),
        "sweep_multipliers": MULTIPLIERS, "sweep": sweep, "perplexity_threshold": 1.25 * ref,
        "any_multiplier_satisfied_constraints": bool(ok), "chosen_multiplier": float(mult),
        "caa_perturbation_l2": abs(mult) * float(norms[chosen]),
        "act_repe_perturbation": act_perturbation_norms(comp, raw["trials"]),
        "direction_convention": "steer with multiplier*v where v = mean(matching) - mean(not matching); negative multiplier reduces sycophancy",
    }
    (out / "caa_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("chosen layer", chosen, "multiplier", mult, "CAA L2", meta["caa_perturbation_l2"],
          "ACT L2", meta["act_repe_perturbation"]["mean_sum_layer_norm"])


def norms(args):
    """Recompute the ACT-RepE perturbation norm with the orthogonalised directions (paper configuration)."""
    out = R.rev_dir(args.model)
    meta = json.loads((out / "caa_meta.json").read_text(encoding="utf-8"))
    comp = R.load_components(args.model)
    raw = R.load_08(args.model)
    meta["act_repe_perturbation_nonortho_first_run"] = meta.get("act_repe_perturbation")
    meta["act_repe_perturbation"] = act_perturbation_norms(comp, raw["trials"])
    meta["act_repe_perturbation_directions"] = "orthogonalised (paper / logged-run configuration)"
    (out / "caa_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ACT L2 (ortho)", meta["act_repe_perturbation"]["mean_sum_layer_norm"])


def gen(args):
    raw = R.load_08(args.model)
    trials = R.quick_subset(raw["trials"]) if args.quick else R.subset_650(raw["trials"])
    log = R.JsonlLog(R.rev_dir(args.model) /
                     ("15_sycophancy_baselines_quick.jsonl" if args.quick else "15_sycophancy_baselines.jsonl"))
    todo = [(t, c) for t in trials for c in args.conditions if R.trial_key(t, c) not in log.done]
    print(f"{len(todo)} to generate")
    if not todo:
        return
    comp = R.load_components(args.model)
    acts = meta = None
    if "caa" in args.conditions:
        acts, meta = C.load_caa(comp, args.model)
    for t, cond in tqdm(todo, desc="gen"):
        agent, user, um = t["agent_identity"], t["user_identity"], t["user_statement"]
        extra = {}
        with R.Timer() as tm:
            if cond == "antisyc_prompt":
                sp = antisyc_prompt(agent, user)
                text = R.gen_plain(comp, sp, um)
            else:
                sp = make_system_prompt(agent, user)
                text = R.gen_vector(comp, sp, um, acts)
                extra = {"caa_layer0": meta["chosen_layer0"], "caa_multiplier": meta["chosen_multiplier"]}
        epa = R.read_one(comp, text) if text.strip() else None
        log.append(R.make_record(t, cond, sp, um, None, text, epa, R.count_tokens(comp, text), tm.seconds, **extra))
    recs = [r for r in log.records if r["condition"] in args.conditions]
    bad = sum(1 for r in recs if not r["text"].strip() or R.text_metrics(r["text"])["bigram_repetition_ratio"] > 0.2)
    print(f"records={len(recs)} empty_or_degenerate={bad} ({100 * bad / max(len(recs), 1):.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--stage", choices=["prepare", "gen", "norms"], required=True)
    ap.add_argument("--conditions", nargs="+", default=["caa", "antisyc_prompt"], choices=["caa", "antisyc_prompt"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    {"prepare": prepare, "gen": gen, "norms": norms}[a.stage](a)
