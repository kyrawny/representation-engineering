"""
Stage 3: how large is the steering perturbation relative to the residual
stream, at strength s = 1?

The paper's steering adds a perturbation of fixed absolute size (unit-norm
directions, clamp limit 1.0).  Residual-stream norms grow with model size, so
the same absolute perturbation is relatively smaller in a 70B model.  This
stage measures

    rel(model) = mean over steered layers and trials of ||delta_l|| / ||h_l||

where ||h_l|| is the median token norm of the decoder-layer output over the
prompt (first two BOS positions excluded) and delta_l is the
paper's perturbation for that trial's ACT target.  The ratio
s_match = rel(8B) / rel(70B) is the strength that gives the 70B model the same
*relative* perturbation as the paper's 8B setting.  It is reported and also
tried in the calibration grid (stage 4); it is not used on its own to choose
the strength.

Run it for both models (run_all.sh stage R does the 8B run; it fits on one 24 GB GPU)::

    python -m examples.act_three.scale70b.s03_norm_probe --model meta-llama/Llama-3.1-8B-Instruct
    python -m examples.act_three.scale70b.s03_norm_probe --model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse

import numpy as np
import torch

from . import settings as S
from .common import (all_trials, load_components, load_json, method_api, save_json, subset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    args = ap.parse_args()

    api = method_api()
    comp = load_components(args.model, need_steerer=True, strength=1.0)
    trials = subset(all_trials(api), "calibration")
    targets = _targets_for(args.model, trials)

    layers_all = comp.steerer.all_layers
    norms = {l: [] for l in layers_all}

    blocks = comp.model.model.layers
    hooks = []

    def make_hook(l):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            tok_norms = h.float().norm(dim=-1)[0]
            # Skip the first two positions: Llama's BOS tokens carry "massive
            # activations" that would dominate a mean.  Median over the rest.
            tok_norms = tok_norms[2:] if tok_norms.numel() > 4 else tok_norms
            norms[l].append(float(tok_norms.median().item()))
        return hook

    for l in layers_all:
        target_block = blocks[l]
        target_block = getattr(target_block, "block", target_block)   # if wrapped by repe
        hooks.append(target_block.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        for t in trials:
            prompt = api["format_chat_prompt"](comp.tokenizer,
                                               api["make_system_prompt"](t["agent_identity"], t["user_identity"]),
                                               t["user_statement"])
            ids = comp.tokenizer(prompt, return_tensors="pt").to(comp.model.device)
            comp.model(**ids)
    for h in hooks:
        h.remove()

    mean_norm = {l: float(np.mean(v)) for l, v in norms.items() if v}
    rel_per_trial, per_layer = [], {}
    for t in trials:
        acts = comp.steerer.build_combined_activations(targets[t["key"]])
        vals = []
        for l, vec in acts.items():
            n = float(vec.float().norm().item())
            if n > 0:
                vals.append(n / mean_norm[l])
                per_layer.setdefault(l, []).append(n / mean_norm[l])
        rel_per_trial.append(float(np.mean(vals)) if vals else 0.0)

    res = {"model": args.model, "n_trials": len(trials), "n_layers": len(layers_all),
           "steered_layers": sorted(per_layer),
           "mean_residual_norm_steered_layers": float(np.mean([mean_norm[l] for l in per_layer])),
           "rel_perturbation_s1_mean": float(np.mean(rel_per_trial)),
           "rel_perturbation_s1_median": float(np.median(rel_per_trial)),
           "per_layer": {str(l): {"mean_norm": mean_norm[l], "rel_s1": float(np.mean(v))}
                         for l, v in sorted(per_layer.items())}}
    ref = S.REFERENCE_DIR / "norm_probe_llama3.json"
    if S.short_name(args.model) != "llama3" and ref.exists():
        r8 = load_json(ref)["rel_perturbation_s1_mean"]
        res["reference_rel_8b"] = r8
        res["s_match"] = r8 / res["rel_perturbation_s1_mean"]
    out = S.results_dir(args.model) / "norm_probe.json"
    save_json(res, out)
    if S.short_name(args.model) == "llama3":
        save_json(res, ref)
        print(f"Saved 8B reference to {ref} (8B reference for stage 3 at scale).")
    print({k: v for k, v in res.items() if k != "per_layer"})


def _targets_for(model_name, trials):
    """Targets from stage 2 if present; for the 8B reference, from the paper's log."""
    p = S.results_dir(model_name) / "targets.json"
    if p.exists():
        m = {t["key"]: t["target_epa"] for t in load_json(p)["trials"]}
    elif S.short_name(model_name) == "llama3" and S.REFERENCE_8B_LOG.exists():
        m = {f"{t['scenario_id']}|{t['pair']}": t["target_epa"]
             for t in load_json(S.REFERENCE_8B_LOG)["trials"]}
    else:
        raise FileNotFoundError(f"{p} missing: run stage 2 first.")
    return {t["key"]: m[t["key"]] for t in trials}


if __name__ == "__main__":
    main()
