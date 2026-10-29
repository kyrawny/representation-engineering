"""
Tracking and specificity analysis of the existing 1,300-trial runs (WS2.3).

All numbers use the *same-model* reader (metric M-S in the revision plan);
they are diagnostic only until rescored with an independent reader (M-X)
or LLM judge (M-J).

For every model x condition x dimension:
- tracking slope b1 / intercept b0 of observed on target EPA, Pearson r
- Lin's CCC
- mean |observed - target| (the original metric)
- specificity: distance to own target minus distance to (a) a constant
  target (mean b*) and (b) the target of a randomly shuffled identity pair
  for the same scenario.  Negative = outputs are closer to their own,
  situation-specific target.
Paired steered-minus-unsteered differences use a cluster bootstrap over
scenarios; a design-effect-adjusted minimum detectable effect (MDE) and the
reader's own test-set MAE are reported as noise references.

Usage::

    python -m examples.act_three.experiments.analysis.tracking
"""

import numpy as np
import pandas as pd

from .common import (CONDITIONS, DIMS, READING_LOGS, load_hybrid_trials, load_json,
                     save_json)
from .metrics import cluster_bootstrap, design_effect, holm, lin_ccc, mde_paired, tracking_fit

N_BOOT = 2000


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial with columns target_<dim>, <cond>_<dim>."""
    base = df[df.condition == "unsteered"].pivot_table(
        index=["trial", "scenario_id", "category", "pair"], columns="dim", values="target")
    base.columns = [f"target_{c}" for c in base.columns]
    obs = df.pivot_table(index="trial", columns=["condition", "dim"], values="observed")
    obs.columns = [f"{c}_{d}" for c, d in obs.columns]
    return base.reset_index().merge(obs.reset_index(), on="trial")


def analyse_model(model: str, rng: np.random.Generator) -> dict:
    df = load_hybrid_trials(model)
    # Integrity: recomputed |obs - target| must equal the logged distance
    integrity = float(np.abs(df.distance - df.logged_distance).max())
    W = _wide(df)
    clusters = W.scenario_id.to_numpy()
    n = len(W)

    # Shuffled-pair target: same scenario, different identity pair
    shuffled_idx = np.empty(n, dtype=int)
    for sid, grp in W.groupby("scenario_id"):
        idx = grp.index.to_numpy()
        perm = idx.copy()
        while len(idx) > 1 and np.any(perm == idx):
            perm = rng.permutation(idx)
        shuffled_idx[idx] = perm
    const = {d: W[f"target_{d}"].mean() for d in DIMS}

    out = {"model": model, "n_trials": n, "integrity_max_abs": integrity,
           "conditions": {}, "paired_vs_unsteered": {}}
    for cond in CONDITIONS:
        cres = {}
        for d in DIMS:
            t = W[f"target_{d}"].to_numpy()
            e = W[f"{cond}_{d}"].to_numpy()
            fit = tracking_fit(e, t)
            b1_ci = cluster_bootstrap(clusters, lambda ix: tracking_fit(e[ix], t[ix])["b1"], N_BOOT)
            ccc_ci = cluster_bootstrap(clusters, lambda ix: lin_ccc(e[ix], t[ix]), N_BOOT)
            dist_own = np.abs(e - t)
            dist_const = np.abs(e - const[d])
            dist_shuf = np.abs(e - t[shuffled_idx])
            cres[d] = {
                **fit,
                "b1_lo": b1_ci["lo"], "b1_hi": b1_ci["hi"],
                "ccc": ccc_ci["est"], "ccc_lo": ccc_ci["lo"], "ccc_hi": ccc_ci["hi"],
                "mean_distance": float(dist_own.mean()),
                "sd_observed": float(e.std(ddof=1)),
                "sd_target": float(t.std(ddof=1)),
                "spec_vs_constant": float((dist_own - dist_const).mean()),
                "spec_vs_shuffled": float((dist_own - dist_shuf).mean()),
            }
        overall = np.mean([np.abs(W[f"{cond}_{d}"] - W[f"target_{d}"]) for d in DIMS], axis=0)
        cres["overall_mean_distance"] = float(overall.mean())
        out["conditions"][cond] = cres

    # Paired differences vs unsteered (negative = closer to target)
    pvals, keys = [], []
    for cond in ("pe_only", "repe_only", "hybrid"):
        pres = {}
        for d in DIMS + ["overall"]:
            if d == "overall":
                delta = np.mean([np.abs(W[f"{cond}_{x}"] - W[f"target_{x}"])
                                 - np.abs(W[f"unsteered_{x}"] - W[f"target_{x}"]) for x in DIMS], axis=0)
            else:
                delta = (np.abs(W[f"{cond}_{d}"] - W[f"target_{d}"])
                         - np.abs(W[f"unsteered_{d}"] - W[f"target_{d}"])).to_numpy()
            ci = cluster_bootstrap(clusters, lambda ix: delta[ix].mean(), N_BOOT)
            deff = design_effect(delta, clusters)
            n_eff = n / deff
            pres[d] = {"delta": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p_boot": ci["p_two_sided"],
                       "sd": float(delta.std(ddof=1)), "design_effect": deff, "n_eff": n_eff,
                       "mde": mde_paired(float(delta.std(ddof=1)), n_eff),
                       "d_z": float(delta.mean() / delta.std(ddof=1))}
            if d != "overall":
                pvals.append(ci["p_two_sided"])
                keys.append((cond, d))
        out["paired_vs_unsteered"][cond] = pres
    for (cond, d), p in zip(keys, holm(pvals)):
        out["paired_vs_unsteered"][cond][d]["p_holm"] = float(p)

    path = READING_LOGS.get(model)
    if path is not None and path.exists():
        tm = load_json(path)["test_metrics"]
        out["reader_test"] = {d: {"r": tm[d]["pearson_r"], "mae": tm[d]["mae"], "rmse": tm[d]["rmse"]}
                              for d in DIMS}
    return out


def run() -> dict:
    rng = np.random.default_rng(0)
    res = {m: analyse_model(m, rng) for m in ("llama3", "ministral")}
    save_json(res, "tracking.json")
    return res


if __name__ == "__main__":
    for m, r in run().items():
        print(f"===== {m}  integrity={r['integrity_max_abs']:.1e}")
        for c, cr in r["conditions"].items():
            print(c, "overall", round(cr["overall_mean_distance"], 3))
            for d in DIMS:
                x = cr[d]
                print(f"   {d[:3]} b1={x['b1']:.2f} [{x['b1_lo']:.2f},{x['b1_hi']:.2f}] r={x['r']:.2f} "
                      f"ccc={x['ccc']:.2f} dist={x['mean_distance']:.3f} sdObs={x['sd_observed']:.2f} "
                      f"sdT={x['sd_target']:.2f} specC={x['spec_vs_constant']:+.3f} specS={x['spec_vs_shuffled']:+.3f}")
        for c, pr in r["paired_vs_unsteered"].items():
            for d, x in pr.items():
                print(f"  {c} {d}: delta={x['delta']:+.3f} [{x['lo']:+.3f},{x['hi']:+.3f}] "
                      f"p_holm={x.get('p_holm', float('nan')):.3g} mde={x['mde']:.3f} deff={x['design_effect']:.2f}")
