"""
ACT target analysis (WS1.2, WS6.1 target-level audit, WS6.2 E-floor).

Uses only the logged inputs of experiment 08 (identity EPAs and the EPA read
from each user message), so it is CPU-only and exactly reproduces the targets
the steered runs used.

Outputs ``results/analysis/act_targets.json`` and figures:
- ``fig_target_distribution`` per-dimension histograms of b*
- ``fig_efloor`` effect of the Evaluation floor on targets

Usage::

    python -m examples.act_three.experiments.analysis.act_targets
"""

import numpy as np
import pandas as pd

from examples.act_three.act_core import (
    EPA,
    compute_act_target,
    get_default_coefficients,
    get_response_epa_for_deflection_minimization,
    has_behaviour_interactions,
)

from .common import (DIMS, DIM_SHORT, FIG_DIR, MODEL_LABEL, READING_LOGS,
                     ensure_dirs, load_hybrid_raw, load_json, save_json,
                     scenario_category)

E_FLOORS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _epa(d):
    return EPA(d["evaluation"], d["potency"], d["activity"])


def _r2_of_factor(y: np.ndarray, labels) -> float:
    labels = np.asarray(labels)
    means = pd.Series(y).groupby(labels).transform("mean").to_numpy()
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - means) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def analyse_model(model: str, coefs, rng: np.random.Generator) -> dict:
    raw = load_hybrid_raw(model)
    trials = raw["trials"]
    logged, legacy, closed, defl, defl1, active = [], [], [], [], [], []
    rows = []
    for t in trials:
        fG, fU, bU = _epa(t["agent_identity_epa"]), _epa(t["user_identity_epa"]), _epa(t["user_statement_epa"])
        logged.append([t["target_epa"][d] for d in DIMS])
        legacy.append(get_response_epa_for_deflection_minimization(fG, fU, bU, coefs).to_array())
        res = compute_act_target(fG, fU, bU, coefs, method="closed_form")
        closed.append(res["target"].to_array())
        defl.append(res["deflection"])
        defl1.append(res["deflection_event1"])
        active.append(res["constraint_active"])
        rows.append({"scenario_id": t["scenario_id"], "category": scenario_category(t["scenario_id"]),
                     "pair": t["pair"], "agent": t["agent_identity"], "user": t["user_identity"],
                     "fU_E": fU.e, "fU_P": fU.p, "fG_E": fG.e, "fG_P": fG.p})
    logged, legacy, closed = map(np.asarray, (logged, legacy, closed))
    meta = pd.DataFrame(rows)

    out = {"model": model, "n_trials": len(trials)}
    out["reproduce_logged_max_abs"] = float(np.abs(legacy - logged).max())
    out["closed_vs_lbfgsb_max_abs"] = float(np.abs(closed - legacy).max())
    out["closed_vs_lbfgsb_mean_abs"] = float(np.abs(closed - legacy).mean())
    out["bounds_active_frac"] = float(np.mean(active))
    out["deflection_event2_mean"] = float(np.mean(defl))
    out["deflection_event1_mean"] = float(np.mean(defl1))

    # Distribution of targets and how much they vary by situation
    dist = {}
    const = closed.mean(axis=0)
    for j, d in enumerate(DIMS):
        y = closed[:, j]
        dist[d] = {
            "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
            "min": float(y.min()), "max": float(y.max()),
            "q05": float(np.quantile(y, 0.05)), "q95": float(np.quantile(y, 0.95)),
            "r2_pair": _r2_of_factor(y, meta["pair"]),
            "r2_scenario": _r2_of_factor(y, meta["scenario_id"]),
            "r2_user": _r2_of_factor(y, meta["user"]),
            "r2_agent": _r2_of_factor(y, meta["agent"]),
            "mean_abs_dev_from_constant": float(np.mean(np.abs(y - const[j]))),
        }
    out["target_distribution"] = dist
    out["constant_target_B8"] = {d: float(const[j]) for j, d in enumerate(DIMS)}
    out["_targets"] = closed  # dropped before saving; used for figures

    # Reader noise propagation: perturb b_U by N(0, RMSE_d) and recompute b*
    rmse = _reader_rmse(model)
    if rmse is not None:
        shifts = []
        idx = rng.choice(len(trials), size=min(300, len(trials)), replace=False)
        for i in idx:
            t = trials[i]
            fG, fU = _epa(t["agent_identity_epa"]), _epa(t["user_identity_epa"])
            bU = np.array([t["user_statement_epa"][d] for d in DIMS])
            for _ in range(10):
                noisy = bU + rng.normal(0, rmse)
                b2 = compute_act_target(fG, fU, EPA.from_array(noisy), coefs)["target"].to_array()
                shifts.append(np.abs(b2 - closed[i]))
        shifts = np.asarray(shifts)
        out["reader_noise_propagation"] = {
            "reader_rmse": {d: float(rmse[j]) for j, d in enumerate(DIMS)},
            "mean_abs_target_shift": {d: float(shifts[:, j].mean()) for j, d in enumerate(DIMS)},
        }

    # E-floor guardrail (WS6.2)
    floors = {}
    for e_min in E_FLOORS:
        bf, df = [], []
        for t in trials:
            r = compute_act_target(_epa(t["agent_identity_epa"]), _epa(t["user_identity_epa"]),
                                   _epa(t["user_statement_epa"]), coefs, e_min=e_min)
            bf.append(r["target"].to_array())
            df.append(r["deflection"])
        bf = np.asarray(bf)
        binds = closed[:, 0] < e_min - 1e-9
        floors[str(e_min)] = {
            "binding_frac": float(binds.mean()),
            "mean_shift": {d: float(np.mean(bf[:, j] - closed[:, j])) for j, d in enumerate(DIMS)},
            "deflection_increase_mean": float(np.mean(np.asarray(df) - np.asarray(defl))),
            "deflection_increase_when_binding": float(np.mean((np.asarray(df) - np.asarray(defl))[binds])) if binds.any() else 0.0,
        }
    out["e_floor"] = floors

    # Target-level status audit (WS6.1, part 1): does ACT prescribe colder or
    # more dominant behaviour toward low-status users?
    meta = meta.assign(bE=closed[:, 0], bP=closed[:, 1], bA=closed[:, 2])
    per_user = meta.groupby("user").agg(fU_E=("fU_E", "first"), fU_P=("fU_P", "first"),
                                         bE=("bE", "mean"), bP=("bP", "mean"), bA=("bA", "mean"))
    out["per_user_targets"] = per_user.reset_index().to_dict(orient="records")
    out["status_corr"] = {
        "corr_userP_bE": float(np.corrcoef(meta["fU_P"], meta["bE"])[0, 1]),
        "corr_userP_bP": float(np.corrcoef(meta["fU_P"], meta["bP"])[0, 1]),
        "corr_userE_bE": float(np.corrcoef(meta["fU_E"], meta["bE"])[0, 1]),
    }
    # Pairs with role reversal: same two identities, agent and user swapped
    swaps = {}
    for a, b in [("boss", "subordinate"), ("teacher", "student"), ("doctor", "patient"),
                 ("counselor", "client"), ("mother", "daughter"), ("receptionist", "visitor")]:
        hi = meta[(meta.agent == a) & (meta.user == b)]
        lo = meta[(meta.agent == b) & (meta.user == a)]
        if len(hi) and len(lo):
            swaps[f"{a}->{b} minus {b}->{a}"] = {
                "dE": float(hi.bE.mean() - lo.bE.mean()),
                "dP": float(hi.bP.mean() - lo.bP.mean()),
            }
    out["role_swap_differences"] = swaps
    return out


def _reader_rmse(model: str):
    path = READING_LOGS.get(model)
    if path is None or not path.exists():
        return None
    tm = load_json(path)["test_metrics"]
    return np.array([tm[d]["rmse"] for d in DIMS])


def make_figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    models = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for j, d in enumerate(DIMS):
        ax = axes[j]
        for m in models:
            ax.hist(results[m]["_targets"][:, j], bins=30, alpha=0.55,
                    label=MODEL_LABEL[m], density=True)
        ax.set_title(f"Target {DIM_SHORT[d]} ($b^*_{DIM_SHORT[d]}$)", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Density", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_target_distribution.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for m in models:
        f = results[m]["e_floor"]
        xs = [float(k) for k in f]
        ax.plot(xs, [f[k]["binding_frac"] for k in f], marker="o", label=MODEL_LABEL[m])
    ax.set_xlabel("$e_{\\min}$", fontsize=8)
    ax.set_ylabel("Trials where floor binds", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_efloor.{ext}", dpi=200)
    plt.close(fig)


def run() -> dict:
    coefs = get_default_coefficients()
    rng = np.random.default_rng(0)
    results = {m: analyse_model(m, coefs, rng) for m in ("llama3", "ministral")}
    make_figures(results)
    payload = {"has_behaviour_interactions": has_behaviour_interactions(coefs)}
    for m, r in results.items():
        r = dict(r)
        r.pop("_targets")
        payload[m] = r
    save_json(payload, "act_targets.json")
    return payload


if __name__ == "__main__":
    import json
    res = run()
    for m in ("llama3", "ministral"):
        r = res[m]
        print(f"== {m}: reproduce={r['reproduce_logged_max_abs']:.2e} "
              f"closed-vs-lbfgsb={r['closed_vs_lbfgsb_max_abs']:.2e} "
              f"bounds-active={r['bounds_active_frac']:.3f}")
        print(json.dumps(r["target_distribution"], indent=1))
        print(json.dumps(r.get("reader_noise_propagation"), indent=1))
        print(json.dumps(r["e_floor"], indent=1))
        print(json.dumps(r["status_corr"], indent=1))
        print(json.dumps(r["role_swap_differences"], indent=1))
