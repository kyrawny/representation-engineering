"""
Independent-measure analyses for the revision (plan T9, judge-free version).

The LLM-judge parts of T9 are skipped (no judge available; see BLOCKERS.md).  This script computes

1. measure agreement matrix (own reader, cross-model reader, Mistral-7B, VADER) + validity on the
   130-utterance calibration set (reader test metrics from the tuning files; VADER computed here)
2. Table 3 under each measure (distance, tracking slope, CCC, paired delta vs unsteered, Holm, MDE)
3. what ACT adds: paired contrasts of ``repe_act`` against non-ACT targets and external baselines
4. role-reversal test (Delta P between high- and low-status agents, observed vs ACT target)
5. are-you-sure benchmark table (from grade_are_you_sure.py)
6. specificity: is each condition closer to its own ACT target or to the constant mean target?

    python -m examples.act_three.experiments.analysis.revision_results
"""
import itertools
import json

import numpy as np
import pandas as pd

from . import rescore_data as RD
from .common import (FIG_DIR, OUT_DIR, REPO_ROOT, ensure_dirs, load_json, save_json)
from .metrics import cluster_bootstrap, design_effect, holm, lin_ccc, mde_paired, tracking_fit

N_BOOT = 2000
ORIG_CONDS = ["unsteered", "pe_only", "repe_only", "hybrid"]
DIMS = ["E", "P", "A"]
MEASURES = ["orig", "own", "cross", "m7b"]
SWAPS = [("counselor_client", "client_counselor"), ("boss_subordinate", "subordinate_boss"),
         ("teacher_student", "student_teacher"), ("doctor_patient", "patient_doctor"),
         ("mother_daughter", "daughter_mother"), ("receptionist_visitor", "visitor_receptionist")]
PRIMARY = ["repe_const", "repe_shuffled", "caa", "antisyc_prompt"]


def dims_for(measure: str):
    return ["E", "A"] if measure == "m7b" else DIMS


def piv(df: pd.DataFrame, measure: str, conds):
    """Wide table: one row per trial, columns <cond>_<dim> (observed) and target_<dim>."""
    d = df[df.cond.isin(conds)].copy()
    d["tid"] = RD.trial_id(d)
    cols = {}
    for dm in dims_for(measure):
        p = d.pivot_table(index="tid", columns="cond", values=f"{measure}_{dm}", aggfunc="first")
        for c in p.columns:
            cols[f"{c}_{dm}"] = p[c]
    W = pd.DataFrame(cols)
    meta = d.drop_duplicates("tid").set_index("tid")[["scenario_id", "pair", "category"] + [f"target_{x}" for x in DIMS]]
    meta["any_trunc"] = d.assign(tr=d.text500.str.len() >= 500).groupby("tid").tr.max()
    return meta.join(W)


def dist(W, cond, dm):
    return (W[f"{cond}_{dm}"] - W[f"target_{dm}"]).abs()


def overall(W, cond, dms):
    return np.mean([dist(W, cond, dm).to_numpy() for dm in dms], axis=0)


def paired(W: pd.DataFrame, a: str, b: str, measure: str) -> dict:
    """Delta = dist(b) - dist(a) (negative: b is closer to the ACT target)."""
    dms = dims_for(measure)
    need = [f"{c}_{dm}" for c in (a, b) for dm in dms]
    if not all(n in W.columns for n in need):
        return {}
    S = W.dropna(subset=need)
    if len(S) < 30:
        return {}
    cl = S.scenario_id.to_numpy()
    out = {"n": int(len(S))}
    for dm in dms + ["overall"]:
        delta = (overall(S, b, dms) - overall(S, a, dms)) if dm == "overall" else \
            (dist(S, b, dm) - dist(S, a, dm)).to_numpy()
        ci = cluster_bootstrap(cl, lambda ix: delta[ix].mean(), N_BOOT)
        deff = design_effect(delta, cl)
        out[dm] = {"delta": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p_two_sided"],
                   "mde": mde_paired(float(delta.std(ddof=1)), len(S) / deff), "design_effect": deff,
                   "mean_dist_a": float(overall(S, a, dms).mean() if dm == "overall" else dist(S, a, dm).mean()),
                   "mean_dist_b": float(overall(S, b, dms).mean() if dm == "overall" else dist(S, b, dm).mean())}
    return out


def track(W, cond, measure):
    res = {}
    for dm in dims_for(measure):
        if f"{cond}_{dm}" not in W.columns:
            continue
        S = W.dropna(subset=[f"{cond}_{dm}"])
        if len(S) < 30:
            continue
        e, t = S[f"{cond}_{dm}"].to_numpy(), S[f"target_{dm}"].to_numpy()
        fit = tracking_fit(e, t)
        cl = S.scenario_id.to_numpy()
        b1 = cluster_bootstrap(cl, lambda ix: tracking_fit(e[ix], t[ix])["b1"], 500)
        res[dm] = {**fit, "b1_lo": b1["lo"], "b1_hi": b1["hi"], "ccc": lin_ccc(e, t), "n": int(len(S))}
    return res


def specificity(W, cond, measure):
    """dist to own target minus dist to the constant mean target (negative = closer to own target)."""
    res = {}
    for dm in dims_for(measure):
        if f"{cond}_{dm}" not in W.columns:
            continue
        S = W.dropna(subset=[f"{cond}_{dm}"])
        if len(S) < 30:
            continue
        e, t = S[f"{cond}_{dm}"].to_numpy(), S[f"target_{dm}"].to_numpy()
        res[dm] = float((np.abs(e - t) - np.abs(e - t.mean())).mean())
    return res


# --------------------------------------------------------------------------- 1. agreement
def agreement(frames: dict) -> dict:
    out = {}
    for gen, df in frames.items():
        d = df[df.source == "08"]
        res = {}
        for dm in DIMS:
            cols = {"orig": f"orig_{dm}", "own": f"own_{dm}", "cross": f"cross_{dm}", "m7b": f"m7b_{dm}"}
            if dm == "E":
                cols["vader"] = "vader_E"
            cols = {k: v for k, v in cols.items() if v in d and d[v].notna().sum() > 30}
            M = {}
            for a, b in itertools.combinations(cols, 2):
                S = d[[cols[a], cols[b]]].dropna()
                M[f"{a}~{b}"] = {"r": float(np.corrcoef(S.iloc[:, 0], S.iloc[:, 1])[0, 1]),
                                 "ccc": lin_ccc(S.iloc[:, 0], S.iloc[:, 1]), "n": int(len(S))}
            # agreement of each measure with the ACT target itself, across all conditions pooled
            for a in cols:
                S = d[[cols[a], f"target_{dm}"]].dropna()
                M[f"{a}~target"] = {"r": float(np.corrcoef(S.iloc[:, 0], S.iloc[:, 1])[0, 1]), "n": int(len(S))}
            res[dm] = M
        out[gen] = res
    # validity on the 130-utterance calibration test set (ElasticNet readers; metrics_test of each tuning file)
    val = {}
    for gen_dir, rn in (("llama3", "llama3"), ("ministral", "ministral"), ("mistral-7b-instruct-v0-3", "mistral7b")):
        p = REPO_ROOT / "examples" / "act_three" / "models" / gen_dir / "epa_reading_tuning_v2_results.json"
        if p.exists():
            en = load_json(p)["methods"]["ElasticNet"]
            val[rn] = {d: {m: en[d]["metrics_test"].get(m) for m in ("pearson", "mae", "rmse")} | {"n_layers": en[d]["n_layers"]}
                       for d in ("evaluation", "potency", "activity")}
    out["reader_calibration_test"] = val
    q = REPO_ROOT / "examples" / "act_three" / "experiments" / "results" / "01_reading_quality.json"
    if q.exists():
        out["table1_llama_source_01_reading_quality"] = {d: {m: v[m] for m in ("pearson_r", "mae")}
                                                         for d, v in load_json(q)["test_metrics"].items()}
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        a = SentimentIntensityAnalyzer()
        items = load_json(REPO_ROOT / "examples" / "act_three" / "models" / "llama3" / "epa_tuning_dataset_test.json")["utterances"]
        vs = np.array([a.polarity_scores(x["text"])["compound"] * 4.3 for x in items])
        te = np.array([x["target_epa"]["e"] for x in items])
        out["vader_calibration_E"] = {"r": float(np.corrcoef(vs, te)[0, 1]), "mae": float(np.mean(np.abs(vs - te))),
                                      "n": len(vs)}
    except Exception as e:  # keep the pipeline going; report why
        out["vader_calibration_E"] = {"error": str(e)[:200]}
    return out


# --------------------------------------------------------------------------- 2. Table 3
def _t3(W, m) -> dict:
    dms = dims_for(m)
    res = {"n_trials": int(len(W)), "conds": {}, "paired_vs_unsteered": {}}
    for c in ORIG_CONDS:
        res["conds"][c] = {"distance": {dm: float(dist(W.dropna(subset=[f"{c}_{dm}"]), c, dm).mean()) for dm in dms},
                           "overall": float(np.nanmean(overall(W, c, dms))),
                           "tracking": track(W, c, m)}
    ps, ks = [], []
    for c in ORIG_CONDS[1:]:
        r = paired(W, "unsteered", c, m)
        res["paired_vs_unsteered"][c] = r
        for dm in dms:
            if dm in r:
                ps.append(r[dm]["p"]); ks.append((c, dm))
    for (c, dm), pa in zip(ks, holm(ps)):
        res["paired_vs_unsteered"][c][dm]["p_holm"] = float(pa)
    return res


def table3(frames: dict) -> dict:
    out = {}
    for gen, df in frames.items():
        out[gen] = {}
        for m in MEASURES:
            if f"{m}_E" not in df or df[f"{m}_E"].notna().sum() < 100:
                continue
            W = piv(df[df.source == "08"], m, ORIG_CONDS)
            out[gen][m] = _t3(W, m)
            # sensitivity: trials in which none of the four texts reached the 500-character logging limit
            out[gen][m]["untruncated_trials"] = _t3(W[~W.any_trunc.astype(bool)], m)
    return out


# --------------------------------------------------------------------------- 3. what ACT adds
def act_ablation(frames: dict) -> dict:
    out = {}
    for gen, df in frames.items():
        have = set(df.cond.unique())
        if "repe_const" not in have:
            continue
        out[gen] = {}
        for m in ("orig", "own", "cross", "m7b"):
            if f"{m}_E" not in df or df[f"{m}_E"].notna().sum() < 100:
                continue
            allc = sorted(have)
            W = piv(df, m, allc)
            res = {"contrasts": {}, "tracking": {}, "specificity": {}}
            # reference = the logged ACT-RepE condition: it reproduces bit for bit with orthogonalised directions
            ref = "repe_only"
            res["reference"] = ref
            if ref:
                fam = [c for c in PRIMARY if c in have]
                sec = ["repe_idonly", "unsteered", "pe_only", "repe_act_nonortho", "repe_const_nonortho",
                       "repe_shuffled_nonortho", "repe_idonly_nonortho"]
                for c in fam + [c for c in sec if c in have]:
                    res["contrasts"][f"{ref}__vs__{c}"] = paired(W, ref, c, m)
                for c in ("hybrid_act", "hybrid_const"):
                    pass
                for dm in dims_for(m) + ["overall"]:
                    ps = [(c, res["contrasts"][f"{ref}__vs__{c}"][dm]["p"]) for c in fam
                          if res["contrasts"].get(f"{ref}__vs__{c}") and dm in res["contrasts"][f"{ref}__vs__{c}"]]
                    for (c, _), pa in zip(ps, holm([p for _, p in ps])):
                        res["contrasts"][f"{ref}__vs__{c}"][dm]["p_holm_primary_family"] = float(pa)
            # (the repe_only vs repe_const contrast above already uses all 1,300 trials for the const condition's overlap)
            if "hybrid_const" in have and "hybrid" in have:
                res["contrasts"]["hybrid_logged__vs__hybrid_const"] = paired(W, "hybrid", "hybrid_const", m)
            for c in allc:
                res["tracking"][c] = track(W, c, m)
                res["specificity"][c] = specificity(W, c, m)
            res["n_trials_by_cond"] = {c: int(W[f"{c}_{dims_for(m)[0]}"].notna().sum()) for c in allc if f"{c}_{dims_for(m)[0]}" in W.columns}
            out[gen][m] = res
    return out


# --------------------------------------------------------------------------- 4. role reversal
def role_reversal(frames: dict) -> dict:
    out = {}
    for gen, df in frames.items():
        out[gen] = {}
        for m in ("orig", "own", "cross"):
            if f"{m}_P" not in df or df[f"{m}_P"].notna().sum() < 100:
                continue
            res = {}
            # scenarios on which every condition has all 13 pairs (so conditions are compared on identical trials)
            cnt = df.groupby(["cond", "scenario_id"]).pair.nunique().unstack(0)
            for cond in sorted(df.cond.unique()):
                d = df[df.cond == cond]
                sids = sorted(s for s in d.scenario_id.unique() if cnt.loc[s, cond] == 13) if cond in cnt else []
                res.setdefault("_scenarios", {})[cond] = len(sids)
            full = [c for c in df.cond.unique() if (df.cond == c).sum() >= 600]
            common = set.intersection(*[set(s for s in df[df.cond == c].scenario_id.unique() if cnt.loc[s, c] == 13)
                                        for c in full])
            for cond in sorted(full):
                d = df[(df.cond == cond) & df.scenario_id.isin(common)].set_index(["scenario_id", "pair"])
                rows = []
                for sid in sorted(common):
                    for hi, lo in SWAPS:
                        try:
                            rows.append({"sid": sid, "grp": hi,
                                         "dP_obs": d.loc[(sid, hi), f"{m}_P"] - d.loc[(sid, lo), f"{m}_P"],
                                         "dP_target": d.loc[(sid, hi), "target_P"] - d.loc[(sid, lo), "target_P"]})
                        except KeyError:
                            pass
                r = pd.DataFrame(rows).dropna()
                if len(r) < 30:
                    continue
                cl = r.sid.to_numpy()
                ci = cluster_bootstrap(cl, lambda ix: r.dP_obs.to_numpy()[ix].mean(), N_BOOT)
                fit = tracking_fit(r.dP_obs.to_numpy(), r.dP_target.to_numpy())
                res[cond] = {"dP_obs": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "dP_target": float(r.dP_target.mean()),
                             "ratio_obs_over_target": float(ci["est"] / r.dP_target.mean()),
                             "slope_vs_target_dP": fit["b1"], "n": int(len(r)), "n_scenarios": len(common)}
            out[gen][m] = res
    return out


# --------------------------------------------------------------------------- frontier rows
def frontier() -> dict:
    """Gemini generations (stored in the repo) vs Llama conditions, all scored by readers independent of both.

    Gemini trials share the experiment-08 (Llama) targets.  Claude Sonnet 5 generations are not in the repository.
    """
    g = load_json(RD.REV.parent / "gemini" / "13_gemini_generations.json")["trials"]
    tgt = {f"{t['scenario_id']}|{t['pair']}": t["target_epa"] for t in g}
    out = {}
    for reader in ("ministral", "mistral7b"):
        rs = RD._rescore(reader)
        rs = rs.drop_duplicates(["generator", "key"]).copy()
        rs["tid"] = rs.key.str.rsplit("|", n=1).str[0]
        rs["cond"] = rs.key.str.rsplit("|", n=1).str[1]
        rs = rs[rs.generator.isin(["gemini", "llama3"]) & ~rs.cond.str.contains("nonortho")]
        dms = ["E", "A"] if reader == "mistral7b" else DIMS
        names = {"E": "evaluation", "P": "potency", "A": "activity"}
        res = {}
        for (gen, cond), grp in rs.groupby(["generator", "cond"]):
            if len(grp) < 600 or grp.tid.map(lambda x: x in tgt).mean() < 1:
                continue
            d = np.mean([np.abs(grp[dm] - grp.tid.map(lambda x: tgt[x][names[dm]])) for dm in dms], axis=0)
            res[f"{gen}|{cond}"] = {"overall": float(np.nanmean(d)), "n": int(len(grp)),
                                    **{dm: float(np.nanmean(np.abs(grp[dm] - grp.tid.map(lambda x: tgt[x][names[dm]])))) for dm in dms}}
        out[reader] = res
    return out


# --------------------------------------------------------------------------- driver
def are_you_sure() -> dict:
    p = RD.REV / "llama3" / "16_are_you_sure_graded.json"
    if not p.exists():
        return {}
    j = load_json(p)
    j.pop("per_item", None)
    return j


def run() -> dict:
    ensure_dirs()
    frames = {}
    for gen in ("llama3", "ministral"):
        frames[gen] = RD.build(gen)
    res = {"agreement": agreement(frames), "table3": table3(frames), "act_ablation": act_ablation(frames),
           "role_reversal": role_reversal(frames), "are_you_sure": are_you_sure(), "frontier": frontier(),
           "n_texts": {g: {"total": int(len(f)), "by_cond": f.cond.value_counts().to_dict()} for g, f in frames.items()}}
    save_json(res, "revision_results.json")
    make_figures(res)
    return res


def make_figures(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ens = res.get("act_ablation", {}).get("llama3", {})
    for m, tag in (("cross", "independent (Ministral reader)"), ("own", "own reader, text500")):
        r = ens.get(m)
        if not r:
            continue
        keys = [k for k in r["contrasts"] if k.startswith("repe_only__vs__") and r["contrasts"][k]]
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 0.45 * len(keys) + 1.0))
        for i, k in enumerate(keys):
            o = r["contrasts"][k]["overall"]
            ax.errorbar(o["delta"], i, xerr=[[o["delta"] - o["lo"]], [o["hi"] - o["delta"]]], fmt="o", color="C0")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.split("__vs__")[1] for k in keys], fontsize=8)
        ax.set_xlabel("distance to ACT target: condition minus ACT-RepE\n(positive = ACT-RepE closer)", fontsize=8)
        ax.set_title(f"Llama-3.1-8B, {tag}", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"fig_act_ablation_{m}.{ext}", dpi=200)
        plt.close(fig)
    rr = res.get("role_reversal", {}).get("llama3", {}).get("cross")
    if rr:
        conds = [c for c in rr if not c.startswith("_")]
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        for i, c in enumerate(conds):
            ax.errorbar(i, rr[c]["dP_obs"], yerr=[[rr[c]["dP_obs"] - rr[c]["lo"]], [rr[c]["hi"] - rr[c]["dP_obs"]]], fmt="o")
        ax.axhline(np.mean([rr[c]["dP_target"] for c in conds]), color="k", ls="--", lw=0.8, label="ACT target $\\Delta P$")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("$\\Delta P$ (high-status minus low-status agent)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"fig_role_reversal.{ext}", dpi=200)
        plt.close(fig)
    ays = res.get("are_you_sure", {}).get("conditions")
    if ays:
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        cs = list(ays)
        for i, c in enumerate(cs):
            v = ays[c]
            if v["flip_rate"] is not None:
                ax.errorbar(i, v["flip_rate"], yerr=[[v["flip_rate"] - v["flip_rate_ci"][0]], [v["flip_rate_ci"][1] - v["flip_rate"]]], fmt="o")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels(cs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("flip rate after pushback", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"fig_are_you_sure.{ext}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in r.items() if k != "n_texts"}, default=str)[:600])
