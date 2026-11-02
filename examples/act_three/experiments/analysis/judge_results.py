"""
Analyse the LLM-judge outputs returned from ``experiments/judge/outputs/<model>/`` (plan T8.2 summary, T9.3, T9.4,
T9.7 and the judge parts of T10).  Run on the machine that has the experiment outputs (it merges judge scores
with ACT targets, reader scores and benchmark grades).

    python -m examples.act_three.experiments.analysis.judge_results --model gemini-3.6-flash

Writes ``results/analysis/judge_validation.json``, ``results/analysis/judge_results.json`` and
``results/analysis/figures/fig_pareto_appropriate_vs_nonsyc.{pdf,png}``.  Items that failed validation
(r < 0.4 vs humans, or CI including 0) are flagged ``validated: false`` and must be reported as secondary only.
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import rescore_data as RD
from .common import EXPERIMENTS_DIR, FIG_DIR, OUT_DIR, ensure_dirs, load_json, save_json
from .metrics import cluster_bootstrap, design_effect, holm, mde_paired

JUDGE_DIR = EXPERIMENTS_DIR / "judge" / "outputs"
REV = EXPERIMENTS_DIR / "results" / "revision"
ITEMS = list("abcdefgh")
ITEM_NAME = {"a": "appropriate", "b": "natural", "c": "non_sycophantic", "d": "validates_flatters", "e": "gives_in",
             "f": "assertive", "g": "harsh", "h": "addresses_concern"}
PRIMARY = ["repe_const", "repe_shuffled", "caa", "antisyc_prompt"]
SECONDARY = ["repe_idonly", "unsteered", "pe_only", "hybrid"]
N_BOOT = 2000


def slug(m):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", m)


def rjsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def j1_frame(d, gen):
    rows = rjsonl(d / f"j1_{gen}.jsonl")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{"key": r["key"], "trial": r["trial"], "cond": r["condition"], "position": r["position"],
                        "n_responses": r["n_responses"], **r["scores"]} for r in rows])
    df["scenario_id"] = df.trial.str.split("|").str[0]
    return df


def paired(W, ref, cond, item):
    S = W[[f"{ref}_{item}", f"{cond}_{item}", "scenario_id"]].dropna()
    if len(S) < 30:
        return None
    delta = (S[f"{cond}_{item}"] - S[f"{ref}_{item}"]).to_numpy(float)
    cl = S.scenario_id.to_numpy()
    ci = cluster_bootstrap(cl, lambda ix: delta[ix].mean(), N_BOOT)
    deff = design_effect(delta, cl)
    return {"n": int(len(S)), "delta_cond_minus_ref": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p_two_sided"],
            "mde": mde_paired(float(delta.std(ddof=1)), len(S) / deff),
            "mean_ref": float(S[f"{ref}_{item}"].mean()), "mean_cond": float(S[f"{cond}_{item}"].mean())}


def wide(df):
    W = df.pivot_table(index=["trial", "scenario_id"], columns="cond", values=ITEMS)
    W.columns = [f"{c}_{i}" for i, c in W.columns]
    return W.reset_index()


def j1_analysis(d, validated):
    out = {}
    for gen in ("llama3", "ministral"):
        df = j1_frame(d, gen)
        if df.empty:
            continue
        W = wide(df)
        conds = sorted(df.cond.unique())
        res = {"n_trials": int(df.trial.nunique()), "condition_means": {}, "contrasts_vs_repe_only": {},
               "contrasts_vs_unsteered": {}, "position_effect": {}}
        for c in conds:
            sub = df[df.cond == c]
            res["condition_means"][c] = {it: {"mean": float(sub[it].mean()),
                                              **{k: v for k, v in cluster_bootstrap(sub.scenario_id.to_numpy(),
                                                 lambda ix, x=sub[it].to_numpy(float): x[ix].mean(), N_BOOT).items()
                                                 if k in ("lo", "hi")}} for it in ITEMS}
        # primary family (plan T9.3): ACT-RepE vs non-ACT targets / baselines, Holm within each item over 4 contrasts
        for it in ITEMS:
            fam = [(c, paired(W, "repe_only", c, it)) for c in PRIMARY if c in conds]
            fam = [(c, r) for c, r in fam if r]
            for (c, r), ph in zip(fam, holm([r["p"] for _, r in fam]) if fam else []):
                r["p_holm"] = float(ph)
                r["family"] = "primary" if it in ("a", "c") else "secondary_item"
                res["contrasts_vs_repe_only"].setdefault(c, {})[it] = r
            for c in SECONDARY:
                if c in conds and c != "repe_only":
                    r = paired(W, "repe_only", c, it)
                    if r:
                        r["family"] = "secondary"
                        res["contrasts_vs_repe_only"].setdefault(c, {})[it] = r
            fam2 = [(c, paired(W, "unsteered", c, it)) for c in ("pe_only", "repe_only", "hybrid")]
            fam2 = [(c, r) for c, r in fam2 if r]
            for (c, r), ph in zip(fam2, holm([r["p"] for _, r in fam2])):
                r["p_holm"] = float(ph)
                res["contrasts_vs_unsteered"].setdefault(c, {})[it] = r
        # presentation-position check: mean score by position (should be flat)
        res["position_effect"] = {it: df.groupby("position")[it].mean().round(3).to_dict() for it in ("a", "c")}
        if gen == "llama3":
            res["human_subsets"] = human_subsets(df, W)
        res["validated_items"] = validated
        out[gen] = res
    return out


def human_subsets(df, W):
    """RepE minus unsteered on the 1,300-trial run, restricted to the human study's 14 scenario-pair cells and to its
    five identity pairs, to see whether the human vignettes are representative of the full run."""
    hv = rjsonl(EXPERIMENTS_DIR / "judge" / "inputs" / "human_vignettes.jsonl")
    tr = {r["trial"]: (r["agent"], r["user"]) for r in rjsonl(EXPERIMENTS_DIR / "judge" / "inputs" / "j1_trials_llama3.jsonl")}
    cells = {f"{v['vig']}|{v['agent']}_{v['user']}" for v in hv}
    pairs = {(v["agent"], v["user"]) for v in hv}
    W = W.assign(_pair=W.trial.map(tr))
    out = {}
    for name, mask in (("vignette_cells", W.trial.map(lambda t: any(t.split("|")[0] == c.split("|")[0] and tr[t] == tuple(c.split("|")[1].split("_", 1)) for c in cells))),
                       ("study_pairs", W._pair.isin(pairs)), ("other_pairs", ~W._pair.isin(pairs))):
        S = W[mask]
        res = {"n_trials": int(len(S))}
        for it in ("a", "b", "c"):
            delta = (S[f"repe_only_{it}"] - S[f"unsteered_{it}"]).to_numpy(float)
            ci = cluster_bootstrap(S.scenario_id.to_numpy(), lambda ix: delta[ix].mean(), N_BOOT)
            res[it] = {"delta_repe_minus_unsteered": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p_two_sided"]}
        out[name] = res
    return out


def pareto_figure(j1, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = j1.get("llama3", {}).get("condition_means")
    if not m:
        return
    human = load_json(OUT_DIR / "human_descriptives.json")["means"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    gvals = [m[c]["g"]["mean"] for c in m]
    norm = plt.Normalize(min(gvals), max(gvals))
    for c, v in m.items():
        x, y = v["a"], v["c"]
        ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"] - x["lo"]], [x["hi"] - x["mean"]]],
                    yerr=[[y["mean"] - y["lo"]], [y["hi"] - y["mean"]]], fmt="none", ecolor="0.6", lw=0.8)
        ax.scatter(x["mean"], y["mean"], s=20 + 25 * v["b"]["mean"], c=[plt.cm.viridis(norm(v["g"]["mean"]))], zorder=3)
        ax.annotate(c, (x["mean"], y["mean"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    for c in ("unsteered", "pe_only", "repe_only", "hybrid"):
        ax.scatter(human["appropriate"][c], human["non_sycophantic"][c], s=40, facecolors="none", edgecolors="k", zorder=4)
        ax.annotate(f"{c} (human)", (human["appropriate"][c], human["non_sycophantic"][c]), fontsize=5, xytext=(3, -8),
                    textcoords="offset points")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    fig.colorbar(sm, ax=ax).set_label("judge: harsh (g)", fontsize=7)
    ax.set_xlabel("appropriateness (1-7)", fontsize=8)
    ax.set_ylabel("non-sycophancy (1-7)", fontsize=8)
    ax.set_title(f"Llama-3.1-8B, judge {model}; size = naturalness", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_pareto_appropriate_vs_nonsyc.{ext}", dpi=200)
    plt.close(fig)


def j2_analysis(d):
    out = {}
    for gen in ("llama3", "ministral"):
        rows = rjsonl(d / f"j2_{gen}.jsonl")
        if not rows:
            continue
        J = pd.DataFrame(rows).rename(columns={"E": "judge_E", "P": "judge_P", "A": "judge_A"})[["key", "judge_E", "judge_P", "judge_A"]]
        df = RD.build(gen, with_vader=True).merge(J, on="key", how="inner")
        res = {"n_texts": int(len(df)), "agreement": {}, "distance_by_condition": {}, "paired_vs_unsteered": {},
               "paired_vs_repe_only": {}}
        for dm in "EPA":
            for m in ("orig", "own", "cross", "m7b", "vader"):
                col = f"{m}_{dm}"
                if col in df and df[col].notna().sum() > 30:
                    S = df[[col, f"judge_{dm}"]].dropna()
                    res["agreement"][f"judge~{m}_{dm}"] = float(np.corrcoef(S.iloc[:, 0], S.iloc[:, 1])[0, 1])
            S = df[[f"judge_{dm}", f"target_{dm}"]].dropna()
            res["agreement"][f"judge~target_{dm}"] = float(np.corrcoef(S.iloc[:, 0], S.iloc[:, 1])[0, 1])
        df["dist"] = np.mean([np.abs(df[f"judge_{dm}"] - df[f"target_{dm}"]) for dm in "EPA"], axis=0)
        res["distance_by_condition"] = df.groupby("cond").dist.mean().round(4).to_dict()
        W = df.pivot_table(index=["scenario_id", "pair"], columns="cond", values="dist").reset_index()
        for ref, cs, key in (("unsteered", ["pe_only", "repe_only", "hybrid"], "paired_vs_unsteered"),
                             ("repe_only", PRIMARY + ["repe_idonly"], "paired_vs_repe_only")):
            fam = []
            for c in cs:
                if c in W and ref in W:
                    S = W[[ref, c, "scenario_id"]].dropna()
                    delta = (S[c] - S[ref]).to_numpy(float)
                    ci = cluster_bootstrap(S.scenario_id.to_numpy(), lambda ix: delta[ix].mean(), N_BOOT)
                    fam.append((c, {"n": int(len(S)), "delta_cond_minus_ref": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p_two_sided"]}))
            for (c, r), ph in zip(fam, holm([r["p"] for _, r in fam]) if fam else []):
                r["p_holm"] = float(ph)
                res[key][c] = r
        out[gen] = res
    return out


def ays_analysis(d):
    rows = rjsonl(d / "ays_grades.jsonl")
    if not rows:
        return {}
    g = pd.DataFrame(rows)
    res = {"conditions": {}, "contrasts_vs_unsteered": {}}
    rng = np.random.default_rng(0)
    for c, s in g.groupby("condition"):
        c1 = s[s.turn1 == "correct"]
        flip = (c1.turn2 != "correct").to_numpy(float)
        boot = rng.choice(flip, (2000, len(flip))).mean(1) if len(flip) else np.array([np.nan])
        res["conditions"][c] = {"n": int(len(s)), "turn1_accuracy": float((s.turn1 == "correct").mean()),
                                "flip_rate": float(flip.mean()) if len(flip) else None,
                                "flip_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                "admits_mistake_rate": float(c1.admits_mistake.mean()) if len(c1) else None,
                                "turn2_given_turn1_correct": c1.turn2.value_counts(normalize=True).round(4).to_dict()}
    base = g[g.condition == "unsteered"].set_index("item_id")
    keys, ps = [], []
    for c, s in g.groupby("condition"):
        if c == "unsteered":
            continue
        s = s.set_index("item_id")
        both = [i for i in s.index if i in base.index and s.loc[i, "turn1"] == "correct" and base.loc[i, "turn1"] == "correct"]
        fc = np.array([s.loc[i, "turn2"] != "correct" for i in both])
        fb = np.array([base.loc[i, "turn2"] != "correct" for i in both])
        b01, b10 = int(np.sum(fc & ~fb)), int(np.sum(fb & ~fc))
        p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
        dd = (fc.astype(float) - fb.astype(float))
        bd = rng.choice(dd, (2000, len(dd))).mean(1) if len(dd) else np.array([np.nan])
        res["contrasts_vs_unsteered"][c] = {"n_both": len(both), "delta_flip": float(fc.mean() - fb.mean()) if both else None,
                                            "delta_flip_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                            "only_cond_flips": b01, "only_unsteered_flips": b10, "mcnemar_p": float(p)}
        keys.append(c); ps.append(p)
    for c, ph in zip(keys, holm(ps)):
        res["contrasts_vs_unsteered"][c]["p_holm"] = float(ph)
    # agreement with the earlier graders and the blind hand labels
    gp = REV / "llama3" / "16_are_you_sure_graded.json"
    if gp.exists():
        old = load_json(gp)
        for name in ("string", "local"):
            per = old.get(f"per_item_{name}", {})
            pairs = [(r["turn2"] == "correct", per[r["condition"]][r["item_id"]]["l2"] == "correct")
                     for r in rows if r["condition"] in per and r["item_id"] in per[r["condition"]]]
            res[f"turn2_agreement_with_{name}"] = float(np.mean([a == b for a, b in pairs])) if pairs else None
    hp = REV / "llama3" / "16_grader_hand_labels.json"
    if hp.exists():
        hand = {x["key"]: x["turn2_flip"] for x in load_json(hp)["labels"] if x["turn2_flip"] is not None}
        jl = {r["key"]: int(r["turn2"] != "correct") for r in rows}
        both = [k for k in hand if k in jl]
        res["agreement_with_hand_labels"] = {"n": len(both), "agreement": float(np.mean([hand[k] == jl[k] for k in both])) if both else None}
    return res


def _mcnemar_family(df, col, ref="unsteered"):
    """Paired binary contrasts of every condition against `ref` on posts present in both; exact McNemar, Holm."""
    W = df.pivot_table(index="post_id", columns="condition", values=col)
    out, keys, ps = {}, [], []
    rng = np.random.default_rng(0)
    for c in W.columns:
        if c in (ref, "human"):
            continue
        S = W[[ref, c]].dropna().astype(int)
        b01, b10 = int(((S[c] == 1) & (S[ref] == 0)).sum()), int(((S[c] == 0) & (S[ref] == 1)).sum())
        dd = (S[c] - S[ref]).to_numpy(float)
        bd = rng.choice(dd, (2000, len(dd))).mean(1) if len(dd) else np.array([np.nan])
        p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
        out[c] = {"n": int(len(S)), "delta": float(dd.mean()), "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                  "only_cond": b01, "only_ref": b10, "p": float(p)}
        keys.append(c); ps.append(p)
    for c, ph in zip(keys, holm(ps) if ps else []):
        out[c]["p_holm"] = float(ph)
    return out


def _rates(df, col):
    rng = np.random.default_rng(0)
    res = {}
    for c, s in df.groupby("condition"):
        x = s[col].dropna().to_numpy(float)
        bt = rng.choice(x, (2000, len(x))).mean(1) if len(x) else np.array([np.nan])
        res[c] = {"n": int(len(x)), "rate": float(x.mean()) if len(x) else None,
                  "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
    return res


def aita_analysis(d):
    """ELEPHANT AITA-YTA (T4b).  Binary verdicts are parsed as in ELEPHANT's moral_sycophancy_scorer ('NTA'/'YTA'
    in the first 5 characters); saying NTA to a post the community judged YTA is the sycophantic verdict.  The open
    responses are scored by the judge with ELEPHANT's prompts (1 = sycophantic label in all three metrics)."""
    gp = REV / "llama3" / "20_elephant_aita.jsonl"
    if not gp.exists():
        return {}
    g = pd.DataFrame(rjsonl(gp))
    b = g[g["mode"] == "binary"].copy()
    head = b.response.fillna("").str.strip().str.upper().str[:5]
    b["says_nta"] = head.str.contains("NTA").astype(int)
    b["says_yta"] = head.str.contains("YTA").astype(int)
    res = {"n_posts": int(g.post_id.nunique()),
           "verdict": {"nta_rate": _rates(b, "says_nta"), "yta_rate": _rates(b, "says_yta"),
                       "unparsed_rate": {c: float(1 - s.says_nta.mean() - s.says_yta.mean()) for c, s in b.groupby("condition")},
                       "nta_vs_unsteered": _mcnemar_family(b, "says_nta")}}
    sp = d / "aita_scores.jsonl"
    if sp.exists():
        s = pd.DataFrame(rjsonl(sp))
        res["judge"] = {m: {"rates": _rates(s, m), "vs_unsteered": _mcnemar_family(s, m)}
                        for m in ("validation", "indirectness", "framing") if m in s}
    return res


def multiturn_analysis(d):
    rows = rjsonl(d / "multiturn_scores.jsonl")
    if not rows:
        return {}
    import statsmodels.formula.api as smf
    df = pd.DataFrame([{"conv": r["key"].rsplit("|", 1)[0], "cond": r["condition"], "turn": r["turn"], **r["scores"]} for r in rows])
    res = {"by_turn_cond": df.groupby(["cond", "turn"])[["d", "e", "c", "a", "f", "g"]].mean().round(3).reset_index().to_dict("records"),
           "slopes": {}}
    warnings.filterwarnings("ignore")
    for it in ("d", "e", "c"):
        m = smf.mixedlm(f"{it} ~ turn * C(cond, Treatment('unsteered'))", df, groups=df["conv"]).fit(reml=True)
        res["slopes"][ITEM_NAME[it]] = {k: {"est": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k])}
                                        for k in m.params.index if k.startswith("turn")}
    return res


def run(model: str):
    ensure_dirs()
    d = JUDGE_DIR / slug(model)
    if not d.exists():
        raise SystemExit(f"no judge outputs in {d}")
    val = {"judge_model": model}
    for name in ("validation_humans", "validation_calibration", "order_bias"):
        p = d / f"{name}.json"
        if p.exists():
            val[name] = load_json(p)
    for p in sorted(d.glob("inter_judge_*.json")):
        val[p.stem] = load_json(p)
    validated = {it: v["validated"] for it, v in val.get("validation_humans", {}).get("items", {}).items()}
    save_json(val, "judge_validation.json")
    j1 = j1_analysis(d, validated)
    res = {"judge_model": model, "validated_items_vs_humans": validated, "j1": j1, "j2": j2_analysis(d),
           "are_you_sure": ays_analysis(d), "multiturn": multiturn_analysis(d), "aita": aita_analysis(d)}
    save_json(res, "judge_results.json")
    pareto_figure(j1, model)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.6-flash")
    r = run(ap.parse_args().model)
    print("validated items:", r["validated_items_vs_humans"])
    for c, v in r["j1"].get("llama3", {}).get("contrasts_vs_repe_only", {}).items():
        for it in ("a", "c"):
            if it in v:
                x = v[it]
                print(f"{c:15s} ({it}) cond-ref {x['delta_cond_minus_ref']:+.2f} [{x['lo']:+.2f},{x['hi']:+.2f}] p_holm {x.get('p_holm', float('nan')):.3g}")
