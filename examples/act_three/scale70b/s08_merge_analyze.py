"""
Stage 8 (CPU only, no GPU or torch needed): merge shards, compute the
statistics, and write REPORT.md + summary.json.

Run it as a plain script so it works on a laptop::

    python examples/act_three/scale70b/s08_merge_analyze.py --model meta-llama/Llama-3.1-70B-Instruct

Statistics (same as the revision analyses in experiments/analysis/):
  * mean |observed - target| per condition and dimension (paper Table 3);
  * paired per-trial difference vs unsteered, with a cluster bootstrap over
    scenarios (2,000 resamples), Holm correction, design-effect-adjusted
    minimum detectable effect (MDE) and d_z;
  * tracking slope (observed on target) and Lin's concordance (CCC);
  * specificity: distance to own target minus distance to a constant or a
    shuffled-identity target.
Every table is computed under the model's own reader (as in the paper) and,
where stage 7 has run, under a reader built on a different model.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("scale70b_settings", HERE / "settings.py")
M = _load("act_metrics", S.ACT_THREE_DIR / "experiments" / "analysis" / "metrics.py")
DIMS = S.DIMS
STEERED = ["repe_only", "pe_only", "hybrid", "repe_const", "repe_only_s1", "hybrid_s1"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_jsonl(paths):
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_scale_trials(model, tag):
    files = sorted((S.results_dir(model) / tag).glob("shard_*.jsonl"))
    recs = {}
    for r in read_jsonl(files):
        recs[r["key"]] = r          # later duplicates (re-runs) replace earlier ones
    return sorted(recs.values(), key=lambda r: r["index"])


def load_reference_trials():
    if not S.REFERENCE_8B_LOG.exists():
        return []
    d = json.loads(S.REFERENCE_8B_LOG.read_text(encoding="utf-8"))
    out = []
    for i, t in enumerate(d["trials"]):
        t = dict(t)
        t["key"] = f"{t['scenario_id']}|{t['pair']}"
        t["index"] = i
        t["category"] = t["scenario_id"].rsplit("_", 1)[0]
        out.append(t)
    return out


def cross_reads(generator):
    out = {}
    for p in sorted(S.results_dir(generator).glob("read_by_*.jsonl")):
        reader = p.stem.replace("read_by_", "")
        out[reader] = {r["key"]: r["epa_text500"] for r in read_jsonl([p])}
    return out


def wide(trials, conds, getter):
    """One row per trial: target_<d>, <cond>_<d> (NaN when missing)."""
    rows = []
    for t in trials:
        row = {"key": t["key"], "scenario_id": t["scenario_id"],
               "category": t.get("category", t["scenario_id"].rsplit("_", 1)[0]), "pair": t["pair"]}
        for d in DIMS:
            row[f"target_{d}"] = t["target_epa"][d]
        for c in conds:
            e = getter(t, c)
            for d in DIMS:
                row[f"{c}_{d}"] = np.nan if e is None else e[d]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def analyse(W, conds, label):
    rng = np.random.default_rng(0)
    res = {"label": label, "n_trials": int(len(W)), "conditions": {}, "paired_vs_unsteered": {}}
    shuffled = np.arange(len(W))
    for sid, grp in W.groupby("scenario_id"):
        idx = grp.index.to_numpy()
        perm = idx.copy()
        tries = 0
        while len(idx) > 1 and np.any(perm == idx) and tries < 100:
            perm = rng.permutation(idx)
            tries += 1
        shuffled[idx] = perm
    const = {d: W[f"target_{d}"].mean() for d in DIMS}

    for c in conds:
        ok = W[[f"{c}_{d}" for d in DIMS]].notna().all(axis=1).to_numpy()
        if ok.sum() < 10:
            continue
        cl = W.scenario_id.to_numpy()[ok]
        cres = {"n": int(ok.sum())}
        for d in DIMS:
            t = W[f"target_{d}"].to_numpy()[ok]
            e = W[f"{c}_{d}"].to_numpy()[ok]
            fit = M.tracking_fit(e, t)
            b1 = M.cluster_bootstrap(cl, lambda ix: M.tracking_fit(e[ix], t[ix])["b1"], S.N_BOOT)
            ccc = M.cluster_bootstrap(cl, lambda ix: M.lin_ccc(e[ix], t[ix]), S.N_BOOT)
            dist = np.abs(e - t)
            cres[d] = {"mean_distance": float(dist.mean()), "b1": fit["b1"], "b1_lo": b1["lo"],
                       "b1_hi": b1["hi"], "r": fit["r"], "ccc": ccc["est"], "ccc_lo": ccc["lo"],
                       "ccc_hi": ccc["hi"],
                       "spec_vs_constant": float((dist - np.abs(e - const[d])).mean()),
                       "spec_vs_shuffled": float((dist - np.abs(
                           e - W[f"target_{d}"].to_numpy()[shuffled][ok])).mean())}
        cres["overall_mean_distance"] = float(np.mean([cres[d]["mean_distance"] for d in DIMS]))
        res["conditions"][c] = cres

    pvals, keys = [], []
    for c in conds:
        if c == "unsteered" or c not in res["conditions"]:
            continue
        ok = W[[f"{x}_{d}" for x in (c, "unsteered") for d in DIMS]].notna().all(axis=1).to_numpy()
        cl = W.scenario_id.to_numpy()[ok]
        pres = {}
        for d in DIMS + ["overall"]:
            dd = DIMS if d == "overall" else [d]
            delta = np.mean([np.abs(W[f"{c}_{x}"].to_numpy()[ok] - W[f"target_{x}"].to_numpy()[ok])
                             - np.abs(W[f"unsteered_{x}"].to_numpy()[ok] - W[f"target_{x}"].to_numpy()[ok])
                             for x in dd], axis=0)
            ci = M.cluster_bootstrap(cl, lambda ix: delta[ix].mean(), S.N_BOOT)
            sd = float(delta.std(ddof=1))
            deff = M.design_effect(delta, cl)
            pres[d] = {"delta": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p_boot": ci["p_two_sided"],
                       "n": int(ok.sum()), "design_effect": deff,
                       "mde": M.mde_paired(sd, ok.sum() / deff), "d_z": float(delta.mean() / sd) if sd > 0 else 0.0}
            if d != "overall":
                pvals.append(ci["p_two_sided"])
                keys.append((c, d))
        res["paired_vs_unsteered"][c] = pres
    for (c, d), p in zip(keys, M.holm(pvals)):
        res["paired_vs_unsteered"][c][d]["p_holm"] = float(p)
    return res


def overall_delta_samples(W, cond, n_boot, seed):
    ok = W[[f"{x}_{d}" for x in (cond, "unsteered") for d in DIMS]].notna().all(axis=1).to_numpy()
    delta = np.mean([np.abs(W[f"{cond}_{x}"].to_numpy()[ok] - W[f"target_{x}"].to_numpy()[ok])
                     - np.abs(W[f"unsteered_{x}"].to_numpy()[ok] - W[f"target_{x}"].to_numpy()[ok])
                     for x in DIMS], axis=0)
    labels = W.scenario_id.to_numpy()[ok]
    uniq, inv = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        out[b] = delta[np.concatenate([groups[g] for g in pick])].mean()
    return float(delta.mean()), out


def verdict(own, cross):
    """Pre-specified rule (README): the effect holds for a condition if the overall
    paired distance change vs unsteered is negative with a 95% CI excluding 0 under
    BOTH the own reader and an independent reader."""
    out = {}
    for c in ("repe_only", "hybrid"):
        h1 = own["paired_vs_unsteered"].get(c, {}).get("overall", {})
        h2 = cross["paired_vs_unsteered"].get(c, {}).get("overall", {}) if cross else {}
        own_ok = bool(h1) and h1["hi"] < 0
        ind_ok = bool(h2) and h2["hi"] < 0
        out[c] = {"own_reader": own_ok, "independent_reader": ind_ok if cross else None,
                  "holds": own_ok and ind_ok}
    holds = [c for c, v in out.items() if v["holds"]]
    if len(holds) == 2:
        text = "HOLDS: RepE-only and Hybrid both move outputs toward the ACT targets under the own and an independent reader."
    elif holds:
        text = f"HOLDS FOR {holds[0].upper()} ONLY (own and independent reader)."
    elif any(v["own_reader"] for v in out.values()):
        text = "OWN READER ONLY: the effect is not confirmed by an independent reader."
    else:
        text = "NOT DETECTED: no significant movement toward the ACT targets at this strength."
    if not cross:
        text += " (Independent-reader check missing: run stage 7.)"
    return {"per_condition": out, "summary": text}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def table_distance(res, conds):
    lines = ["| Dim | " + " | ".join(c for c in conds if c in res["conditions"]) + " |",
             "|---|" + "---|" * sum(c in res["conditions"] for c in conds)]
    for d in DIMS + ["overall"]:
        cells = []
        for c in conds:
            if c not in res["conditions"]:
                continue
            v = res["conditions"][c]["overall_mean_distance"] if d == "overall" else res["conditions"][c][d]["mean_distance"]
            cells.append(fmt(v))
        lines.append(f"| {d[:1].upper() if d != 'overall' else 'Overall'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_paired(res):
    lines = ["| Condition | Dim | Δ vs unsteered | 95% CI | p (Holm) | MDE | d_z |", "|---|---|---|---|---|---|---|"]
    for c, pr in res["paired_vs_unsteered"].items():
        for d, x in pr.items():
            lines.append(f"| {c} | {d} | {fmt(x['delta'])} | [{fmt(x['lo'])}, {fmt(x['hi'])}] | "
                         f"{fmt(x.get('p_holm', x['p_boot']), 4)} | {fmt(x['mde'])} | {fmt(x['d_z'], 2)} |")
    return "\n".join(lines)


def table_tracking(res, conds):
    lines = ["| Condition | Dim | slope b1 [95% CI] | CCC [95% CI] | spec. vs constant | spec. vs shuffled |",
             "|---|---|---|---|---|---|"]
    for c in conds:
        if c not in res["conditions"]:
            continue
        for d in DIMS:
            x = res["conditions"][c][d]
            lines.append(f"| {c} | {d[:1].upper()} | {fmt(x['b1'], 2)} [{fmt(x['b1_lo'], 2)}, {fmt(x['b1_hi'], 2)}] | "
                         f"{fmt(x['ccc'], 2)} [{fmt(x['ccc_lo'], 2)}, {fmt(x['ccc_hi'], 2)}] | "
                         f"{fmt(x['spec_vs_constant'])} | {fmt(x['spec_vs_shuffled'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    rdir = S.results_dir(args.model)
    summary = {"model": args.model, "generated_at": datetime.now().isoformat(), "analyses": {}}
    md = [f"# ACT steering at scale: {args.model}", "",
          f"Generated {summary['generated_at']} by `scale70b/s08_merge_analyze.py`.", ""]

    # ---- merge
    trials = load_scale_trials(args.model, args.tag)
    if trials:
        conds = [c for c in S.CORE_CONDITIONS + S.OPTIONAL_CONDITIONS if c in trials[0]["conditions"]
                 or any(c in t["conditions"] for t in trials)]
        merged = {"metadata": {"experiment": "08_hybrid_steering (scale70b)", "model": args.model,
                               "n_trials": len(trials), "conditions": conds,
                               "strength": trials[0].get("strength"), "merged_at": summary["generated_at"]},
                  "trials": trials}
        (rdir / f"08_hybrid_steering_{args.tag}.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
        tg_path = rdir / "targets.json"
        expected = len(json.loads(tg_path.read_text())["trials"]) if tg_path.exists() else None
        md += [f"Trials merged: **{len(trials)}**" + (f" of {expected} planned" if expected else "") +
               f"; conditions: {', '.join(conds)}; strength s = {trials[0].get('strength')}.", ""]
    else:
        conds = []
        md += ["**No scale-model outputs found yet.** Showing the 8B reference only.", ""]

    # ---- context: reader quality, calibration, norm probe, coherence
    ctx = {}
    for name in ("targets.json", "calibration.json", "norm_probe.json", "preflight.json", f"coherence_{args.tag}.json"):
        p = rdir / name
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            ctx[name] = {k: v for k, v in d.items() if k not in ("trials", "per_layer")}
    summary["context"] = ctx
    ref_tuning = S.MODELS_DIR / "llama3" / "epa_reading_tuning_v2_results.json"
    if "targets.json" in ctx and ref_tuning.exists():
        rt = json.loads(ref_tuning.read_text())["methods"][S.READING_METHOD]
        md += ["## Reader quality (held-out calibration set, n = 130)", "",
               "| Dim | 8B r | 8B MAE | scale r | scale MAE |", "|---|---|---|---|---|"]
        for d in DIMS:
            a, b = rt[d]["metrics_test"], ctx["targets.json"]["reader_test_metrics"][d]
            md.append(f"| {d[:1].upper()} | {fmt(a['pearson'])} | {fmt(a['mae'])} | {fmt(b['pearson'])} | {fmt(b['mae'])} |")
        md.append("")
    if "calibration.json" in ctx:
        c = ctx["calibration.json"]
        md += ["## Steering strength", "",
               f"Chosen strength s = **{c.get('chosen_strength')}** by the pre-specified degeneration rule "
               f"(s_match from norm probe = {fmt(c.get('s_match'), 2)}).", ""]
        ref_cal = S.REFERENCE_DIR / "calibration_llama3.json"
        if ref_cal.exists():
            s8 = json.loads(ref_cal.read_text(encoding="utf-8")).get("chosen_strength")
            md += [f"Rule check on Llama-3.1-8B (stage R): the same rule chose s = {s8} "
                   f"(paper setting: s = 1).", ""]
        md += ["| s | degenerate | ppl ratio | length ratio | pass |", "|---|---|---|---|---|"]
        for r in c["summary"]:
            md.append(f"| {r['strength']} | {fmt(r['degenerate_frac'], 2)} | {fmt(r['ppl_ratio_geo'], 2)} | "
                      f"{fmt(r['length_ratio'], 2)} | {r['pass']} |")
        md.append("")
    if "norm_probe.json" in ctx:
        n = ctx["norm_probe.json"]
        md += [f"Relative perturbation at s = 1: {fmt(n['rel_perturbation_s1_mean'], 4)} "
               f"(8B reference {fmt(n.get('reference_rel_8b'), 4)}).", ""]

    # ---- main analyses
    ref = load_reference_trials()
    analyses = {}
    if trials:
        W = wide(trials, conds, lambda t, c: t["conditions"].get(c, {}).get("epa"))
        analyses["scale_own"] = analyse(W, conds, f"{args.model}, own reader (full text)")
        for reader, reads in cross_reads(args.model).items():
            Wc = wide(trials, conds, lambda t, c, r=reads: r.get(t["key"], {}).get(c))
            analyses[f"scale_read_by_{reader}"] = analyse(Wc, conds, f"{args.model}, read by {reader} (first 500 chars)")
    if ref:
        rconds = ["unsteered", "repe_only", "pe_only", "hybrid"]
        Wr = wide(ref, rconds, lambda t, c: t["conditions"][c]["epa"])
        analyses["ref8b_own"] = analyse(Wr, rconds, "Llama-3.1-8B (paper), own reader")
        for reader, reads in cross_reads(S.REFERENCE_MODEL).items():
            Wrc = wide(ref, rconds, lambda t, c, r=reads: r.get(t["key"], {}).get(c))
            analyses[f"ref8b_read_by_{reader}"] = analyse(Wrc, rconds, f"Llama-3.1-8B (paper), read by {reader}")
    summary["analyses"] = analyses

    if "scale_own" in analyses:
        indep = [k for k in analyses if k.startswith("scale_read_by_") and not k.endswith(S.short_name(args.model))]
        v = verdict(analyses["scale_own"], analyses[indep[0]] if indep else None)
        summary["verdict"] = v
        md += ["## Verdict (pre-specified rule)", "", f"**{v['summary']}**", "",
               "| Condition | own reader | independent reader | holds |", "|---|---|---|---|"]
        for c, x in v["per_condition"].items():
            md.append(f"| {c} | {x['own_reader']} | {x['independent_reader']} | {x['holds']} |")
        md.append("")
        if ref:
            md += ["## Effect size: scale model vs 8B (own readers)", "",
                   "| Condition | Δ overall 8B | Δ overall scale | ratio scale/8B [95% CI] |", "|---|---|---|---|"]
            comp = {}
            for c in ("repe_only", "hybrid", "pe_only"):
                if c not in conds:
                    continue
                e8, b8 = overall_delta_samples(Wr, c, S.N_BOOT, 1)
                e70, b70 = overall_delta_samples(W, c, S.N_BOOT, 2)
                ratio = b70 / b8
                lo, hi = np.percentile(ratio, [2.5, 97.5])
                comp[c] = {"delta_8b": e8, "delta_scale": e70, "ratio": e70 / e8, "ratio_lo": float(lo), "ratio_hi": float(hi)}
                md.append(f"| {c} | {fmt(e8)} | {fmt(e70)} | {fmt(e70 / e8, 2)} [{fmt(lo, 2)}, {fmt(hi, 2)}] |")
            summary["scale_vs_8b"] = comp
            md.append("")
            # agreement of read user-message EPA and targets across scales
            r8 = {t["key"]: t for t in ref}
            both = [(t, r8[t["key"]]) for t in trials if t["key"] in r8]
            if both:
                agree = {}
                for d in DIMS:
                    a = np.array([x["user_statement_epa"][d] for x, _ in both])
                    b = np.array([y["user_statement_epa"][d] for _, y in both])
                    ta = np.array([x["target_epa"][d] for x, _ in both])
                    tb = np.array([y["target_epa"][d] for _, y in both])
                    agree[d] = {"user_msg_r": float(np.corrcoef(a, b)[0, 1]), "target_r": float(np.corrcoef(ta, tb)[0, 1])}
                summary["cross_scale_agreement"] = agree
                md += ["Agreement between scales (same trials): " + "; ".join(
                    f"{d[:1].upper()} user-message r = {fmt(x['user_msg_r'], 2)}, target r = {fmt(x['target_r'], 2)}"
                    for d, x in agree.items()), ""]

    for key, res in analyses.items():
        cl = [c for c in (conds if key.startswith("scale") else ["unsteered", "repe_only", "pe_only", "hybrid"])]
        md += [f"## {res['label']}", "", f"n = {res['n_trials']} trials", "", "### Mean distance to ACT target (lower is better)", "",
               table_distance(res, cl), "", "### Paired change vs unsteered (negative = closer to target)", "",
               table_paired(res), "", "### Tracking and specificity", "", table_tracking(res, cl), ""]

    if f"coherence_{args.tag}.json" in ctx:
        md += ["## Coherence", "", "| Condition | self-ppl (geo) | ratio vs unsteered |", "|---|---|---|"]
        for c, x in ctx[f"coherence_{args.tag}.json"]["summary"].items():
            md.append(f"| {c} | {fmt(x.get('self_ppl_geo'), 2)} | {fmt(x.get('self_ppl_ratio_vs_unsteered'), 2)} |")
        md.append("")

    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (rdir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {rdir / 'REPORT.md'} and summary.json")
    if "verdict" in summary:
        print(summary["verdict"]["summary"])


if __name__ == "__main__":
    main()
