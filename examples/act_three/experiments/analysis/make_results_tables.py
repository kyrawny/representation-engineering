"""
Results tables and macros from revision_results.json, grade_are_you_sure, multiturn_results and hybrid_human_gap.
Called from make_revision_tables.main().  Missing inputs are skipped, never invented.
"""
import json

from .common import GENERATED_DIR, OUT_DIR, load_json, REPO_ROOT

REV = REPO_ROOT / "examples" / "act_three" / "experiments" / "results" / "revision"


def _tex(name, rows):
    (GENERATED_DIR / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(mac, put, cname, num, ci, pstr):
    p = OUT_DIR / "revision_results.json"
    if not p.exists():
        return
    rr = load_json(p)
    COND = {"unsteered": "Unst.", "pe_only": "PE", "repe_only": "RepE", "hybrid": "Hybrid"}
    # ---------------------------------------------------------------- Table 3 under each measure
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"Texts & Measure & Unst. & PE & RepE & Hybrid \\", r"\midrule"]
    lab = {"orig": "own reader (original)", "own": "own reader, 500 chars", "cross": "independent reader",
           "m7b": "Mistral-7B (E,A)"}
    gl = {"llama3": "Llama", "ministral": "Ministral"}
    for g in ("llama3", "ministral"):
        t3 = rr["table3"].get(g, {})
        first = True
        for m in ("orig", "cross", "m7b"):
            if m not in t3:
                continue
            c = t3[m]["conds"]
            cells = " & ".join(num(c[k]["overall"]) for k in COND)
            rows.append(f"{gl[g] if first else ''} & {lab[m]} & {cells} \\\\")
            first = False
            for k in COND:
                mac("Dist", gl[g], m, k, val=c[k]["overall"])
            for k in ("pe_only", "repe_only", "hybrid"):
                d = t3[m]["paired_vs_unsteered"][k]["overall"]
                mac("Delta", gl[g], m, k, val=d["delta"], sign=True)
                put(cname("Delta", gl[g], m, k, "Ci"), ci(d["lo"], d["hi"], 3))
                mac("Delta", gl[g], m, k, "Mde", val=d["mde"])
                put(cname("Delta", gl[g], m, k, "P"), pstr(d.get("p_holm", d["p"])) if "p_holm" in d else pstr(d["p"]))
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    _tex("tab_steering.tex", rows)
    # paired deltas (overall) table: condition minus unsteered with CI and MDE
    rows = [r"\begin{tabular}{llccc}", r"\toprule", r"Texts / measure & Contrast & $\Delta$ distance & 95\% CI & MDE \\", r"\midrule"]
    for g in ("llama3", "ministral"):
        t3 = rr["table3"].get(g, {})
        for m in ("orig", "cross"):
            if m not in t3:
                continue
            for k in ("pe_only", "repe_only", "hybrid"):
                d = t3[m]["paired_vs_unsteered"][k]["overall"]
                rows.append(f"{gl[g]} / {lab[m]} & {COND[k]} $-$ Unst. & {num(d['delta'], 3, True)} & {ci(d['lo'], d['hi'], 3)} & {num(d['mde'])} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    _tex("tab_steering_deltas.tex", rows)

    # ---------------------------------------------------------------- measure agreement
    ag = rr["agreement"]
    rows = [r"\begin{tabular}{llcccc}", r"\toprule", r"Texts & Pair & $r_E$ & $r_P$ & $r_A$ & \\", r"\midrule"]
    pairs = [("orig~cross", "own vs independent"), ("orig~m7b", "own vs Mistral-7B"), ("orig~vader", "own vs VADER"),
             ("orig~own", "full vs 500-char text")]
    for g in ("llama3", "ministral"):
        if g not in ag:
            continue
        for key, labl in pairs:
            vals = []
            for dm in "EPA":
                v = ag[g][dm].get(key)
                vals.append(num(v["r"], 2) if v and v["r"] == v["r"] else "--")
                if v and v["r"] == v["r"]:
                    mac("Agree", gl[g], key, dm, val=v["r"], nd=2)
            if any(x != "--" for x in vals):
                rows.append(f"{gl[g]} & {labl} & {' & '.join(vals)} & \\\\")
        vals = [num(ag[g][dm]["orig~target"]["r"], 2) for dm in "EPA"]
        rows.append(f"{gl[g]} & own reader vs ACT target (all conds) & {' & '.join(vals)} & \\\\")
        for dm in "EPA":
            mac("Agree", gl[g], "OrigTarget", dm, val=ag[g][dm]["orig~target"]["r"], nd=2)
    rows += [r"\bottomrule", r"\end{tabular}"]
    _tex("tab_measure_agreement.tex", rows)

    # ---------------------------------------------------------------- ACT ablation
    ab = rr.get("act_ablation", {}).get("llama3", {})
    rr_roles = rr.get("role_reversal", {}).get("llama3", {})
    order = [("unsteered", "Unsteered"), ("pe_only", "PE only"), ("repe_only", "ACT-RepE"), ("repe_const", "Const. target"),
             ("repe_shuffled", "Shuffled target"), ("repe_idonly", "Identity-only target"), ("caa", "CAA vector"),
             ("antisyc_prompt", "Anti-syc. prompt")]
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"Condition & Dist.\ (own) & Dist.\ (indep.) & $\Delta$ vs ACT-RepE (indep.) & $b_1^P$ (indep.) & $\Delta P_{\mathrm{obs}}$ \\",
            r"\midrule"]
    have = False
    for cond, labl in order:
        cells = []
        for m in ("own", "cross"):
            t = ab.get(m)
            if t is None:
                cells.append("--")
                continue
            tr = t["tracking"].get(cond, {})
            cells.append(None)
        # distance per measure from a paired contrast (mean_dist_b) or from the reference side
        def dist(m, cond=cond):
            t = ab.get(m)
            if not t:
                return None
            for k, v in t["contrasts"].items():
                if k.endswith("__vs__" + cond) and v:
                    return v["overall"]["mean_dist_b"]
                if k.startswith(cond + "__vs__") and v:
                    return v["overall"]["mean_dist_a"]
            return None
        do, dc = dist("own"), dist("cross")
        d_ref = ab.get("cross", {}).get("contrasts", {}).get(f"repe_only__vs__{cond}", {})
        dd = d_ref.get("overall") if d_ref else None
        b1 = ab.get("cross", {}).get("tracking", {}).get(cond, {}).get("P", {}).get("b1")
        rv = rr_roles.get("cross", {}).get(cond)
        if do is None and dc is None:
            continue
        have = True
        rows.append(f"{labl} & {num(do) if do is not None else '--'} & {num(dc) if dc is not None else '--'} & "
                    f"{(num(-dd['delta'], 3, True) + ' ' + ci(-dd['hi'], -dd['lo'], 3)) if dd else '--'} & "
                    f"{num(b1, 2) if b1 is not None else '--'} & {num(rv['dP_obs'], 2) if rv else '--'} \\\\")
        if dd:
            mac("Abl", cond, "Delta", val=-dd["delta"], sign=True)
            put(cname("Abl", cond, "Ci"), ci(-dd["hi"], -dd["lo"], 3))
            put(cname("Abl", cond, "P"), pstr(dd.get("p_holm_primary_family", dd["p"])))
            mac("Abl", cond, "Mde", val=dd["mde"])
        if rv:
            mac("Role", cond, "Obs", val=rv["dP_obs"], nd=2, sign=True)
            put(cname("Role", cond, "Ci"), ci(rv["lo"], rv["hi"], 2))
            mac("Role", cond, "Target", val=rv["dP_target"], nd=2)
    rows += [r"\bottomrule", r"\end{tabular}"]
    if have:
        _tex("tab_act_ablation.tex", rows)

    # ---------------------------------------------------------------- are-you-sure
    ays_p = REV / "llama3" / "16_are_you_sure_graded.json"
    if ays_p.exists():
        g = load_json(ays_p)
        gr = g["graders"].get("local") or g["graders"]["string"]
        gname = "local" if "local" in g["graders"] else "string"
        lab2 = {"unsteered": "Unsteered", "repe_act": "ACT-RepE", "hybrid_act": "ACT-Hybrid", "repe_const": "Const. target",
                "caa": "CAA vector", "antisyc_prompt": "Anti-syc. prompt"}
        rows = [r"\begin{tabular}{lcccc}", r"\toprule",
                r"Condition & Turn-1 acc. & Flip rate [95\% CI] & $\Delta$ flip vs Unst. & Holm $p$ \\", r"\midrule"]
        for c, l in lab2.items():
            v = gr["conditions"].get(c)
            if not v:
                continue
            ct = gr["contrasts_vs_unsteered"].get(c)
            rows.append(f"{l} & {num(v['turn1_accuracy'], 2)} & {num(v['flip_rate'], 2)} {ci(*v['flip_rate_ci'], nd=2)} & "
                        f"{(num(ct['delta_flip'], 2, True)) if ct else '--'} & {pstr(ct['p_holm']) if ct else '--'} \\\\")
            mac("Ays", c, "Acc", val=v["turn1_accuracy"], nd=2)
            mac("Ays", c, "Flip", val=v["flip_rate"], nd=2)
            put(cname("Ays", c, "FlipCi"), ci(*v["flip_rate_ci"], nd=2))
            mac("Ays", c, "Admit", val=v["admits_mistake_rate"], nd=2)
            if ct:
                mac("Ays", c, "Delta", val=ct["delta_flip"], nd=2, sign=True)
                put(cname("Ays", c, "DeltaCi"), ci(*ct["delta_flip_ci"], nd=2))
                put(cname("Ays", c, "P"), pstr(ct["p_holm"]))
        rows += [r"\bottomrule", r"\end{tabular}"]
        _tex("tab_sycophancy_bench.tex", rows)
        if "grader_agreement" in g:
            mac("Ays", "Grader", "Agree", val=g["grader_agreement"]["l2"]["agreement"], nd=2)
        mac("Ays", "Grader", "Name", val=gname)


def more(mac, put, cname, num, ci, pstr):
    """Macros for numbers quoted in the text: truncation, untruncated sensitivity, frontier, T10, multi-turn."""
    import pandas as pd
    from . import rescore_data as RD
    rr = load_json(OUT_DIR / "revision_results.json")
    gl = {"llama3": "Llama", "ministral": "Ministral"}
    for g in ("llama3", "ministral"):
        df = RD.build(g, with_vader=False)
        d = df[df.source == "08"]
        for c, k in (("unsteered", "Unst"), ("pe_only", "PE"), ("repe_only", "RepE"), ("hybrid", "Hybrid")):
            mac("Trunc", gl[g], k, val=100 * float((d[d.cond == c].text500.str.len() >= 500).mean()), nd=0)
        t3 = rr["table3"][g]
        for m in ("orig", "cross", "m7b"):
            if m not in t3:
                continue
            u = t3[m]["untruncated_trials"]
            if m == "orig":
                mac("Untr", gl[g], "N", val=str(u["n_trials"]))
            for c, k in (("pe_only", "Peonly"), ("repe_only", "Repeonly"), ("hybrid", "Hybrid")):
                dd = u["paired_vs_unsteered"][c]["overall"]
                mac("Untr", gl[g], m, k, val=dd["delta"], sign=True)
                put(cname("Untr", gl[g], m, k, "Ci"), ci(dd["lo"], dd["hi"], 3))
    fr = rr.get("frontier", {}).get("ministral", {})
    for key, v in fr.items():
        gen, cond = key.split("|")
        mac("Frontier", gen, cond, val=v["overall"])
    # T10
    hp = OUT_DIR / "hybrid_human_gap.json"
    if hp.exists():
        h = load_json(hp)
        tf = h["text_features"]["llama3"]["paired_vs_unsteered"]
        for c, k in (("pe_only", "PE"), ("repe_only", "RepE"), ("hybrid", "Hybrid")):
            mac("Feat", k, "Words", val=tf[c]["n_words"]["delta"], nd=0, sign=True)
            mac("Feat", k, "Excl", val=tf[c]["exclaim_per_sent"]["delta"], nd=2, sign=True)
        m = h["text_features"]["llama3"]["means"]
        for c, k in (("unsteered", "Unst"), ("hybrid", "Hybrid"), ("repe_only", "RepE")):
            mac("Feat", k, "MeanWords", val=m[c]["n_words"], nd=0)
        am = h["human"]["alignment_models"]
        for it, lab in (("appropriate", "Appropriate"), ("natural", "Natural")):
            for dm in ("E", "P", "A"):
                x = am[it]["within_condition"][f"abs_dev_{dm}"]
                mac("Align", lab, dm, val=x["est"], nd=2, sign=True)
                put(cname("Align", lab, dm, "P"), pstr(x["p"]))
        asr = h["human"]["assertiveness_models"]
        mac("Assert", "Unadj", val=asr["repe_only_minus_unsteered_unadjusted"]["est"], nd=2)
        mac("Assert", "Adj", val=asr["repe_only_minus_unsteered_adjusted"]["est"], nd=2)
        mac("Assert", "Atten", val=asr["attenuation_pct"], nd=0)
        os_ = h["oversteer"]["llama3"].get("cross", {})
        for c, k in (("unsteered", "Unst"), ("pe_only", "PE"), ("repe_only", "RepE"), ("hybrid", "Hybrid")):
            for dm in ("E", "P", "A"):
                if c in os_ and dm in os_[c]:
                    mac("Over", k, dm, val=os_[c][dm]["mean_signed_dev"], nd=2, sign=True)
    mt = OUT_DIR / "multiturn_results.json"
    if mt.exists():
        m = load_json(mt)
        for key, v in m["deltas_turn3_minus_turn1"].items():
            c, sc = key.split("|")
            mac("Mt", c, sc, val=v["delta"], nd=2, sign=True)
            put(cname("Mt", c, sc, "Ci"), ci(v["lo"], v["hi"], 2))
