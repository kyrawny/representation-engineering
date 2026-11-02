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
            r"Condition & Dist.\ (own) & Dist.\ (indep.) & Dist.\ minus ACT-RepE (indep.) & $b_1^P$ (indep.) & $\Delta P_{\mathrm{obs}}$ \\",
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
                    f"{(num(dd['delta'], 3, True) + ' ' + ci(dd['lo'], dd['hi'], 3)) if dd else '--'} & "
                    f"{num(b1, 2) if b1 is not None else '--'} & {num(rv['dP_obs'], 2) if rv else '--'} \\\\")
        if dd:
            mac("Abl", cond, "Delta", val=dd["delta"], sign=True)
            put(cname("Abl", cond, "Ci"), ci(dd["lo"], dd["hi"], 3))
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
        lab2 = {"unsteered": "Unsteered", "repe_act": "ACT-RepE", "hybrid_act": "ACT-Hybrid", "repe_const": "Const. target",
                "caa": "CAA vector", "antisyc_prompt": "Anti-syc. prompt"}
        loc, strg = g["graders"].get("local"), g["graders"]["string"]
        jp = OUT_DIR / "judge_results.json"
        jd = load_json(jp)["are_you_sure"] if jp.exists() else None
        rows = [r"\begin{tabular}{lcccccc}" if jd else r"\begin{tabular}{lccccc}", r"\toprule",
                r"Condition & Acc.\ (local / string) & Flip, local [95\% CI] & Flip, string [95\% CI] & $\Delta$ local (Holm $p$) & $\Delta$ string (Holm $p$)"
                + (r" & Flip, judge; $\Delta$ (Holm $p$)" if jd else "") + r" \\",
                r"\midrule"]
        for c, l in lab2.items():
            vs = strg["conditions"].get(c)
            if not vs:
                continue
            vl = loc["conditions"].get(c) if loc else None
            cs, cl = strg["contrasts_vs_unsteered"].get(c), (loc["contrasts_vs_unsteered"].get(c) if loc else None)
            def cell(v):
                return f"{num(v['flip_rate'], 2)} {ci(*v['flip_rate_ci'], nd=2)}" if v else "--"
            def dcell(ct):
                return f"{num(ct['delta_flip'], 2, True)} ({pstr(ct['p_holm'])})" if ct else "--"
            jcell = ""
            if jd:
                jv, jc = jd["conditions"].get(c), jd["contrasts_vs_unsteered"].get(c)
                jcell = " & " + (f"{num(jv['flip_rate'], 2)}" if jv else "--") + ("; " + dcell(jc) if jc else "")
            rows.append(f"{l} & {num(vl['turn1_accuracy'], 2) if vl else '--'} / {num(vs['turn1_accuracy'], 2)} & {cell(vl)} & {cell(vs)} & {dcell(cl)} & {dcell(cs)}{jcell} \\\\")
            for tag, v, ct in (("Loc", vl, cl), ("Str", vs, cs)):
                if not v:
                    continue
                mac("Ays", tag, c, "Acc", val=v["turn1_accuracy"], nd=2)
                mac("Ays", tag, c, "Flip", val=v["flip_rate"], nd=2)
                put(cname("Ays", tag, c, "FlipCi"), ci(*v["flip_rate_ci"], nd=2))
                if ct:
                    mac("Ays", tag, c, "Delta", val=ct["delta_flip"], nd=2, sign=True)
                    put(cname("Ays", tag, c, "DeltaCi"), ci(*ct["delta_flip_ci"], nd=2))
                    put(cname("Ays", tag, c, "P"), pstr(ct["p_holm"]))
        rows += [r"\bottomrule", r"\end{tabular}"]
        _tex("tab_sycophancy_bench.tex", rows)
        if "grader_agreement" in g:
            mac("Ays", "Grader", "Agree", val=g["grader_agreement"]["l2"]["agreement"], nd=2)
        vp = REV / "llama3" / "16_grader_validation.json"
        if vp.exists():
            v = load_json(vp)
            mac("Ays", "Val", "N", val=str(v["n"]))
            mac("Ays", "Val", "String", val=v["string_agreement"], nd=2)
            mac("Ays", "Val", "Local", val=v["local_agreement"], nd=2)
            for c, k in (("unsteered", "Unst"), ("antisyc_prompt", "Anti")):
                mac("Ays", "Val", "Hand", k, val=v["hand_flip_by_condition"][c], nd=2)
                mac("Ays", "Val", "HandN", k, val=str(v["n_by_condition"][c]))
                mac("Ays", "Val", "StrFlip", k, val=v["string_flip_by_condition"][c], nd=2)
                mac("Ays", "Val", "LocFlip", k, val=v["local_flip_by_condition"][c], nd=2)


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
    for reader, pre in (("ministral", "Frontier"), ("mistral7b", "FrontierSeven")):
        for key, v in rr.get("frontier", {}).get(reader, {}).items():
            gen, cond = key.split("|")
            mac(pre, gen, cond, val=v["overall"])
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


def ablation_extra(mac, put, cname, num, ci, pstr):
    """Per-measure ablation contrasts (cond minus ACT-RepE distance; positive = ACT-RepE closer) and tracking slopes."""
    rr = load_json(OUT_DIR / "revision_results.json")
    ab = rr.get("act_ablation", {}).get("llama3", {})
    for m in ("own", "cross", "m7b"):
        t = ab.get(m)
        if not t:
            continue
        for k, v in t["contrasts"].items():
            if not v or not k.startswith("repe_only__vs__"):
                continue
            cond = k.split("__vs__")[1]
            o = v["overall"]
            mac("AblM", m, cond, "Delta", val=o["delta"], sign=True)
            put(cname("AblM", m, cond, "Ci"), ci(o["lo"], o["hi"], 3))
            put(cname("AblM", m, cond, "P"), pstr(o.get("p_holm_primary_family", o["p"])))
            mac("AblM", m, cond, "Dist", val=o["mean_dist_b"])
            if cond == "unsteered":
                mac("AblM", m, "RefDist", val=o["mean_dist_a"])
        for cond, tr in t["tracking"].items():
            for dm in ("P", "A"):
                if dm in tr:
                    mac("Track", m, cond, dm, val=tr[dm]["b1"], nd=2)
    rrv = rr.get("role_reversal", {}).get("llama3", {})
    for m in ("own", "cross"):
        for cond, v in rrv.get(m, {}).items():
            if cond.startswith("_"):
                continue
            mac("RoleM", m, cond, val=v["dP_obs"], nd=2, sign=True)
            put(cname("RoleM", m, cond, "Ci"), ci(v["lo"], v["hi"], 2))
            mac("RoleM", m, cond, "Frac", val=100 * v["dP_obs"] / v["dP_target"], nd=0)


def per_dim(mac, put, cname, num, ci, pstr):
    """Per-dimension paired deltas vs unsteered (Table 3 under each measure)."""
    rr = load_json(OUT_DIR / "revision_results.json")
    gl = {"llama3": "Llama", "ministral": "Ministral"}
    for g in ("llama3", "ministral"):
        for m in ("orig", "cross"):
            t = rr["table3"][g].get(m)
            if not t:
                continue
            for c, k in (("pe_only", "Peonly"), ("repe_only", "Repeonly"), ("hybrid", "Hybrid")):
                for dm in ("E", "P", "A"):
                    d = t["paired_vs_unsteered"][c][dm]
                    mac("DeltaDim", gl[g], m, k, dm, val=d["delta"], sign=True)
                    put(cname("DeltaDim", gl[g], m, k, dm, "P"), pstr(d["p_holm"]))


def per_dim_table(num):
    rr = load_json(OUT_DIR / "revision_results.json")
    COND = {"unsteered": "Unst.", "pe_only": "PE", "repe_only": "RepE", "hybrid": "Hybrid"}
    lab = {"orig": "own reader (full text)", "cross": "independent reader"}
    rows = [r"\begin{tabular}{llcccc}", r"\toprule", r"Texts / measure & Dim. & Unst. & PE & RepE & Hybrid \\", r"\midrule"]
    for g, gn in (("llama3", "Llama"), ("ministral", "Ministral")):
        for m in ("orig", "cross"):
            t = rr["table3"][g].get(m)
            if not t:
                continue
            for i, dm in enumerate(("E", "P", "A")):
                head = f"{gn} / {lab[m]}" if i == 0 else ""
                rows.append(f"{head} & {dm} & " + " & ".join(num(t["conds"][c]["distance"][dm]) for c in COND) + r" \\")
            rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    _tex("tab_steering_dims.tex", rows)


def multiturn_extra(mac, put, cname, num, ci, pstr):
    mt = OUT_DIR / "multiturn_results.json"
    if not mt.exists():
        return
    m = load_json(mt)
    for c in ("unsteered", "repe_act", "hybrid_act", "repe_const"):
        mac("Mt", "Turn3", c, "Agree", val=m["by_turn_cond"][f"{c}|3"]["agree"], nd=2)
        mac("Mt", "Turn1", c, "Agree", val=m["by_turn_cond"][f"{c}|1"]["agree"], nd=2)
        mac("Mt", "Turn3", c, "Words", val=m["by_turn_cond"][f"{c}|3"]["n_words"], nd=0)
    for k, v in m["slopes"]["agree"].items():
        if "turn:" in k:
            c = k.split("T.")[1].rstrip("]")
            mac("Mt", "Slope", c, val=v["est"], nd=2, sign=True)
            put(cname("Mt", "Slope", c, "P"), pstr(v["p"]))
    mac("Mt", "Conversations", val=str(m["n_conversations"]))
    v = load_json(OUT_DIR / "multiturn_marker_validation.json")
    mac("Mt", "Val", "N", val=str(v["n"]))
    mac("Mt", "Val", "Tp", val=str(v["tp"]))
    mac("Mt", "Val", "Fn", val=str(v["fn"]))
    mac("Mt", "Val", "Fp", val=str(v["fp"]))
    mac("Mt", "Val", "Hand", val=str(v["tp"] + v["fn"]))


def legacy_appendix(mac, put, cname, num, ci, pstr):
    """Coherence (exp 07), inference speed (exp 12) and dimension ablation (exp 11): unchanged analyses, moved to the appendix."""
    R = REV.parent
    for g, gn, f7, f12, f11 in (("llama3", "Llama", R / "07_coherence_evaluation.json", R / "12_inference_speed.json", R / "11_dimension_ablation.json"),
                                ("ministral", "Ministral", R / "ministral" / "07_coherence_evaluation.json", R / "ministral" / "12_inference_speed.json",
                                 R / "ministral" / "11_dimension_ablation.json")):
        a = load_json(f7)["aggregate"]
        for c, k in (("unsteered", "Unst"), ("steered", "Steered")):
            mac("Coh", gn, k, "Ppl", val=a[c]["mean_ppl_self"], nd=1)
            mac("Coh", gn, k, "Utr", val=a[c]["mean_unique_token_ratio"])
            mac("Coh", gn, k, "Bigram", val=100 * a[c]["mean_bigram_repetition"], nd=2)
        sp = load_json(f12)["per_condition"]
        mac("Speed", gn, "Unst", val=sp["unsteered"]["mean_tokens_per_second"], nd=1)
        mac("Speed", gn, "RepE", val=sp["repe_only"]["mean_tokens_per_second"], nd=1)
        mac("Speed", gn, "Overhead", val=100 * (sp["unsteered"]["mean_tokens_per_second"] / sp["repe_only"]["mean_tokens_per_second"] - 1), nd=0)
        ab = load_json(f11)["ablation_aggregate"]
        for sub in ("E", "P", "A", "EP", "EA", "PA", "EPA"):
            mac("Dimabl", gn, sub, val=ab[sub]["overall_mean_distance"])
    rows = [r"\begin{tabular}{lccccccc}", r"\toprule", r"Model & E & P & A & EP & EA & PA & EPA \\", r"\midrule"]
    for gn in ("Llama", "Ministral"):
        rows.append(gn + " & " + " & ".join("\\" + cname("Dimabl", gn, s) for s in ("E", "P", "A", "EP", "EA", "PA", "EPA")) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    _tex("tab_dimabl.tex", rows)
