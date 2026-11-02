"""
Macros and table fragments for the LLM-judge results (``results/analysis/judge_{validation,results}.json``, written by
``judge_results.py``).  Called from ``make_revision_tables.main``; does nothing if the judge has not been run.
"""
from .common import GENERATED_DIR, OUT_DIR, load_json

COND = {"unsteered": "Unsteered", "pe_only": "PE only", "repe_only": "ACT-RepE", "hybrid": "ACT-Hybrid",
        "repe_const": "Const.\\ target", "repe_shuffled": "Shuffled target", "repe_idonly": "Identity-only target",
        "caa": "CAA vector", "antisyc_prompt": "Anti-syc.\\ prompt"}
ITEMS = [("a", "Approp."), ("b", "Natural"), ("c", "Non-syc."), ("d", "Flatters"), ("e", "Gives in"), ("g", "Harsh")]


def judge(mac, put, cname, num, ci, pstr):
    vp, rp = OUT_DIR / "judge_validation.json", OUT_DIR / "judge_results.json"
    if not (vp.exists() and rp.exists()):
        return
    val, res = load_json(vp), load_json(rp)
    put("JudgeModel", val["judge_model"].replace("gemini-", "Gemini-").replace("-flash", "-Flash"))

    # validation against the 48-participant means and the calibration set
    for it, v in val["validation_humans"]["items"].items():
        mac("JdHum", it, "R", val=v["pearson"], nd=2)
        put(cname("JdHum", it, "Ci"), ci(*v["pearson_ci"], nd=2))
        for c, k in (("unsteered", "Unst"), ("repe_only", "RepE"), ("pe_only", "PE"), ("hybrid", "Hybrid")):
            mac("JdHum", it, k, val=v["judge_condition_means"][c], nd=2)
    for dm, v in val["validation_calibration"]["dims"].items():
        mac("JdCal", dm, "R", val=v["pearson"], nd=2)
        put(cname("JdCal", dm, "Ci"), ci(*v["pearson_ci"], nd=2))
    ob = val.get("order_bias")
    if ob:
        put("JdOrderN", str(ob["n_trials"]))
        mac("JdOrder", "MaxAbs", val=max(ob["mean_abs_change"].values()), nd=2)
        mac("JdOrder", "MaxSigned", val=max(abs(x) for x in ob["mean_signed_change_rev_minus_fwd"].values()), nd=2)

    # J1: condition means and contrasts
    for gen, gl in (("llama3", "Llama"), ("ministral", "Ministral")):
        j = res["j1"].get(gen)
        if not j:
            continue
        for c, v in j["condition_means"].items():
            for it, _ in ITEMS + [("h", "")]:
                mac("Jd", gl, c, it, val=v[it]["mean"], nd=2)
        for ref, key in (("RepE", "contrasts_vs_repe_only"), ("Unst", "contrasts_vs_unsteered")):
            for c, items in j[key].items():
                for it, x in items.items():
                    base = cname("Jd", gl, c, "Vs", ref, it)
                    mac(base, val=x["delta_cond_minus_ref"], nd=2, sign=True)
                    put(base + "Ci", ci(x["lo"], x["hi"], 2))
                    put(base + "P", pstr(x.get("p_holm", x["p"])))
    for name, v in res["j1"]["llama3"].get("human_subsets", {}).items():
        put(cname("JdSub", name, "N"), str(v["n_trials"]))
        for it in ("a", "b", "c"):
            mac("JdSub", name, it, val=v[it]["delta_repe_minus_unsteered"], nd=2, sign=True)
            put(cname("JdSub", name, it, "Ci"), ci(v[it]["lo"], v[it]["hi"], 2))

    # J2: judge EPA as an independent measure of distance to target
    for gen, gl in (("llama3", "Llama"), ("ministral", "Ministral")):
        j = res["j2"].get(gen)
        if not j:
            continue
        put(cname("JdEpa", gl, "N"), f"{j['n_texts']:,}")
        for k, v in j["agreement"].items():
            m, dm = k.split("~")[1].split("_")
            mac("JdEpa", gl, "R", m, dm, val=v, nd=2)
        for ref, key in (("Unst", "paired_vs_unsteered"), ("RepE", "paired_vs_repe_only")):
            for c, x in j[key].items():
                base = cname("JdEpa", gl, c, "Vs", ref)
                mac(base, val=x["delta_cond_minus_ref"], nd=3, sign=True)
                put(base + "Ci", ci(x["lo"], x["hi"], 3))
                put(base + "P", pstr(x["p_holm"]))

    # "Are you sure?" graded by the judge
    a = res["are_you_sure"]
    for c, v in a["conditions"].items():
        mac("AysJd", c, "Acc", val=v["turn1_accuracy"], nd=2)
        mac("AysJd", c, "Flip", val=v["flip_rate"], nd=2)
        put(cname("AysJd", c, "FlipCi"), ci(*v["flip_ci"], nd=2))
        mac("AysJd", c, "ToIncorrect", val=v["turn2_given_turn1_correct"].get("incorrect", 0.0), nd=2)
    for c, v in a["contrasts_vs_unsteered"].items():
        mac("AysJd", c, "Delta", val=v["delta_flip"], nd=2, sign=True)
        put(cname("AysJd", c, "DeltaCi"), ci(*v["delta_flip_ci"], nd=2))
        put(cname("AysJd", c, "P"), pstr(v["p_holm"]))
        put(cname("AysJd", c, "Discord"), f"{v['only_cond_flips']}/{v['only_unsteered_flips']}")
    mac("AysJd", "AgreeString", val=a["turn2_agreement_with_string"], nd=2)
    mac("AysJd", "AgreeLocal", val=a["turn2_agreement_with_local"], nd=2)
    mac("AysJd", "HandAgree", val=a["agreement_with_hand_labels"]["agreement"], nd=2)
    put("AysJdHandN", str(a["agreement_with_hand_labels"]["n"]))

    # multi-turn: per-turn slopes of judge items
    for item, v in res["multiturn"]["slopes"].items():
        for k, x in v.items():
            c = "base" if k == "turn" else k.split("T.")[1].rstrip("]")
            base = cname("JdMt", item, c)
            mac(base, val=x["est"], nd=2, sign=True)
            put(base + "P", pstr(x["p"]))
    for r in res["multiturn"]["by_turn_cond"]:
        mac("JdMt", r["cond"], "Turn", str(r["turn"]), "GivesIn", val=r["e"], nd=2)

    # table: J1 condition means (Llama) with the two primary contrasts against ACT-RepE
    j = res["j1"]["llama3"]
    rows = [r"\begin{tabular}{l" + "c" * len(ITEMS) + "cc}", r"\toprule",
            "Condition & " + " & ".join(l for _, l in ITEMS) + r" & $\Delta$ Approp. & $\Delta$ Non-syc. \\", r"\midrule"]
    for c, lab in COND.items():
        if c not in j["condition_means"]:
            continue
        m = j["condition_means"][c]
        cells = [num(m[it]["mean"], 2) for it, _ in ITEMS]
        ct = j["contrasts_vs_repe_only"].get(c)
        for it in ("a", "c"):
            cells.append("ref." if c == "repe_only" else (f"{num(ct[it]['delta_cond_minus_ref'], 2, True)}{'*' if ct[it].get('p_holm', ct[it]['p']) < 0.05 else ''}" if ct else "--"))
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if c == "hybrid":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED_DIR / "tab_judge.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
