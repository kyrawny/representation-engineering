"""
Why does Hybrid disagree with humans, and is the non-sycophancy effect just assertiveness? (plan T10, judge-free)

1. text features per condition (all trials; paired by scenario, cluster bootstrap) and for the 56 human-rated texts
2. human ratings ~ |deviation from ACT target| (+ length): exploratory mixed models (crossed random effects)
3. non-sycophancy ~ cond + assertiveness / harshness *proxies* (own-reader P and -E of the shown text, imperative
   and exclamation rates).  No LLM judge exists, so these proxies come from the same reader that sets targets;
   this is reported as a weaker test.
4. over-steering: signed deviation (observed - target) of Hybrid vs PE vs RepE on the independent reader

    python -m examples.act_three.experiments.analysis.hybrid_human_gap
"""
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import rescore_data as RD
from .common import OUT_DIR, save_json
from .human_reanalysis import parse_survey
from .metrics import cluster_bootstrap

LEX = {
    "hedge_apology": r"\b(sorry|apologi[sz]e|apologies|i think|perhaps|maybe|might|possibly|i'?m afraid|i understand)\b",
    "praise": r"\b(great|wonderful|excellent|amazing|fantastic|thank you|thanks|appreciate|impressive|brilliant|proud|well done|kind)\b",
    "validation": r"\b(you'?re right|you are right|absolutely|of course|you'?re welcome|i completely agree|i agree|totally|that'?s a great)\b",
    "second_person": r"\b(you|your|you'?re|yours|yourself|you'?ll|you'?ve)\b",
}
IMPERATIVE = set("let please take tell give consider remember try listen focus make do don't stop get come go think look ask "
                 "explain start keep be show find review schedule trust understand know see use help follow bring send put "
                 "stay calm relax talk speak call meet check read write set work".split())


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def features(t: str) -> dict:
    toks = t.split()
    ss = sentences(t) or [t]
    low = t.lower()
    first = [re.sub(r"[^a-z']", "", s.split()[0].lower()) if s.split() else "" for s in ss]
    return {
        "n_words": len(toks), "n_sentences": len(ss),
        "question_per_sent": t.count("?") / len(ss), "exclaim_per_sent": t.count("!") / len(ss),
        "imperative_rate": float(np.mean([f in IMPERATIVE for f in first])),
        **{k: len(re.findall(p, low)) / max(len(toks), 1) for k, p in LEX.items()},
        "type_token_ratio": len(set(w.lower() for w in toks)) / max(len(toks), 1)}


def feature_table(df: pd.DataFrame, conds, ref="unsteered") -> dict:
    F = pd.DataFrame([features(t) for t in df.text500]).assign(cond=df.cond.to_numpy(), sid=df.scenario_id.to_numpy(),
                                                              tid=RD.trial_id(df).to_numpy())
    out = {"means": F.groupby("cond").mean(numeric_only=True).loc[[c for c in conds if c in set(F.cond)]].to_dict("index"),
           "paired_vs_" + ref: {}}
    base = F[F.cond == ref].set_index("tid")
    for c in conds:
        if c == ref or c not in set(F.cond):
            continue
        cur = F[F.cond == c].set_index("tid")
        idx = base.index.intersection(cur.index)
        if len(idx) < 30:
            continue
        res = {}
        cl = cur.loc[idx, "sid"].to_numpy()
        for k in [x for x in F.columns if x not in ("cond", "sid", "tid")]:
            delta = (cur.loc[idx, k] - base.loc[idx, k]).to_numpy()
            ci = cluster_bootstrap(cl, lambda ix: delta[ix].mean(), 1000)
            res[k] = {"delta": ci["est"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p_two_sided"]}
        out["paired_vs_" + ref][c] = res
    return out


def mixed(formula, d):
    d = d.copy(); d["one"] = 1
    m = smf.mixedlm(formula, d, groups="one", vc_formula={"pid": "0 + C(pid)", "vig": "0 + C(vig)"}).fit(reml=True)
    return m


def human_part() -> dict:
    long = pd.read_csv(OUT_DIR / "human_long.csv")
    blocks = parse_survey()[["vig", "cond", "text"]]
    long = long.merge(blocks, on=["vig", "cond"], how="left")
    uniq = long.drop_duplicates(["vig", "cond"]).copy()
    feats = pd.DataFrame([features(t) for t in uniq.text.fillna("")], index=uniq.index)
    uniq = pd.concat([uniq[["vig", "cond"]], feats], axis=1)
    long = long.merge(uniq, on=["vig", "cond"], how="left")
    long["abs_dev_E"] = long.signed_dev_e.abs(); long["abs_dev_P"] = long.signed_dev_p.abs(); long["abs_dev_A"] = long.signed_dev_a.abs()
    long["assert_proxy"] = long.obs_p; long["harsh_proxy"] = -long.obs_e
    long["len_z"] = (long.n_words - long.n_words.mean()) / long.n_words.std()
    long["cond"] = pd.Categorical(long.cond, ["unsteered", "pe_only", "repe_only", "hybrid"])
    res = {"feature_means_by_cond_56_texts": uniq.groupby("cond", observed=True).mean(numeric_only=True).to_dict("index"),
           "alignment_models": {}, "assertiveness_models": {}}
    warnings.filterwarnings("ignore")
    ok = long.dropna(subset=["abs_dev_E", "abs_dev_P", "abs_dev_A", "len_z"])
    for item in ("appropriate", "natural", "non_sycophantic"):
        d = ok[ok["item"] == item]
        m = mixed("rating ~ abs_dev_E + abs_dev_P + abs_dev_A + len_z", d)
        res["alignment_models"][item] = {k: {"est": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k])}
                                         for k in ("abs_dev_E", "abs_dev_P", "abs_dev_A", "len_z")}
        res["alignment_models"][item]["n"] = int(len(d))
        # same, condition-adjusted: does alignment predict ratings *within* condition?
        m2 = mixed("rating ~ C(cond) + abs_dev_E + abs_dev_P + abs_dev_A + len_z", d)
        res["alignment_models"][item]["within_condition"] = {k: {"est": float(m2.params[k]), "se": float(m2.bse[k]), "p": float(m2.pvalues[k])}
                                                             for k in ("abs_dev_E", "abs_dev_P", "abs_dev_A", "len_z")}
    d = ok[ok["item"] == "non_sycophantic"].dropna(subset=["assert_proxy", "harsh_proxy", "imperative_rate", "exclaim_per_sent"])
    m0 = mixed("rating ~ C(cond)", d)
    m1 = mixed("rating ~ C(cond) + assert_proxy + harsh_proxy + imperative_rate + exclaim_per_sent", d)
    key = "C(cond)[T.repe_only]"
    res["assertiveness_models"] = {
        "n": int(len(d)),
        "repe_only_minus_unsteered_unadjusted": {"est": float(m0.params[key]), "se": float(m0.bse[key])},
        "repe_only_minus_unsteered_adjusted": {"est": float(m1.params[key]), "se": float(m1.bse[key]), "p": float(m1.pvalues[key])},
        "attenuation_pct": float(100 * (1 - m1.params[key] / m0.params[key])),
        "proxy_coefs": {k: {"est": float(m1.params[k]), "se": float(m1.bse[k]), "p": float(m1.pvalues[k])}
                        for k in ("assert_proxy", "harsh_proxy", "imperative_rate", "exclaim_per_sent")},
        "caveat": "proxies come from the same EPA reader/lexicons, not an independent judge"}
    return res


def oversteer(df: pd.DataFrame, measure: str) -> dict:
    out = {}
    d = df[df.source == "08"]
    for c in ("unsteered", "pe_only", "repe_only", "hybrid"):
        s = d[d.cond == c]
        o = {}
        for dm in "EPA":
            if f"{measure}_{dm}" not in s or s[f"{measure}_{dm}"].notna().sum() < 30:
                continue
            S = s.dropna(subset=[f"{measure}_{dm}"])
            sd = (S[f"{measure}_{dm}"] - S[f"target_{dm}"]).to_numpy()
            ci = cluster_bootstrap(S.scenario_id.to_numpy(), lambda ix: sd[ix].mean(), 1000)
            o[dm] = {"mean_signed_dev": ci["est"], "lo": ci["lo"], "hi": ci["hi"],
                     "mean_obs": float(S[f"{measure}_{dm}"].mean()), "mean_target": float(S[f"target_{dm}"].mean())}
        out[c] = o
    return out


def run() -> dict:
    res = {"human": human_part(), "text_features": {}, "oversteer": {}}
    for gen in ("llama3", "ministral"):
        df = RD.build(gen, with_vader=False)
        conds = [c for c in ["unsteered", "pe_only", "repe_only", "hybrid", "repe_act", "repe_const", "repe_shuffled",
                             "repe_idonly", "hybrid_const", "caa", "antisyc_prompt"] if c in set(df.cond)]
        res["text_features"][gen] = feature_table(df, conds)
        res["oversteer"][gen] = {m: oversteer(df, m) for m in ("orig", "own", "cross", "m7b")
                                 if f"{m}_E" in df and df[f"{m}_E"].notna().sum() > 100}
    save_json(res, "hybrid_human_gap.json")
    return res


if __name__ == "__main__":
    r = run()
    print(r["human"]["assertiveness_models"])
