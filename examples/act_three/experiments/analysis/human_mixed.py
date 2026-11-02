"""
Human-study reanalysis (plan T7.2): linear mixed model with crossed random effects,
participant-level Friedman / Wilcoxon (Holm), Krippendorff alpha, ICC, and a comparison with the
original pooled Kruskal-Wallis / Mann-Whitney claims.

    python -m examples.act_three.experiments.analysis.human_mixed
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .common import OUT_DIR, save_json
from .metrics import holm

CONDS = ["unsteered", "pe_only", "repe_only", "hybrid"]
ITEMS = ["appropriate", "natural", "non_sycophantic"]


def lmm(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["cond"] = pd.Categorical(d["cond"], CONDS)
    d["one"] = 1
    m = smf.mixedlm("rating ~ C(cond, Treatment('unsteered'))", d, groups="one",
                    vc_formula={"pid": "0 + C(pid)", "vig": "0 + C(vig)"}).fit(reml=True)
    coefs = {}
    for name in m.params.index:
        if name.startswith("C(cond"):
            k = name.split("[T.")[1].rstrip("]")
            lo, hi = m.conf_int().loc[name]
            coefs[k] = {"est": float(m.params[name]), "se": float(m.bse[name]), "lo": float(lo), "hi": float(hi),
                        "p": float(m.pvalues[name])}
    # all pairwise contrasts via the fixed-effect covariance
    b = m.fe_params
    cov = m.cov_params().loc[b.index, b.index]
    def vec(c):
        v = pd.Series(0.0, index=b.index)
        for n in b.index:
            if c != "unsteered" and n.endswith(f"[T.{c}]"):
                v[n] = 1.0
        return v
    pair = {}
    ps = []
    for a, c in itertools.combinations(CONDS, 2):
        v = vec(c) - vec(a)
        est = float(v @ b); se = float(np.sqrt(v @ cov.values @ v))
        z = est / se; p = float(2 * (1 - stats.norm.cdf(abs(z))))
        pair[f"{c}-{a}"] = {"est": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se, "p": p}
        ps.append(p)
    for k, pa in zip(pair, holm(ps)):
        pair[k]["p_holm"] = float(pa)
    var = {k: float(v) for k, v in zip(m.model.exog_vc.names, np.asarray(m.vcomp))}
    resid = float(m.scale)
    tot = sum(var.values()) + resid
    return {"coef_vs_unsteered": coefs, "pairwise": pair, "variance": var, "residual": resid,
            "icc_pid": var.get("pid", np.nan) / tot, "icc_vig": var.get("vig", np.nan) / tot}


def participant_level(d: pd.DataFrame) -> dict:
    pm = d.groupby(["pid", "cond"]).rating.mean().unstack()[CONDS]
    fr = stats.friedmanchisquare(*[pm[c] for c in CONDS])
    ps, keys = [], []
    out = {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "n": int(len(pm)),
           "means": pm.mean().to_dict(), "pairwise": {}}
    for a, c in itertools.combinations(CONDS, 2):
        w = stats.wilcoxon(pm[c], pm[a])
        diff = pm[c] - pm[a]
        boot = np.random.default_rng(0).choice(diff.to_numpy(), (2000, len(diff))).mean(1)
        out["pairwise"][f"{c}-{a}"] = {"mean_diff": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)),
                                       "hi": float(np.percentile(boot, 97.5)), "W": float(w.statistic),
                                       "p": float(w.pvalue)}
        ps.append(w.pvalue); keys.append(f"{c}-{a}")
    for k, pa in zip(keys, holm(ps)):
        out["pairwise"][k]["p_holm"] = float(pa)
    return out


def alpha_ordinal(d: pd.DataFrame) -> float:
    import krippendorff
    mat = d.assign(unit=d.vig + "|" + d.cond).pivot_table(index="pid", columns="unit", values="rating").to_numpy()
    return float(krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal"))


def original_claims(d: pd.DataFrame) -> dict:
    """The pooled tests as originally reported (independence assumed) -- for the comparison table."""
    out = {}
    for item in ITEMS:
        s = d[d["item"] == item]
        groups = [s[s.cond == c].rating for c in CONDS]
        kw = stats.kruskal(*groups)
        out[item] = {"H": float(kw.statistic), "p": float(kw.pvalue), "mw": {}}
        for a, c in itertools.combinations(CONDS, 2):
            u = stats.mannwhitneyu(s[s.cond == c].rating, s[s.cond == a].rating)
            out[item]["mw"][f"{c}-{a}"] = float(u.pvalue)
    return out


def run() -> dict:
    d = pd.read_csv(OUT_DIR / "human_long.csv")
    res = {"lmm": {}, "participant_level": {}, "reliability": {}, "original_pooled": original_claims(d)}
    warnings.filterwarnings("ignore")
    for item in ITEMS:
        s = d[d["item"] == item]
        res["lmm"][item] = lmm(s)
        res["participant_level"][item] = participant_level(s)
        res["reliability"][item] = {"krippendorff_alpha_ordinal": alpha_ordinal(s)}
    save_json(res["participant_level"], "human_participant_level.json")
    save_json(res, "human_mixed.json")
    return res


if __name__ == "__main__":
    import json
    r = run()
    for item in ITEMS:
        print(item, "LMM vs unsteered:",
              {k: (round(v["est"], 2), round(v["p"], 4)) for k, v in r["lmm"][item]["coef_vs_unsteered"].items()})
        print("   participant-level friedman p =", r["participant_level"][item]["friedman_p"],
              "alpha =", round(r["reliability"][item]["krippendorff_alpha_ordinal"], 3),
              "icc pid/vig =", round(r["lmm"][item]["icc_pid"], 3), round(r["lmm"][item]["icc_vig"], 3))
