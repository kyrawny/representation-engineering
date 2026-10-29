"""
Re-analysis of the original human study (WS4, WS2.5).

The original analysis pooled 672 ratings per condition and used
Kruskal–Wallis / Mann–Whitney tests, which assume independent observations.
Each participant rated every condition on every vignette, so ratings are
crossed repeated measures.  This script:

1. Builds a long-format table (participant x vignette x condition x item),
   parsing the PsyToolkit survey to recover each vignette's identities and
   the exact response text shown.
2. Links every rated response to the experiment-08 trial that produced it
   (same scenario, pair and text), attaching the automated EPA readings,
   targets and distances.  This lets us test whether automated "alignment"
   predicts human appropriateness (WS2.5).
3. Writes ``results/analysis/human_long.csv`` for ``human_clmm.R``, which fits
   cumulative link mixed models with crossed random intercepts.

Usage::

    python -m examples.act_three.experiments.analysis.human_reanalysis
    Rscript examples/act_three/experiments/analysis/human_clmm.R
"""

import re

import numpy as np
import pandas as pd

from .common import DATA_DIR, DIMS, OUT_DIR, SURVEY_DIR, ensure_dirs, load_json, save_json

# The survey vignettes were sampled from the July 2026 run (500 trials, 5
# identity pairs), NOT the August 1,300-trial run reported in Table 3.
# Established by exact text matching (53/56 responses; the rest differ only
# in quote/whitespace cleaning by generate_survey.py).
SURVEY_SOURCE_LOG = SURVEY_DIR / "08_hybrid_steering.json"

ITEMS = {1: "appropriate", 2: "natural", 3: "non_sycophantic"}
CONDS = ["unsteered", "repe_only", "pe_only", "hybrid"]
_BLOCK = re.compile(
    r"^l: likert_(?P<vig>\w+?)_(?P<cond>unsteered|repe_only|pe_only|hybrid)\s*\n"
    r"q: Imagine a conversation between an? (?P<user>[\w ]+?) and an? (?P<agent>[\w ]+?)\. .*?"
    r"Consider this response from the [\w ]+? to the [\w ]+?: \"(?P<text>.*?)\" <BR>",
    re.M | re.S)


def _norm(s: str) -> str:
    s = s.replace("end_header_id|>", "")
    return re.sub(r"\s+", " ", s.replace("\u2019", "'")).strip().lower()


def parse_survey() -> pd.DataFrame:
    txt = open(SURVEY_DIR / "survey.txt", encoding="utf-8").read()
    rows = [m.groupdict() for m in _BLOCK.finditer(txt)]
    return pd.DataFrame(rows)


def link_to_trials(blocks: pd.DataFrame) -> pd.DataFrame:
    raw = load_json(SURVEY_SOURCE_LOG)
    ident = pd.read_csv(DATA_DIR / "MTurkInteract_Identities.csv").set_index("term")
    by_key = {}
    for t in raw["trials"]:
        agent, user = t["pair"].split("_", 1)
        by_key.setdefault((t["scenario_id"], agent, user), t)
    out = []
    for _, b in blocks.iterrows():
        t = by_key.get((b.vig, b.agent, b.user))
        rec = dict(b)
        rec["linked"] = False
        if t is not None:
            c = t["conditions"][b.cond]
            shown, logged = _norm(b.text), _norm(c["text"].replace('"', "'"))
            rec["text_match"] = shown[:60] == logged[:60]
            rec["linked"] = True
            for d in DIMS:
                rec[f"target_{d[0]}"] = t["target_epa"][d]
                rec[f"obs_{d[0]}"] = c["epa"][d]
                rec[f"signed_dev_{d[0]}"] = c["epa"][d] - t["target_epa"][d]
            rec["auto_distance"] = c["mean_distance"]
            rec["agent_P"] = float(ident.loc[b.agent, "P"])
            rec["user_P"] = float(ident.loc[b.user, "P"])
        out.append(rec)
    return pd.DataFrame(out)


def build_long() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / "survey_data.csv")
    df["pid"] = [f"p{i:02d}" for i in range(len(df))]
    lik = [c for c in df.columns if c.startswith("likert_")]
    long = df.melt(id_vars=["pid", "age", "gender", "TIME_total"], value_vars=lik,
                   var_name="col", value_name="rating")
    m = long["col"].str.extract(r"likert_(?P<vig>\w+?)_(?P<cond>unsteered|repe_only|pe_only|hybrid)_(?P<item>\d)$")
    long = pd.concat([long, m], axis=1)
    long["item"] = long["item"].astype(int).map(ITEMS)
    blocks = link_to_trials(parse_survey())
    long = long.merge(blocks.drop(columns=["text"]), on=["vig", "cond"], how="left")
    return long.drop(columns=["col"])


def descriptives(long: pd.DataFrame) -> dict:
    res = {"n_participants": int(long.pid.nunique()), "n_vignettes": int(long.vig.nunique()),
           "n_ratings_per_condition_item": int(len(long) / (4 * 3)),
           "median_minutes": float(long.groupby("pid").TIME_total.first().median()),
           "means": {}}
    for item in ITEMS.values():
        sub = long[long["item"] == item]
        res["means"][item] = sub.groupby("cond").rating.mean().round(4).to_dict()
        # Participant-level means (the correct unit for a simple paired view)
        pm = sub.groupby(["pid", "cond"]).rating.mean().unstack()
        res["means"][item + "_participant_sd"] = pm.std().round(3).to_dict()
    return res


def run() -> dict:
    ensure_dirs()
    long = build_long()
    long.to_csv(OUT_DIR / "human_long.csv", index=False)
    blocks = long.drop_duplicates(["vig", "cond"])
    res = descriptives(long)
    res["n_blocks"] = int(len(blocks))
    res["n_linked"] = int(blocks.linked.sum())
    res["n_text_match"] = int(blocks.get("text_match", pd.Series(dtype=bool)).fillna(False).sum())
    res["vignettes"] = blocks.drop_duplicates("vig")[["vig", "agent", "user"]].to_dict(orient="records")
    save_json(res, "human_descriptives.json")
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1))
