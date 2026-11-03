"""
Export the self-contained input bundle for the LLM judge (``experiments/judge/inputs/``).

The generation logs the judge needs are gitignored, so the judge runs from this bundle instead: it can be
committed and run on a machine without a GPU, torch or the experiment outputs.  Every text is the first 500
characters (plan G4), the same text the cross-model readers scored.

Human-participant data: only the 56 response texts already shown in the survey and the per-response MEAN
ratings (aggregated over 48 participants) are exported; no participant-level rows.

    python -m examples.act_three.experiments.analysis.export_judge_inputs
"""
import json
from datetime import datetime

import pandas as pd

from ..scenarios import SCENARIOS
from . import rescore_data as RD
from .common import ACT_THREE_DIR, EXPERIMENTS_DIR, OUT_DIR, file_sha, load_json, strip_header_residue
from .human_reanalysis import parse_survey

BUNDLE = EXPERIMENTS_DIR / "judge" / "inputs"
REV = EXPERIMENTS_DIR / "results" / "revision"
SCN_TEXT = {s["id"]: s["text"] for s in SCENARIOS}


def _write_jsonl(name, rows):
    with open(BUNDLE / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def export_j1():
    counts = {}
    for gen in ("llama3", "ministral"):
        df = RD.build(gen, with_vader=False)
        df = df[~df.cond.str.contains("nonortho")]
        rows = []
        for (sid, pair), g in df.groupby(["scenario_id", "pair"], sort=True):
            rows.append({"trial": f"{sid}|{pair}", "generator": gen, "scenario_id": sid, "pair": pair,
                         "agent": g.agent.iloc[0], "user": g.user.iloc[0], "user_msg": SCN_TEXT[sid],
                         "responses": {c: t for c, t in zip(g.cond, g.text500)}})
        counts[gen] = {"trials": _write_jsonl(f"j1_trials_{gen}.jsonl", rows),
                       "texts": int(sum(len(r["responses"]) for r in rows)),
                       "conditions": sorted(df.cond.unique())}
    return counts


def export_ays():
    rows = [{"key": r["key"], "item_id": r["item_id"], "condition": r["condition"], "question": r["question"],
             "gold": r["correct_answer"], "aliases": r["gold_aliases"], "response_1": r["response_1"],
             "response_2": r["response_2"]} for r in _read_jsonl(REV / "llama3" / "16_are_you_sure.jsonl")]
    return _write_jsonl("ays_records.jsonl", rows)


def export_multiturn():
    rows = []
    for r in _read_jsonl(REV / "llama3" / "18_multiturn.jsonl"):
        rows.append({"key": r["key"], "scenario_id": r["scenario_id"], "pair": r["pair"], "agent": r["agent"],
                     "user": r["user"], "category": r["category"], "condition": r["condition"],
                     "turns": [{"turn": t["turn"], "user_msg": t["user_msg"], "text": t["text500"]} for t in r["turns"]]})
    return _write_jsonl("multiturn.jsonl", rows)


def export_aita():
    """Open-ended ELEPHANT AITA-YTA responses (full text, at most 256 tokens) plus the Reddit top comment of each
    post as condition 'human'.  Skipped if the generation log does not exist yet."""
    p = REV / "llama3" / "20_elephant_aita.jsonl"
    if not p.exists():
        return None
    gen = [r for r in _read_jsonl(p) if r["mode"] == "open"]
    src = pd.read_csv(REV / "data" / "elephant" / "AITA-YTA.csv", encoding_errors="replace").set_index("Unnamed: 0")
    rows = [{"key": r["key"], "post_id": r["post_id"], "condition": r["condition"], "post": r["user_msg"],
             "response": strip_header_residue(r["response"])} for r in gen]
    for pid, srow, post in sorted({(r["post_id"], r["src_row"], r["user_msg"]) for r in gen}):
        rows.append({"key": f"{pid}|reddit|human", "post_id": pid, "condition": "human", "post": post,
                     "response": str(src.loc[srow, "top_comment"])})
    return _write_jsonl("aita_records.jsonl", rows)


def export_human():
    blocks = parse_survey()
    rows = []
    for vig, g in blocks.groupby("vig", sort=True):
        rows.append({"vig": vig, "agent": g.agent.iloc[0], "user": g.user.iloc[0], "user_msg": SCN_TEXT[vig],
                     "responses": {c: t for c, t in zip(g.cond, g.text)}})
    n = _write_jsonl("human_vignettes.jsonl", rows)
    long = pd.read_csv(OUT_DIR / "human_long.csv")
    means = long.groupby(["vig", "cond", "item"]).rating.agg(["mean", "std", "count"]).reset_index()
    means.to_csv(BUNDLE / "human_response_means.csv", index=False)
    return {"vignettes": n, "response_item_means": int(len(means))}


def export_calibration():
    items = load_json(ACT_THREE_DIR / "models" / "llama3" / "epa_tuning_dataset_test.json")["utterances"]
    rows = [{"id": i, "text": x["text"], "E": x["target_epa"]["e"], "P": x["target_epa"]["p"], "A": x["target_epa"]["a"],
             "behavior": x.get("behavior")} for i, x in enumerate(items)]
    rr = load_json(OUT_DIR / "revision_results.json")["agreement"]
    val = {"llama3_table1": {d[0].upper(): v["pearson_r"] for d, v in rr["table1_llama_source_01_reading_quality"].items()},
           "ministral": {d[0].upper(): v["pearson"] for d, v in rr["reader_calibration_test"]["ministral"].items()},
           "mistral7b": {d[0].upper(): v["pearson"] for d, v in rr["reader_calibration_test"]["mistral7b"].items()}}
    (BUNDLE / "reader_validity.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    return _write_jsonl("calibration_utterances.jsonl", rows)


def run():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    man = {"created": datetime.now().isoformat(timespec="seconds"), "text_rule": "first 500 characters of every text (plan G4)",
           "j1": export_j1(), "ays_records": export_ays(), "multiturn_conversations": export_multiturn(),
           "human": export_human(), "calibration_utterances": export_calibration(), "aita_records": export_aita(),
           "sources": {"llama3_08": file_sha(EXPERIMENTS_DIR / "results" / "08_hybrid_steering.json"),
                       "ministral_08": file_sha(EXPERIMENTS_DIR / "results" / "ministral" / "08_hybrid_steering.json")}}
    (BUNDLE / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
