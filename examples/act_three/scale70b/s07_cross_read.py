"""
Stage 7: score responses with a reader built on a DIFFERENT model.

Reviewers objected that the same reader both sets the steering targets and
scores the outputs.  This stage removes that overlap in both directions:

  * 70B outputs scored by the paper's Llama-3.1-8B reader (fits on any 24 GB GPU):
        --reader-model meta-llama/Llama-3.1-8B-Instruct --generator meta-llama/Llama-3.1-70B-Instruct
  * The paper's 8B outputs scored by the 70B reader:
        --reader-model meta-llama/Llama-3.1-70B-Instruct --generator meta-llama/Llama-3.1-8B-Instruct

Every condition is scored on its first 500 characters, because that is all the
8B logs kept.  The targets are unchanged; only the measurement of the output
changes.

Output: results/scale/<generator>/read_by_<reader>.jsonl
"""

import argparse

from . import settings as S
from .common import JsonlLog, load_components, load_json, read_jsonl, safe_read_batch_size


def load_generator_trials(generator: str, tag: str):
    if S.short_name(generator) == "llama3":
        d = load_json(S.REFERENCE_8B_LOG)
        out = []
        for i, t in enumerate(d["trials"]):
            out.append({"key": f"{t['scenario_id']}|{t['pair']}", "index": i,
                        "texts": {c: v["text"][:S.TEXT500] for c, v in t["conditions"].items()}})
        return out
    files = sorted((S.results_dir(generator) / tag).glob("shard_*.jsonl"))
    recs = sorted({r["key"]: r for r in read_jsonl(files)}.values(), key=lambda r: r["index"])
    return [{"key": r["key"], "index": r["index"],
             "texts": {c: v["text"][:S.TEXT500] for c, v in r["conditions"].items()}} for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-model", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--read-batch-size", type=int, default=16)
    args = ap.parse_args()
    if S.short_name(args.reader_model) == S.short_name(args.generator):
        print("NOTE: reader == generator. This is the own reader on the first 500 characters "
              "(a like-for-like control for the cross reads), not an independent measure.")

    trials = load_generator_trials(args.generator, args.tag)
    out = S.results_dir(args.generator) / f"read_by_{S.short_name(args.reader_model)}.jsonl"
    log = JsonlLog(out)
    todo = [t for t in trials if t["key"] not in log.done]
    print(f"{len(todo)} of {len(trials)} trials to score -> {out}")
    comp = load_components(args.reader_model, need_steerer=False)
    comp.fast_reader.batch_size = safe_read_batch_size(args.reader_model, args.read_batch_size)
    chunk = 8
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        flat = [(t, c, txt) for t in batch for c, txt in t["texts"].items()]
        epas = comp.fast_reader.read([x[2] for x in flat])
        per = {}
        for (t, c, _), e in zip(flat, epas):
            per.setdefault(t["key"], {})[c] = e
        for t in batch:
            log.append({"key": t["key"], "index": t["index"], "reader": args.reader_model,
                        "generator": args.generator, "epa_text500": per[t["key"]]})
        print(f"{min(i + chunk, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
