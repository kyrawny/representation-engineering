"""
T5 -- cross-model re-scoring.  Every text (``text500`` for ALL conditions, plan G4) is read by
readers built on other models (and by the own reader, for a like-for-like comparison).

Reader assignment
    llama3 texts      -> ministral (E,P,A), mistral7b (E,A)   [+ llama3 own reader]
    ministral texts   -> llama3 (E,P,A),    mistral7b (E,A)   [+ ministral own reader]
    gemini texts      -> llama3, ministral, mistral7b
The Mistral-7B P reader has 0 layers (r = NaN in its tuning file): P is always NaN for it.

    python -m examples.act_three.experiments.revision.17_cross_reader_rescore --reader ministral [--quick]
"""
import argparse
import json
import math

from tqdm import tqdm

from ..analysis.common import RESULTS_DIR
from . import common as R

REV = R.REV_RESULTS
READER_FOR = {
    "llama3": {"llama3", "ministral", "gemini"},
    "ministral": {"llama3", "ministral", "gemini"},
    "mistral7b": {"llama3", "ministral", "gemini"},
}


def collect_texts():
    items = []
    for gen in ("llama3", "ministral"):
        raw = R.load_08(gen)
        for t in raw["trials"]:
            for cond, c in t["conditions"].items():
                items.append({"generator": gen, "key": R.trial_key(t, cond), "condition": cond,
                              "source": "08", "text500": c["text"][:500]})
        d = REV / gen
        for p in sorted(d.glob("*.jsonl")):
            if "quick" in p.name or p.name.startswith("16_"):
                continue
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r.get("text500") is not None and r.get("scenario_id"):
                        items.append({"generator": gen, "key": r["key"], "condition": r["condition"],
                                      "source": p.name, "text500": r["text500"]})
    g = json.load(open(RESULTS_DIR / "gemini" / "13_gemini_generations.json", encoding="utf-8"))
    for t in g["trials"]:
        for cond, c in t["conditions"].items():
            items.append({"generator": "gemini", "key": R.trial_key(t, cond), "condition": cond,
                          "source": "13_gemini", "text500": c["text"][:500]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=["llama3", "ministral", "mistral7b"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    items = [i for i in collect_texts() if i["generator"] in READER_FOR[args.reader]
             or args.reader == "mistral7b"]
    if args.quick:
        items = items[:3] + [i for i in items if i["generator"] == "gemini"][:2]
    out = REV / ("rescore_%s_quick.jsonl" % args.reader if args.quick else "rescore_%s.jsonl" % args.reader)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["generator"], r["key"]))
    todo = [i for i in items if (i["generator"], i["key"]) not in done]
    print(f"{len(items)} texts for reader {args.reader}; {len(todo)} to read")
    if not todo:
        return
    comp = R.load_components(args.reader, load_steerer=False)
    with open(out, "a", encoding="utf-8") as f:
        for s in tqdm(range(0, len(todo), args.batch)):
            chunk = todo[s:s + args.batch]
            texts = [c["text500"] if c["text500"].strip() else "." for c in chunk]
            epas = R.read_all(comp, texts, args.batch)
            for c, e in zip(chunk, epas):
                P = float("nan") if args.reader == "mistral7b" else e["potency"]
                rec = {"key": c["key"], "generator": c["generator"], "condition": c["condition"],
                       "source": c["source"], "reader": args.reader,
                       "E": e["evaluation"], "P": None if math.isnan(P) else P, "A": e["activity"],
                       "empty": not c["text500"].strip()}
                f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
