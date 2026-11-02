"""
Stage 6: coherence of the main-run outputs (paper "Coherence Analysis").

Self-perplexity of each condition's responses under the *unsteered* model, on a
fixed random sample of trials (default 300, seed 0), plus text metrics (length,
unique-token ratio, bigram repetition) on every trial.  Optionally adds GPT-2
perplexity as a model-independent reference.

Usage::

    python -m examples.act_three.scale70b.s06_coherence --model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse

import numpy as np

from . import settings as S
from .common import coherence_api, geo_mean, load_model, read_jsonl, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-ppl", type=int, default=300)
    ap.add_argument("--gpt2", action="store_true")
    args = ap.parse_args()

    files = sorted((S.results_dir(args.model) / args.tag).glob("shard_*.jsonl"))
    trials = sorted({r["key"]: r for r in read_jsonl(files)}.values(), key=lambda r: r["index"])
    if not trials:
        raise SystemExit("No main-run outputs found.")
    conds = list(trials[0]["conditions"].keys())
    ppl_fn, text_metrics, gpt2_fn = coherence_api()

    metrics = {c: [text_metrics(t["conditions"][c]["text"]) for t in trials] for c in conds}
    summary = {c: {k: float(np.mean([m[k] for m in metrics[c] if isinstance(m.get(k), (int, float))]))
                   for k in metrics[c][0] if isinstance(metrics[c][0][k], (int, float))} for c in conds}

    rng = np.random.default_rng(0)
    sample = [trials[i] for i in sorted(rng.choice(len(trials), size=min(args.n_ppl, len(trials)), replace=False))]
    model, tok = load_model(args.model)
    for c in conds:
        ppl = ppl_fn(model, tok, [t["conditions"][c]["text"] for t in sample])
        summary[c]["self_ppl_geo"] = geo_mean(ppl)
        summary[c]["self_ppl_median"] = float(np.median([p for p in ppl if np.isfinite(p)]))
        if args.gpt2 and gpt2_fn is not None:
            g = gpt2_fn([t["conditions"][c]["text"] for t in sample])
            summary[c]["gpt2_ppl_geo"] = geo_mean(g)
    for c in conds:
        if c != "unsteered":
            summary[c]["self_ppl_ratio_vs_unsteered"] = summary[c]["self_ppl_geo"] / summary["unsteered"]["self_ppl_geo"]

    path = save_json({"model": args.model, "n_trials": len(trials), "n_ppl": len(sample), "summary": summary},
                     S.results_dir(args.model) / f"coherence_{args.tag}.json")
    print(f"Saved {path}")
    for c in conds:
        print(c, {k: round(v, 3) for k, v in summary[c].items()})


if __name__ == "__main__":
    main()
