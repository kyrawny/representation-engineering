"""Quality guardrails: repetition, length, reference-model perplexity (S11).

Questions answered: **did steering degenerate the text?**

Steering can trade off deflection reduction against quality: repeated
n-grams, collapsed length, or off-distribution text. These guardrails are
computed per condition on the generated responses:

- **repetition rate**: fraction of duplicated word n-grams (default 3);
- **length**: word count;
- **perplexity**: under a *separate* reference model (never the steered
  model — review W7.3), injected as a scorer callable ``text -> CE in
  nats``; perplexity = exp(CE).

The scorer is injectable so the CPU ``--quick`` smoke mode runs with a
deterministic mock; the real reference model is wired in the GPU phase.

CLI
---
::

    python -m evaluation.quality_guardrails --quick --out results/quality_guardrails.json
    python -m evaluation.quality_guardrails --input cond_a=responses_a.json --out results/qg.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_MODEL, DEFAULT_SEED
from evaluation import stats
from evaluation.deflection_reduction import run_mock_condition

__all__ = [
    "build_mock_scorer",
    "compute_quality",
    "load_responses",
    "main",
    "repetition_rate",
]


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of duplicated word n-grams in a text.

    Args:
        text: The text to measure.
        n: The n-gram size (default 3).

    Returns:
        A value in [0, 1]; 0.0 for texts with fewer than ``n`` words.

    Raises:
        ValueError: If ``n < 2``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def build_mock_scorer() -> Callable[[str], float]:
    """A deterministic mock reference scorer (CPU smoke mode).

    Returns:
        A callable mapping text to a pseudo cross-entropy in [1, 5) nats.
    """

    def scorer(text: str) -> float:
        digest = hashlib.sha256(f"ref:{text}".encode()).digest()
        return 1.0 + (digest[0] / 255.0) * 4.0

    return scorer


def build_model_scorer(model: Any, tokenizer: Any) -> Callable[[str], float]:
    """Build the reference-model scorer (per-token cross-entropy in nats).

    The reference model should be a model the steered model is not, so the
    perplexity guardrail does not share directions with the steering
    (WS2: separate-model robustness check).

    Args:
        model: HuggingFace causal LM used as the reference.
        tokenizer: Its tokenizer.

    Returns:
        A callable mapping text to the mean cross-entropy (nats/token).
    """
    from act_steering.readers import build_reference_scorer

    return build_reference_scorer(model, tokenizer)


def load_responses(path: Path | str) -> list[str]:
    """Load responses from a JSON list or a pipeline trace log.

    Args:
        path: A ``.json`` file — either a list of strings or a pipeline
            trace log (turns with non-null ``response`` are extracted).

    Returns:
        The response strings in file order.

    Raises:
        ValueError: If the file is neither a list nor a trace log.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "turns" in data:
        return [str(t["response"]) for t in data["turns"] if t.get("response")]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError(f"{path}: expected a JSON list of strings or a pipeline trace log")


def compute_quality(
    responses: Sequence[str],
    *,
    scorer: Callable[[str], float] | None = None,
    n_gram: int = 3,
) -> dict[str, Any]:
    """Compute quality guardrail metrics for a list of responses.

    Args:
        responses: The generated response strings.
        scorer: Optional reference-model scorer (``text -> CE in nats``);
            perplexity = exp(CE). Failures are counted, not raised.
        n_gram: N-gram size for the repetition rate.

    Returns:
        A JSON-safe dict with ``n_responses``, ``repetition_rate``,
        ``n_words`` (summary stats each), and — when a scorer is given —
        ``perplexity`` plus ``perplexity_failed``.
    """
    if not responses:
        empty = {"n": 0.0, "mean": 0.0, "std": 0.0}
        return {"n_responses": 0.0, "repetition_rate": empty, "n_words": empty}

    reps = [repetition_rate(r, n_gram) for r in responses]
    words = [len(r.split()) for r in responses]
    out: dict[str, Any] = {
        "n_responses": float(len(responses)),
        "repetition_rate": stats.summary_stats(reps),
        "n_words": stats.summary_stats(words),
    }

    if scorer is not None:
        ces: list[float] = []
        failed = 0
        for r in responses:
            try:
                ce = float(scorer(r))
            except Exception:  # noqa: BLE001 - scorer failures are counted, not raised
                failed += 1
                continue
            ces.append(ce)
        out["perplexity_failed"] = float(failed)
        if ces:
            out["perplexity"] = stats.summary_stats([float(np.exp(c)) for c in ces])
        else:
            out["perplexity"] = None
    return out


# --- CLI -----------------------------------------------------------------------


def _parse_input_arg(spec: str) -> tuple[str, Path]:
    """Parse a ``name=path`` input argument."""
    if "=" not in spec:
        raise ValueError(f"--input expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quality guardrails and write a JSON report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On invalid arguments or a missing directions-free reference model.
    """
    parser = argparse.ArgumentParser(description="Repetition / length / perplexity guardrails.")
    parser.add_argument("--out", required=True, type=Path, help="Output report JSON path")
    parser.add_argument(
        "--input", action="append", default=[], metavar="NAME=PATH",
        help="Responses file (JSON list or pipeline trace log)",
    )
    parser.add_argument("--scorer", choices=("mock", "none", "model"), default="mock")
    parser.add_argument("--quick", action="store_true", help="Smoke mode: mock pipeline responses")
    parser.add_argument("--n-gram", type=int, default=3, help="N-gram size for repetition rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-model", default=DEFAULT_MODEL,
        help="Reference HF model for the perplexity metric (model scorer); "
        "should be a model the steered model is not (WS2)",
    )
    parser.add_argument("--device", default="cuda", help="torch device (model scorer)")
    parser.add_argument("--dtype", default="float16", help="torch dtype (model scorer)")
    args = parser.parse_args(argv)

    if args.scorer == "model":
        from act_steering.directions import load_model_and_pipeline

        ref_model, ref_tokenizer, _ = load_model_and_pipeline(
            args.reference_model, args.device, args.dtype
        )
        scorer = build_model_scorer(ref_model, ref_tokenizer)
    else:
        scorer = build_mock_scorer() if args.scorer == "mock" else None

    if args.quick:
        conditions: dict[str, list[str]] = {}
        for name, k in (("unsteered", 1), ("best_of_k", 4)):
            trace = run_mock_condition(k=k, seed=args.seed)
            conditions[name] = [str(t["response"]) for t in trace["turns"] if t.get("response")]
    else:
        if not args.input:
            parser.error("provide --input NAME=PATH (or --quick)")
        parsed = list(map(_parse_input_arg, args.input))
        conditions = {name: load_responses(path) for name, path in parsed}

    report: dict[str, Any] = {
        "conditions": {
            name: compute_quality(resps, scorer=scorer, n_gram=args.n_gram)
            for name, resps in conditions.items()
        },
        "n_gram": float(args.n_gram),
        "scorer": args.scorer,
    }
    meta = stats.build_meta(
        model=None,
        seed=args.seed,
        config={"scorer": args.scorer, "n_gram": args.n_gram, "quick": args.quick},
    )
    stats.write_results(args.out, report, meta)
    for name, m in report["conditions"].items():
        ppl = m.get("perplexity")
        ppl_s = f"{ppl['mean']:.2f}" if ppl else "n/a"
        print(
            f"{name}: n={int(m['n_responses'])} rep={m['repetition_rate']['mean']:.3f} "
            f"words={m['n_words']['mean']:.1f} ppl={ppl_s}"
        )
    print(f"wrote quality guardrails report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
