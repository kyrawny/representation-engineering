"""Dose-response analysis: steering strength vs shift and quality (S11).

Question answered: **how much steering strength is usable before the text
degenerates?** (review WS1.3: coefficient vs EPA shift, reference-model
perplexity, repetition rate — with the degeneration threshold marked)

The steering strength (alpha) is swept and an injected **probe** measures,
per alpha, the mean per-dimension reader shift, the reference-model
perplexity, and the repetition rate. A sweep is *degenerate* at an alpha
when the perplexity grows beyond ``baseline * (1 + max_ppl_increase)`` or
the repetition rate exceeds ``max_repetition``; the report marks
``degenerated_at`` (first degenerate alpha) and ``safe_alpha_max`` (largest
healthy alpha), plus per-dimension shift slopes in the healthy region.

The probe is injected so the CPU ``--quick`` smoke mode runs with a
deterministic synthetic curve; the real probe (steered model + reference
model) is wired in the GPU phase.

CLI
---
::

    python -m evaluation.dose_response --quick --out results/dose_response.json
    python -m evaluation.dose_response --probe mock --alphas 0 0.5 1 1.5 2 --out ...
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_MODEL, DEFAULT_SEED
from evaluation import stats

__all__ = [
    "ProbeResult",
    "analyze_sweep",
    "build_mock_probe",
    "build_model_probe",
    "main",
    "sweep",
]

#: Per-dimension names for shift curves.
DIMENSIONS: tuple[str, str, str] = ("E", "P", "A")


@dataclass(frozen=True)
class ProbeResult:
    """Measured metrics at one steering strength.

    Attributes:
        epa_shift: Mean per-dimension reader shift (E, P, A) vs alpha=0.
        perplexity: Reference-model perplexity of the steered outputs.
        repetition_rate: Mean word n-gram repetition rate.
    """

    epa_shift: tuple[float, float, float]
    perplexity: float
    repetition_rate: float


def build_mock_probe() -> Callable[[float], ProbeResult]:
    """A deterministic synthetic probe for the CPU smoke mode.

    Saturating per-dimension shifts, quadratic perplexity growth and
    repetition growth; degenerates (at default thresholds) around alpha 1.5.
    """

    def probe(alpha: float) -> ProbeResult:
        shifts = (
            math.tanh(0.8 * alpha),
            math.tanh(0.6 * alpha),
            math.tanh(1.1 * alpha),
        )
        ppl = 10.0 * math.exp(0.25 * alpha * alpha)
        rep = min(0.9, 0.05 + 0.35 * alpha * alpha)
        return ProbeResult(shifts, ppl, rep)

    return probe


def build_model_probe(
    model: Any,
    tokenizer: Any,
    bundle: Any,
    reference_scorer: Callable[[str], float],
    *,
    prompts: Sequence[str],
    u_pre: float = 1.0,
    max_new_tokens: int = 64,
    steering: Any = None,
    steerer_cls: type = None,  # type: ignore[assignment]
) -> Callable[[float], ProbeResult]:
    """Build the real dose-response probe (GPU phase).

    One steerer wraps the model for the whole sweep; the per-alpha gain is
    set by mutating ``steerer.config.alpha`` (the offsets are computed from
    the config at every call). The EPA shift is measured against the
    alpha=0 baseline (captured on the first zero-alpha call); perplexity is
    the reference model's per-token cross-entropy exponentiated; the
    repetition rate is the mean word n-gram repetition of the responses.

    Args:
        model: HuggingFace causal LM (mutated by the steerer wrapper).
        tokenizer: Its tokenizer.
        bundle: The directions artifact bundle (S6).
        reference_scorer: ``text -> CE in nats/token`` (a model the
            steered model is not, per the WS2 measurement stack).
        prompts: Fixed probe prompts (the sweep keeps the inputs
            constant and only varies the steering strength).
        u_pre: Event-1 deflection driving the proportional gain.
        max_new_tokens: Generation cap per probe response.
        steering: Base steering settings (defaults to ``SteeringConfig()``;
            its ``alpha`` is overridden per sweep point).
        steerer_cls: Steerer class (injectable for tests; defaults to
            :class:`act_steering.steerer.DeflectionSteerer`).

    Returns:
        A probe callable ``alpha -> ProbeResult``.

    Raises:
        ValueError: If ``prompts`` is empty.
    """
    from act_steering.config import SteeringConfig
    from act_steering.readers import build_epa_reader
    from act_steering.steerer import DeflectionSteerer
    from evaluation.quality_guardrails import repetition_rate

    if not prompts:
        raise ValueError("build_model_probe: prompts must not be empty")
    if steerer_cls is None:
        steerer_cls = DeflectionSteerer

    steerer = steerer_cls(model, tokenizer, bundle, steering or SteeringConfig())
    reader = build_epa_reader(model, tokenizer, bundle)
    baseline: tuple[float, float, float] | None = None

    def _responses(alpha: float) -> list[str]:
        steerer.config.alpha = float(alpha)
        out: list[str] = []
        for prompt in prompts:
            full = steerer.generate(
                prompt, u_pre=u_pre, max_new_tokens=max_new_tokens,
                do_sample=False, return_text=True,
            )
            text = full[0] if isinstance(full, (list, tuple)) else full
            if isinstance(text, dict):
                text = text.get("generated_text", "")
            if isinstance(text, str) and text.startswith(prompt):
                text = text[len(prompt):]
            out.append(str(text))
        return out

    def probe(alpha: float) -> ProbeResult:
        nonlocal baseline
        responses = _responses(alpha)
        eps = [reader(r) for r in responses]
        means = (
            float(np.mean([e.e for e in eps])),
            float(np.mean([e.p for e in eps])),
            float(np.mean([e.a for e in eps])),
        )
        if alpha == 0.0 and baseline is None:
            baseline = means
        if baseline is None:
            base_responses = _responses(0.0)
            base_eps = [reader(r) for r in base_responses]
            baseline = (
                float(np.mean([e.e for e in base_eps])),
                float(np.mean([e.p for e in base_eps])),
                float(np.mean([e.a for e in base_eps])),
            )
            steerer.config.alpha = float(alpha)
        shifts = (means[0] - baseline[0], means[1] - baseline[1], means[2] - baseline[2])
        ce = float(np.mean([reference_scorer(r) for r in responses]))
        rep = float(np.mean([repetition_rate(r) for r in responses]))
        return ProbeResult(shifts, math.exp(ce), rep)

    return probe


def sweep(probe: Callable[[float], ProbeResult], alphas: Sequence[float]) -> list[dict[str, Any]]:
    """Run the probe at each alpha (sorted), collecting metrics or errors.

    Args:
        probe: The probe callable (``alpha -> ProbeResult``).
        alphas: The steering strengths to sweep.

    Returns:
        Rows with ``alpha``, ``ok`` and — when ok — ``epa_shift`` (E, P, A),
        ``perplexity``, ``repetition_rate``; otherwise ``error``.

    Raises:
        ValueError: If the sweep does not include the alpha=0 baseline.
    """
    if 0.0 not in list(alphas):
        raise ValueError("the sweep must include alpha=0 (the baseline)")
    rows: list[dict[str, Any]] = []
    for alpha in sorted(alphas):
        try:
            result = probe(alpha)
        except Exception as exc:  # noqa: BLE001 - probe failures are recorded per row
            rows.append({"alpha": float(alpha), "ok": False, "error": str(exc)})
            continue
        rows.append(
            {
                "alpha": float(alpha),
                "ok": True,
                "epa_shift": [float(v) for v in result.epa_shift],
                "perplexity": float(result.perplexity),
                "repetition_rate": float(result.repetition_rate),
            }
        )
    return rows


def analyze_sweep(
    rows: list[dict[str, Any]],
    *,
    max_ppl_increase: float = 0.5,
    max_repetition: float = 0.5,
) -> dict[str, Any]:
    """Mark degeneration and fit healthy-region shift slopes.

    Args:
        rows: The sweep rows from :func:`sweep` (must include an ok alpha=0).
        max_ppl_increase: Relative perplexity growth allowed over baseline.
        max_repetition: Maximum allowed repetition rate.

    Returns:
        A dict with ``rows`` (each ok row annotated with ``degenerated``),
        ``degeneration`` (``degenerated_at``, ``safe_alpha_max``,
        ``thresholds``), and ``per_dimension_slopes`` (healthy-region
        linear fit of shift vs alpha, or None with fewer than 2 points).

    Raises:
        ValueError: If no ok alpha=0 baseline row is present.
    """
    baseline = next(
        (r for r in rows if r.get("ok") and r["alpha"] == 0.0), None
    )
    if baseline is None:
        raise ValueError("sweep rows must include an ok alpha=0 baseline")
    ppl_threshold = baseline["perplexity"] * (1.0 + max_ppl_increase)

    annotated: list[dict[str, Any]] = []
    degenerated_at: float | None = None
    safe_alpha_max = 0.0
    healthy: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if not row.get("ok"):
            annotated.append(row)
            continue
        if row["alpha"] == 0.0:
            row["degenerated"] = False
        else:
            row["degenerated"] = (
                row["perplexity"] > ppl_threshold or row["repetition_rate"] > max_repetition
            )
        if row["degenerated"]:
            if degenerated_at is None:
                degenerated_at = row["alpha"]
        else:
            safe_alpha_max = max(safe_alpha_max, row["alpha"])
            healthy.append(row)
        annotated.append(row)

    slopes: dict[str, float | None] = {}
    for d, dim in enumerate(DIMENSIONS):
        if len(healthy) >= 2:
            xs = np.array([r["alpha"] for r in healthy], dtype=np.float64)
            ys = np.array([r["epa_shift"][d] for r in healthy], dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0])
            slopes[dim] = slope
        else:
            slopes[dim] = None

    return {
        "rows": annotated,
        "degeneration": {
            "degenerated_at": degenerated_at,
            "safe_alpha_max": float(safe_alpha_max),
            "thresholds": {
                "max_ppl_increase": float(max_ppl_increase),
                "max_repetition": float(max_repetition),
                "ppl_threshold": float(ppl_threshold),
            },
        },
        "per_dimension_slopes": slopes,
    }


# --- CLI -----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dose-response sweep and write a JSON report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On invalid arguments or a missing prompts file.
    """
    parser = argparse.ArgumentParser(
        description="Steering dose-response sweep and degeneration detection."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output report JSON path")
    parser.add_argument("--probe", choices=("mock", "model"), default="mock")
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        help="Steering strengths (must include 0)",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke mode: mock probe")
    parser.add_argument("--max-ppl-increase", type=float, default=0.5)
    parser.add_argument("--max-repetition", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Steered HF model id (model probe)")
    parser.add_argument(
        "--directions", type=Path, default=None,
        help="Directions .npz (default: artifacts/directions_<model>.npz)",
    )
    parser.add_argument(
        "--reference-model", default=None,
        help="Reference HF model for the perplexity metric (default: same as --model)",
    )
    parser.add_argument("--device", default="cuda", help="torch device (model probe)")
    parser.add_argument("--dtype", default="float16", help="torch dtype (model probe)")
    parser.add_argument(
        "--prompts", type=Path, default=None,
        help="JSON list of probe prompts (required for --probe model)",
    )
    parser.add_argument(
        "--u-pre", type=float, default=1.0, help="Event-1 deflection for the gain (model probe)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Generation cap per probe response (model probe)",
    )
    args = parser.parse_args(argv)

    if args.probe == "model":
        if args.prompts is None:
            raise SystemExit("--probe model requires --prompts PATH (JSON list of prompts)")
        import json

        from act_steering.directions import (
            DirectionsBundle,
            load_model_and_pipeline,
            resolve_directions_paths,
        )
        from evaluation.quality_guardrails import build_model_scorer

        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise SystemExit("--prompts must be a JSON list of strings")
        npz_path, json_path = resolve_directions_paths(args.directions, args.model)
        if not npz_path.exists():
            raise SystemExit(
                f"directions artifact {npz_path} not found; run "
                f"`python -m act_steering.directions --model {args.model}` first"
            )
        model, tokenizer, _ = load_model_and_pipeline(args.model, args.device, args.dtype)
        bundle = DirectionsBundle.load(npz_path, json_path)
        ref_id = args.reference_model or args.model
        ref_model, ref_tokenizer, _ = load_model_and_pipeline(ref_id, args.device, args.dtype)
        scorer = build_model_scorer(ref_model, ref_tokenizer)
        probe = build_model_probe(
            model, tokenizer, bundle, scorer,
            prompts=prompts, u_pre=args.u_pre, max_new_tokens=args.max_new_tokens,
        )
    else:
        probe = build_mock_probe()

    alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0] if args.quick else args.alphas

    rows = sweep(probe, alphas)
    report = analyze_sweep(
        rows, max_ppl_increase=args.max_ppl_increase, max_repetition=args.max_repetition
    )
    meta = stats.build_meta(
        model=args.model if args.probe == "model" else None,
        seed=args.seed,
        config={"probe": args.probe, "alphas": alphas, "quick": args.quick},
    )
    stats.write_results(args.out, report, meta)
    deg = report["degeneration"]
    slope_str = ", ".join(
        f"{k}: {v:.3f}" if v is not None else f"{k}: n/a"
        for k, v in report["per_dimension_slopes"].items()
    )
    print(
        f"degenerated_at={deg['degenerated_at']} "
        f"safe_alpha_max={deg['safe_alpha_max']:.2f} slopes={{ {slope_str} }}"
    )
    print(f"wrote dose response report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
