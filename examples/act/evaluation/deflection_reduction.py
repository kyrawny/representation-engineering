"""Deflection-reduction analysis over pipeline trace logs (S11).

Question answered: **does steering / best-of-k actually reduce per-turn
deflection U?**

Input is the JSON trace logs written by ``act_steering.pipeline`` (one log
per condition, e.g. unsteered / steered / best-of-k). For each turn the
chosen response's post-event deflection ``u_post`` is extracted; conditions
are paired by (scenario, turn) and the paired difference
``u_treatment - u_baseline`` gets a cluster-bootstrap confidence interval
with the scenario as cluster (review W4: ratings within a scenario are not
independent).

CLI
---
::

    python -m evaluation.deflection_reduction --quick --out results/deflection_reduction.json
    python -m evaluation.deflection_reduction --log unsteered=trace_u.json --log b=trace_b.json \\
        --baseline unsteered --treatments b --out results/deflection_reduction.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_SEED, DataPaths, PipelineConfig
from act_steering.dictionary import ACTDictionary
from act_steering.pipeline import (
    QUICK_SCRIPT,
    ACTCoefficients,
    ACTSteeringPipeline,
    build_mock_components,
)
from act_steering.state import ConversationState
from evaluation import stats

__all__ = [
    "TurnRecord",
    "compare_conditions",
    "extract_turns",
    "load_trace",
    "main",
    "run_mock_condition",
    "summarize_condition",
]


@dataclass(frozen=True)
class TurnRecord:
    """One turn's deflection values extracted from a trace log.

    Attributes:
        turn: The turn index in the log.
        scenario: The scenario identifier (user message text by default).
        u_pre: The event-1 deflection for the turn.
        u_post: The chosen response's post-event deflection (None if the
            turn produced no scorable response).
    """

    turn: int
    scenario: str
    u_pre: float
    u_post: float | None


def load_trace(path: Path | str) -> dict:
    """Load a pipeline trace log from disk.

    Raises:
        ValueError: If the file is not a pipeline trace log (no "turns").
    """
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    if "turns" not in trace:
        raise ValueError(f"{path}: not a pipeline trace log (missing 'turns')")
    return trace


def extract_turns(trace: dict) -> list[TurnRecord]:
    """Extract per-turn deflection records from a trace log.

    Args:
        trace: A pipeline trace log (``load_trace`` output).

    Returns:
        The turn records in log order.
    """
    records: list[TurnRecord] = []
    for entry in trace.get("turns", []):
        turn = int(entry.get("turn", len(records)))
        scenario = str(entry.get("scenario", entry.get("user_message", f"turn-{turn}")))
        u_pre = entry.get("u_pre")
        u_post = entry.get("u_post")
        records.append(
            TurnRecord(
                turn=turn,
                scenario=scenario,
                u_pre=float(u_pre) if u_pre is not None else 0.0,
                u_post=float(u_post) if u_post is not None else None,
            )
        )
    return records


def summarize_condition(trace: dict) -> dict[str, float]:
    """Summarize one condition's trace log.

    Args:
        trace: The condition's pipeline trace log.

    Returns:
        A dict with ``n_turns``, ``u_post_mean``, ``u_post_std``,
        ``u_pre_mean``.
    """
    records = extract_turns(trace)
    u_posts = [r.u_post for r in records if r.u_post is not None]
    return {
        "n_turns": float(len(records)),
        "u_post_mean": float(np.mean(u_posts)) if u_posts else 0.0,
        "u_post_std": float(np.std(u_posts, ddof=1)) if len(u_posts) > 1 else 0.0,
        "u_pre_mean": float(np.mean([r.u_pre for r in records])) if records else 0.0,
    }


def compare_conditions(
    traces: dict[str, dict],
    *,
    baseline: str,
    treatments: Sequence[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare treatment conditions against a baseline on per-turn U.

    Pairs turns by (scenario, turn); the paired difference
    ``u_treatment - u_baseline`` is analysed with the cluster bootstrap
    (scenario = cluster).

    Args:
        traces: Condition name -> pipeline trace log.
        baseline: The baseline condition name.
        treatments: The treatment condition names.
        n_bootstrap: Bootstrap resamples per comparison.
        alpha: Significance level for the percentile interval.
        seed: RNG seed.

    Returns:
        A JSON-safe report with ``baseline``, ``comparisons`` (one entry
        per treatment with the cluster-bootstrap result), ``conditions``
        (per-condition summaries), ``n_bootstrap``, ``alpha``.

    Raises:
        ValueError: If the baseline or a treatment is not in ``traces``.
    """
    if baseline not in traces:
        raise ValueError(f"baseline {baseline!r} not in traces")
    for t in treatments:
        if t not in traces:
            raise ValueError(f"treatment {t!r} not in traces")

    baseline_records = extract_turns(traces[baseline])
    baseline_index = {(r.scenario, r.turn): r for r in baseline_records}

    comparisons: list[dict[str, Any]] = []
    for name in treatments:
        treatment_records = extract_turns(traces[name])
        treatment_index = {(r.scenario, r.turn): r for r in treatment_records}
        deltas: list[float] = []
        clusters: list[str] = []
        for key, b_rec in baseline_index.items():
            t_rec = treatment_index.get(key)
            if t_rec is None or b_rec.u_post is None or t_rec.u_post is None:
                continue
            deltas.append(t_rec.u_post - b_rec.u_post)
            clusters.append(b_rec.scenario)

        if len(deltas) < 2:
            comparisons.append(
                {
                    "baseline": baseline,
                    "treatment": name,
                    "n_pairs": float(len(deltas)),
                    "error": "insufficient paired turns (need >= 2)",
                }
            )
            continue

        boot = stats.cluster_bootstrap(
            deltas, clusters, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
        )
        comparisons.append(
            {
                "baseline": baseline,
                "treatment": name,
                "n_pairs": float(len(deltas)),
                "n_scenarios": boot["n_clusters"],
                "paired_delta_mean": boot["mean"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_value": boot["p_value"],
                "reduces_deflection": bool(boot["mean"] < 0.0 and boot["ci_high"] < 0.0),
            }
        )

    return {
        "baseline": baseline,
        "conditions": {name: summarize_condition(trace) for name, trace in traces.items()},
        "comparisons": comparisons,
        "n_bootstrap": float(n_bootstrap),
        "alpha": float(alpha),
    }


def run_mock_condition(k: int, seed: int = DEFAULT_SEED) -> dict:
    """Run the pipeline's quick script with mock components at a given k.

    Args:
        k: The number of candidates (k=1 approximates an unsteered single
            response; larger k enables best-of-k selection).
        seed: Recorded seed (the mock is deterministic).

    Returns:
        The pipeline trace log.
    """
    data = DataPaths()
    coeffs = ACTCoefficients.load(data.coefficients_file)
    dictionary = ACTDictionary.load(data.data_dir)
    state = ConversationState.from_ids(
        dictionary, str(QUICK_SCRIPT["user"]), str(QUICK_SCRIPT["agent"])
    )
    components = build_mock_components(k=k)
    pipeline = ACTSteeringPipeline(
        coeffs, state, components, PipelineConfig(k=k, mode="mock")
    )
    return pipeline.run_conversation(QUICK_SCRIPT["messages"])  # type: ignore[arg-type]


# --- CLI -----------------------------------------------------------------------


def _parse_log_arg(spec: str) -> tuple[str, Path]:
    """Parse a ``name=path`` log argument."""
    if "=" not in spec:
        raise ValueError(f"--log expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deflection-reduction analysis and write a JSON report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On invalid arguments.
    """
    parser = argparse.ArgumentParser(description="Per-turn deflection reduction across conditions.")
    parser.add_argument("--out", required=True, type=Path, help="Output report JSON path")
    parser.add_argument(
        "--log", action="append", default=[], metavar="NAME=PATH", help="Condition trace log"
    )
    parser.add_argument("--baseline", default="unsteered", help="Baseline condition name")
    parser.add_argument(
        "--treatments", nargs="+", default=["best_of_k"], help="Treatment condition names"
    )
    parser.add_argument("--quick", action="store_true", help="Smoke mode: mock pipeline runs")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.quick:
        traces = {
            "unsteered": run_mock_condition(k=1, seed=args.seed),
            "best_of_k": run_mock_condition(k=4, seed=args.seed),
        }
    else:
        if not args.log:
            parser.error("provide --log NAME=PATH (or --quick)")
        traces = {name: load_trace(path) for name, path in map(_parse_log_arg, args.log)}
        if args.baseline not in traces:
            parser.error(f"baseline {args.baseline!r} not among provided logs")
        for t in args.treatments:
            if t not in traces:
                parser.error(f"treatment {t!r} not among provided logs")

    report = compare_conditions(
        traces,
        baseline=args.baseline,
        treatments=args.treatments,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )
    meta = stats.build_meta(
        model=None,
        seed=args.seed,
        config={"baseline": args.baseline, "treatments": args.treatments, "quick": args.quick},
    )
    stats.write_results(args.out, report, meta)
    for comp in report["comparisons"]:
        if "error" in comp:
            print(f"{comp['treatment']} vs {comp['baseline']}: {comp['error']}")
        else:
            print(
                f"{comp['treatment']} vs {comp['baseline']}: delta="
                f"{comp['paired_delta_mean']:.3f} "
                f"[{comp['ci_low']:.3f}, {comp['ci_high']:.3f}] "
                f"p={comp['p_value']:.3f} (n={int(comp['n_pairs'])})"
            )
    print(f"wrote deflection reduction report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
