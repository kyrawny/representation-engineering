"""Condition-matrix ablations over the ACT steering pipeline (S11).

Question answered: **which part of the design does the work?**

The ablation runner executes a labeled condition matrix through the real
pipeline control loop (read-compute-steer-select-monitor-repair) and
summarizes each condition's per-turn deflection and control-loop activity:

| label                 | k | monitor | steering stream                |
|-----------------------|---|---------|--------------------------------|
| ``unsteered``         | 1 | off     | none                           |
| ``monitor_only``      | 1 | on      | none                           |
| ``steer``             | 1 | off     | ACT (situation-dependent)      |
| ``random_dir``        | 1 | off     | random direction, matched norm |
| ``constant_vec``      | 1 | off     | constant vector                |
| ``identity_only``     | 1 | off     | identity target, no message read |
| ``situation_dependent``| 1 | off     | full ACT two-event target      |
| ``best_of_k``         | 4 | off     | none                           |
| ``best_of_k_steer``   | 4 | off     | ACT (situation-dependent)      |

In the CPU ``--quick`` mode the model-dependent components are deterministic
mocks, so each condition gets a distinct (labeled) candidate stream; the
control-loop machinery (state, selector, monitor, repair) is the real thing.

Battery mode (``--battery``) runs each condition over the scenario battery
(one user message per scenario, fresh conversation state per scenario) and
reports per-condition summaries with **cluster bootstrap CIs** (clusters =
scenarios, the correct unit for the H3 contrasts) plus paired contrasts
against the unsteered baseline. In mock mode the components are the same
deterministic mocks as ``--quick`` (wiring smoke); with ``--model NAME``
the real model-bound components are used and each ablation label maps to a
:func:`act_steering.pipeline.SteerCondition` (the H3 ablation matrix).

CLI
---
::

    python -m evaluation.ablations --quick --out results/ablations.json
    python -m evaluation.ablations --quick --conditions unsteered best_of_k --out ...
    python -m evaluation.ablations --battery artifacts/scenario_battery.json \
        --out results/ablations_battery.json
    python -m evaluation.ablations --battery artifacts/scenario_battery.json \
        --model meta-llama/Llama-3.1-8B-Instruct --out results/ablations_battery_llama.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_SEED, DataPaths, PipelineConfig
from act_steering.dictionary import ACTDictionary
from act_steering.monitor import build_token_trace
from act_steering.pipeline import (
    QUICK_SCRIPT,
    ACTCoefficients,
    ACTSteeringPipeline,
    PipelineComponents,
    SteerCondition,
    build_real_components,
)
from act_steering.state import ConversationState
from evaluation import stats

__all__ = [
    "CONDITIONS",
    "DEFAULT_CONSTANT_TARGET",
    "STEER_AXIS_BY_LABEL",
    "ConditionSpec",
    "battery_condition_summary",
    "build_condition_components",
    "contrast_vs_baseline",
    "load_battery",
    "main",
    "run_battery_condition",
    "run_condition",
    "steer_condition_for",
    "summarize_trace",
]


@dataclass(frozen=True)
class ConditionSpec:
    """One labeled condition in the ablation matrix.

    Attributes:
        name: The condition label.
        k: Number of candidates per turn.
        monitor: Whether the deflection monitor runs.
        steer_salt: Salt for the (mock) candidate stream; None = no steering.
        description: Human-readable description of the condition.
    """

    name: str
    k: int = 1
    monitor: bool = False
    steer_salt: str | None = None
    description: str = ""


#: The full ablation matrix (order = report order).
CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("unsteered", 1, False, None, "single unsteered response"),
    ConditionSpec("monitor_only", 1, True, None, "single response + deflection monitor"),
    ConditionSpec("steer", 1, False, "act", "ACT steering, k=1"),
    ConditionSpec("random_dir", 1, False, "random", "random direction, matched norm"),
    ConditionSpec("constant_vec", 1, False, "constant", "constant steering vector"),
    ConditionSpec("identity_only", 1, False, "identity_only", "identity target, no message read"),
    ConditionSpec("shuffled", 1, False, "shuffled", "shuffled-identity b* (B7)"),
    ConditionSpec(
        "situation_dependent", 1, False, "situation", "full ACT situation-dependent target"
    ),
    ConditionSpec("best_of_k", 4, False, None, "best-of-4 unsteered"),
    ConditionSpec("best_of_k_steer", 4, False, "act", "best-of-4 with ACT steering"),
)


#: Default fixed EPA for the ``constant_vec`` condition (B8): mildly
#: positive Evaluation, neutral Potency/Activity ("warm-professional").
DEFAULT_CONSTANT_TARGET: tuple[float, float, float] = (0.5, 0.0, 0.0)

#: Ablation label -> (steer axis, b* target kind) for real (model) runs.
STEER_AXIS_BY_LABEL: dict[str, tuple[str, str | None]] = {
    "unsteered": ("none", None),
    "monitor_only": ("none", None),
    "steer": ("deflection", None),
    "random_dir": ("random", None),
    "constant_vec": ("epa_target", "constant"),
    "identity_only": ("epa_target", "identity_only"),
    "shuffled": ("epa_target", "shuffled"),
    "situation_dependent": ("epa_target", "situation"),
    "best_of_k": ("none", None),
    "best_of_k_steer": ("deflection", None),
}


def steer_condition_for(
    spec: ConditionSpec,
    scenario_index: int = 0,
    constant_target: tuple[float, float, float] = DEFAULT_CONSTANT_TARGET,
    seed: int = DEFAULT_SEED,
) -> SteerCondition:
    """Map one ablation label to its real-run steering condition.

    Args:
        spec: The condition spec (the label drives the mapping).
        scenario_index: The scenario's index in the battery (determines
            the seeded random direction / shuffled identity pair).
        constant_target: Fixed EPA for the ``constant_vec`` label.
        seed: Master seed for seeded draws.

    Returns:
        The :class:`~act_steering.pipeline.SteerCondition`.

    Raises:
        KeyError: If the label has no mapping.
    """
    axis, kind = STEER_AXIS_BY_LABEL[spec.name]
    return SteerCondition(
        name=spec.name,
        axis=axis,  # type: ignore[arg-type]
        target_kind=kind,  # type: ignore[arg-type]
        constant_target=constant_target if kind == "constant" else None,
        seed=seed,
        scenario_index=scenario_index,
    )


def _mock_epa(text: str, salt: str) -> tuple[float, float, float]:
    """A deterministic mock EPA in [-4.3, 4.3]^3 (CPU mode)."""
    digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
    return tuple((digest[i] / 255.0 - 0.5) * 8.6 for i in range(3))  # type: ignore[return-value]


def build_condition_components(spec: ConditionSpec) -> PipelineComponents:
    """Build mock model-dependent components for one condition.

    Args:
        spec: The condition spec.

    Returns:
        Components whose candidate stream is labeled by the condition name
        (deterministic per name/k).
    """

    def candidate_generator(_state: ConversationState) -> list[str]:
        return [f"{spec.name}:candidate-{j}" for j in range(spec.k)]

    monitor = None
    if spec.monitor:

        def monitor(text: str) -> Any:
            words = text.split()
            scores = np.array(
                [
                    (hashlib.sha256(f"mon:{text}:{i}".encode()).digest()[0] / 255.0 - 0.5) * 4.0
                    for i in range(len(words))
                ]
            )
            return build_token_trace(scores, threshold=1.0)

    return PipelineComponents(
        message_reader=lambda text: _mock_epa(text, "msg"),
        candidate_generator=candidate_generator,
        behaviour_reader=lambda text: _mock_epa(text, "beh"),
        monitor=monitor,
        repair_policy=None,
    )


def run_condition(
    spec: ConditionSpec,
    script: dict[str, object],
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run one condition's scripted conversation through the pipeline.

    Args:
        spec: The condition spec.
        script: A script dict with ``user``, ``agent`` and ``messages``.
        seed: Recorded seed (the mock components are deterministic).

    Returns:
        The pipeline trace log.
    """
    data = DataPaths()
    coeffs = ACTCoefficients.load(data.coefficients_file)
    dictionary = ACTDictionary.load(data.data_dir)
    state = ConversationState.from_ids(dictionary, str(script["user"]), str(script["agent"]))
    components = build_condition_components(spec)
    pipeline = ACTSteeringPipeline(
        coeffs,
        state,
        components,
        PipelineConfig(k=spec.k, mode="mock", monitor_positions=4 if spec.monitor else 0),
    )
    return pipeline.run_conversation(list(script["messages"]))  # type: ignore[arg-type]


def summarize_trace(trace: dict, spec: ConditionSpec) -> dict[str, Any]:
    """Summarize one condition's trace log.

    Args:
        trace: The condition's pipeline trace log.
        spec: The condition spec (for ``k`` and the description).

    Returns:
        A dict with ``k``, ``description``, ``n_turns``, ``u_pre_mean``,
        ``u_post_mean``, ``u_post_std``, ``n_repairs_adopted``,
        ``n_fallbacks``, ``n_monitor_fires``.
    """
    turns = trace.get("turns", [])
    u_posts = [t["u_post"] for t in turns if t.get("u_post") is not None]
    u_pres = [t["u_pre"] for t in turns if t.get("u_pre") is not None]
    return {
        "k": float(spec.k),
        "description": spec.description,
        "n_turns": float(len(turns)),
        "u_pre_mean": float(np.mean(u_pres)) if u_pres else 0.0,
        "u_post_mean": float(np.mean(u_posts)) if u_posts else 0.0,
        "u_post_std": float(np.std(u_posts, ddof=1)) if len(u_posts) > 1 else 0.0,
        "n_repairs_adopted": float(
            sum(1 for t in turns if t.get("repair") and t["repair"].get("adopted"))
        ),
        "n_fallbacks": float(sum(1 for t in turns if t.get("fallback"))),
        "n_monitor_fires": float(
            sum(1 for t in turns if t.get("monitor") and t["monitor"].get("fired"))
        ),
    }


# --- battery mode (H3 ablation matrix over the scenario battery) -----------------


def load_battery(path: Path) -> dict:
    """Load a scenario-battery artifact (see :mod:`act_steering.battery`).

    Args:
        path: Path to ``scenario_battery.json``.

    Returns:
        The artifact payload (``seed``, ``n_scenarios``, ``scenarios``).

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the artifact carries no scenarios.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("scenarios"):
        raise ValueError(f"battery artifact {path} has no scenarios")
    return data


def _run_one_scenario(
    scenario: dict,
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    components: PipelineComponents,
    config: PipelineConfig,
) -> dict:
    """Run one battery scenario (a single user turn) through the pipeline.

    Args:
        scenario: One battery scenario dict (``user_term``, ``agent_term``,
            ``message_term``).
        coeffs: The coefficient set.
        dictionary: The ACT dictionary.
        components: The condition's model-dependent components.
        config: The control-loop settings.

    Returns:
        A dict with the scenario's ``u_post`` and control-loop counters.
    """
    state = ConversationState.from_ids(dictionary, scenario["user_term"], scenario["agent_term"])
    pipeline = ACTSteeringPipeline(coeffs, state, components, config)
    log = pipeline.run_conversation([scenario["message_term"]])
    turn = log["turns"][-1]
    return {
        "scenario_index": int(scenario["index"]),
        "u_post": turn.get("u_post"),
        "repair_adopted": bool(turn.get("repair") and turn["repair"].get("adopted")),
        "fallback": bool(turn.get("fallback")),
        "monitor_fired": bool(turn.get("monitor") and turn["monitor"].get("fired")),
    }


def run_battery_condition(
    spec: ConditionSpec,
    scenarios: list[dict],
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    *,
    components: PipelineComponents,
    mode: str,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run one condition over the scenario battery and summarise it.

    Each scenario gets a fresh conversation state; the per-scenario metric
    is the post-response deflection ``u_post``. The summary reports the
    mean with a cluster-bootstrap CI (clusters = scenarios) and the
    control-loop counters.

    Args:
        spec: The condition spec.
        scenarios: The battery scenarios to run.
        coeffs: The coefficient set.
        dictionary: The ACT dictionary.
        components: The condition's model-dependent components (mock or
            real; shared across scenarios).
        mode: ``"mock"`` or ``"static"`` (the pipeline mode).
        seed: RNG seed for the bootstrap.

    Returns:
        A dict with ``k``, ``description``, ``n_scenarios``, ``u_post``
        (mean / ci_low / ci_high / n / n_clusters / n_bootstrap / p_value),
        and the loop counters.
    """
    config = PipelineConfig(
        k=spec.k,
        mode=mode,
        monitor_positions=4 if spec.monitor else 0,
    )
    results = [_run_one_scenario(s, coeffs, dictionary, components, config) for s in scenarios]
    return battery_condition_summary(spec, results, seed=seed)


def battery_condition_summary(
    spec: ConditionSpec, results: list[dict], *, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Summarise per-scenario results with a cluster bootstrap CI.

    Args:
        spec: The condition spec.
        results: Per-scenario result dicts (see :func:`_run_one_scenario`).
        seed: RNG seed for the bootstrap.

    Returns:
        The summary dict (see :func:`run_battery_condition`).

    Raises:
        ValueError: If fewer than two scenarios yield a usable ``u_post``.
    """
    scored = [r for r in results if r.get("u_post") is not None]
    if len(scored) < 2:
        raise ValueError(f"condition {spec.name!r}: need >= 2 scored scenarios, got {len(scored)}")
    boot = stats.cluster_bootstrap(
        [r["u_post"] for r in scored],
        [str(r["scenario_index"]) for r in scored],
        seed=seed,
    )
    return {
        "k": float(spec.k),
        "description": spec.description,
        "n_scenarios": float(len(results)),
        "n_scored": float(len(scored)),
        "u_post": boot,
        "n_repairs_adopted": float(sum(1 for r in results if r["repair_adopted"])),
        "n_fallbacks": float(sum(1 for r in results if r["fallback"])),
        "n_monitor_fires": float(sum(1 for r in results if r["monitor_fired"])),
    }


def contrast_vs_baseline(
    condition_results: list[dict],
    baseline_results: list[dict],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, float] | None:
    """Paired contrast of one condition's per-scenario ``u_post`` vs baseline.

    The paired differences (condition - baseline) are resampled by
    scenario (cluster bootstrap), so the CI is valid for the H3 question
    "does the condition change the outcome, scenario by scenario?".

    Args:
        condition_results: Per-scenario results for the condition.
        baseline_results: Per-scenario results for the baseline
            (``unsteered``).
        seed: RNG seed for the bootstrap.

    Returns:
        A dict with ``mean`` (negative = lower deflection than baseline),
        ``ci_low``, ``ci_high``, ``p_value``, ``n``, ``n_clusters`` — or
        ``None`` when fewer than two scenarios overlap.
    """
    by_index = {r["scenario_index"]: r for r in baseline_results}
    diffs: list[float] = []
    clusters: list[str] = []
    for r in condition_results:
        base = by_index.get(r["scenario_index"])
        if base is None or r.get("u_post") is None or base.get("u_post") is None:
            continue
        diffs.append(r["u_post"] - base["u_post"])
        clusters.append(str(r["scenario_index"]))
    if len(diffs) < 2:
        return None
    return stats.cluster_bootstrap(diffs, clusters, seed=seed)


# --- CLI -----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation matrix and write a JSON report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: On invalid arguments.
    """
    parser = argparse.ArgumentParser(description="Condition-matrix ablations over the pipeline.")
    parser.add_argument("--out", required=True, type=Path, help="Output report JSON path")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Condition labels to run (default: the full matrix)",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke mode: built-in quick script")
    parser.add_argument(
        "--script", type=Path, default=None, help="Script JSON with user/agent/messages"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--battery",
        type=Path,
        default=None,
        help="Scenario-battery artifact (enables battery mode; see act_steering.battery)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Cap on battery scenarios per condition (0 = all)",
    )
    parser.add_argument("--model", default=None, help="HF model id (battery real mode)")
    parser.add_argument("--device", default="cuda", help="torch device (real mode)")
    parser.add_argument("--dtype", default="float16", help="torch dtype (real mode)")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Steering base strength (real mode)"
    )
    parser.add_argument(
        "--token-pos",
        default="end",
        choices=("end", "start", "none"),
        help="Where the steering vector applies (real mode)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation cap (real mode)")
    parser.add_argument(
        "--constant-target",
        default=f"{DEFAULT_CONSTANT_TARGET[0]},{DEFAULT_CONSTANT_TARGET[1]},{DEFAULT_CONSTANT_TARGET[2]}",
        help="Comma-separated E,P,A for the constant_vec condition",
    )
    args = parser.parse_args(argv)

    if args.conditions:
        known = {spec.name for spec in CONDITIONS}
        unknown = [c for c in args.conditions if c not in known]
        if unknown:
            parser.error(f"unknown conditions: {unknown}; known: {sorted(known)}")
        selected = [spec for spec in CONDITIONS if spec.name in set(args.conditions)]
    else:
        selected = list(CONDITIONS)

    # Battery mode (H3 ablation matrix over the scenario battery).
    if args.battery is not None:
        return main_battery(args, args.battery, selected)

    if args.script is not None:
        script = json.loads(args.script.read_text(encoding="utf-8"))
        for key in ("user", "agent", "messages"):
            if key not in script:
                parser.error(f"script missing key {key!r}")
    elif args.quick:
        script = QUICK_SCRIPT
    else:
        parser.error("provide --script PATH (or --quick)")

    conditions: dict[str, Any] = {}
    for spec in selected:
        trace = run_condition(spec, script, seed=args.seed)
        conditions[spec.name] = summarize_trace(trace, spec)

    report = {
        "conditions": conditions,
        "order": [spec.name for spec in selected],
        "script": {
            "user": str(script["user"]),
            "agent": str(script["agent"]),
            "n_messages": len(script["messages"]),
        },
    }
    meta = stats.build_meta(
        model=None,
        seed=args.seed,
        config={"conditions": [spec.name for spec in selected], "quick": args.quick},
    )
    stats.write_results(args.out, report, meta)
    for name in report["order"]:
        s = conditions[name]
        print(
            f"{name}: u_post={s['u_post_mean']:.3f} (n={int(s['n_turns'])}, k={int(s['k'])}, "
            f"fires={int(s['n_monitor_fires'])}, repairs={int(s['n_repairs_adopted'])})"
        )
    print(f"wrote ablations report to {args.out}")
    return 0


def main_battery(
    args: argparse.Namespace, battery_path: Path, selected: list[ConditionSpec]
) -> int:
    """Run the selected conditions over the scenario battery (one turn each).

    Mock mode uses the deterministic mock components (wiring smoke); with
    ``--model`` the real model-bound components are used and each label
    maps to a :class:`~act_steering.pipeline.SteerCondition`. Reports
    per-condition cluster-bootstrap CIs and paired contrasts vs unsteered.

    Args:
        args: Parsed CLI arguments (see :func:`main`).
        battery_path: The battery artifact path.
        selected: The condition specs to run (in order).

    Returns:
        0 on success.

    Raises:
        SystemExit: On invalid arguments.
    """
    data = DataPaths()
    coeffs = ACTCoefficients.load(data.coefficients_file)
    dictionary = ACTDictionary.load(data.data_dir)
    battery = load_battery(battery_path)
    scenarios = list(battery["scenarios"])
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    constant_target = tuple(float(x) for x in args.constant_target.split(","))
    if len(constant_target) != 3:
        raise SystemExit("--constant-target must be 'E,P,A'")

    model_mode = args.model is not None
    real_bundle: dict[str, Any] = {}
    if model_mode:
        from act_steering.config import SteeringConfig
        from act_steering.directions import (
            DirectionsBundle,
            load_model_and_pipeline,
            resolve_directions_paths,
        )

        npz_path, json_path = resolve_directions_paths(None, args.model)
        if not npz_path.exists():
            raise SystemExit(
                f"directions artifact {npz_path} not found; run "
                f"`python -m act_steering.directions --model {args.model}` first"
            )
        model, tokenizer, _ = load_model_and_pipeline(args.model, args.device, args.dtype)
        real_bundle = {
            "model": model,
            "tokenizer": tokenizer,
            "bundle": DirectionsBundle.load(npz_path, json_path),
            "steering": SteeringConfig(
                alpha=args.alpha,
                token_pos=None if args.token_pos == "none" else args.token_pos,
            ),
        }

    conditions: dict[str, Any] = {}
    per_condition_results: dict[str, list[dict]] = {}
    for spec in selected:
        if model_mode:
            # One steerer (one model wrap) per condition; the scenario
            # index is read from the holder at draw time so per-scenario
            # "random" / "shuffled" draws vary across scenarios.
            holder: dict[str, int] = {"index": 0}
            components = build_real_components(
                real_bundle["model"],
                real_bundle["tokenizer"],
                real_bundle["bundle"],
                PipelineConfig(k=spec.k, mode="static"),
                steering=real_bundle["steering"],
                coeffs=coeffs,
                dictionary=dictionary,
                max_new_tokens=args.max_new_tokens,
                condition=steer_condition_for(spec, 0, constant_target, args.seed),
                scenario_index_provider=lambda h=holder: h["index"],
            )
        else:
            components = build_condition_components(spec)
        cfg = PipelineConfig(
            k=spec.k,
            mode="static" if model_mode else "mock",
            monitor_positions=4 if spec.monitor else 0,
        )
        results: list[dict] = []
        for s in scenarios:
            if model_mode:
                holder["index"] = int(s["index"])
            results.append(_run_one_scenario(s, coeffs, dictionary, components, cfg))
        per_condition_results[spec.name] = results
        conditions[spec.name] = battery_condition_summary(
            spec, per_condition_results[spec.name], seed=args.seed
        )

    baseline = per_condition_results.get("unsteered")
    contrasts: dict[str, Any] = {}
    if baseline:
        for name in conditions:
            if name == "unsteered":
                continue
            c = contrast_vs_baseline(per_condition_results[name], baseline, seed=args.seed)
            if c is not None:
                contrasts[name] = c

    report = {
        "conditions": conditions,
        "order": [spec.name for spec in selected],
        "contrasts_vs_unsteered": contrasts,
        "battery": {
            "path": str(battery_path),
            "seed": battery.get("seed"),
            "n_scenarios": float(len(scenarios)),
        },
        "mode": "model" if model_mode else "mock",
    }
    meta = stats.build_meta(
        model=args.model if model_mode else None,
        seed=args.seed,
        config={
            "conditions": [spec.name for spec in selected],
            "max_scenarios": args.max_scenarios,
            "constant_target": list(constant_target),
        },
    )
    stats.write_results(args.out, report, meta)
    for name in report["order"]:
        s = conditions[name]
        up = s["u_post"]
        print(
            f"{name}: u_post={up['mean']:.3f} [{up['ci_low']:.3f}, {up['ci_high']:.3f}] "
            f"(n={int(s['n_scored'])}, k={int(s['k'])}, "
            f"fires={int(s['n_monitor_fires'])}, repairs={int(s['n_repairs_adopted'])})"
        )
    for name, c in contrasts.items():
        print(
            f"{name} vs unsteered: {c['mean']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f}"
        )
    print(f"wrote ablations report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
