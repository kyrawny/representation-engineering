"""Scenario battery generation with a leakage check (WS1.7 / T3-S3).

Builds a fresh set of conversation scenarios (user identity, agent
identity, user message behaviour) for the behavioural ablation matrix,
sampled with a seed distinct from the contrast-set seed, and asserts
**zero trial-context overlap** with the contrast set artifact: no
``(user, agent, message)`` triple in the battery may appear among the
contrast pairs, so no evaluation scenario was part of the reader's
training distribution.

Within the battery, trial contexts are also deduplicated (no two
scenarios share a context), and identity-pair reuse is reported as a
diagnostic (shared identity pairs across scenarios are legitimate — the
claim is generalisation over *situations*, not over identities).

CLI::

    python -m act_steering.battery --n-scenarios 500 --seed 20271004 \
        --contrast artifacts/contrast_set.json --out artifacts/scenario_battery.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .act_core import ACTCoefficients
from .config import DataPaths
from .dictionary import ACTDictionary
from .stimuli import CONTEXT_TEMPLATE, Trial, simulate_trials

__all__ = [
    "DEFAULT_BATTERY_SEED",
    "DEFAULT_N_SCENARIOS",
    "trial_key",
    "contrast_trial_keys",
    "generate_battery",
    "assert_zero_overlap",
    "write_battery",
    "main",
]

#: Fresh seed for the evaluation battery (deliberately distinct from the
#: contrast-set seed :data:`DEFAULT_SEED` = 42).
DEFAULT_BATTERY_SEED = 20_271_004

#: Default battery size for the ablation matrix.
DEFAULT_N_SCENARIOS = 500


def trial_key(trial: Trial) -> tuple[str, str, str]:
    """The identity triple ``(user, agent, message)`` of a trial."""
    return (trial.user_term, trial.agent_term, trial.message_term)


def contrast_trial_keys(contrast_path: Path) -> set[tuple[str, str, str]]:
    """All distinct trial contexts in a contrast-set artifact.

    Args:
        contrast_path: Path to the contrast-set JSON (see
            :func:`act_steering.stimuli.main`).

    Returns:
        The set of ``(user_term, agent_term, message_term)`` triples.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    data = json.loads(contrast_path.read_text(encoding="utf-8"))
    keys: set[tuple[str, str, str]] = set()
    for pair in data.get("pairs", []):
        for side in ("low", "high"):
            item = pair[side]
            keys.add((item["user_term"], item["agent_term"], item["message_term"]))
    return keys


def generate_battery(
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    n_scenarios: int,
    seed: int,
    contrast_keys: set[tuple[str, str, str]] | None = None,
    max_attempts_per_scenario: int = 100,
) -> list[Trial]:
    """Sample ``n_scenarios`` fresh, context-unique trials.

    Sampling continues past duplicates (within the battery and against
    ``contrast_keys`` when given) until the battery is full.

    Args:
        coeffs: The impression-formation coefficient set.
        dictionary: The MTurk-Interact dictionaries.
        n_scenarios: Battery size.
        seed: RNG seed (must differ from the contrast-set seed).
        contrast_keys: Trial contexts to exclude (the contrast set).
        max_attempts_per_scenario: Safety cap on resampling per slot.

    Returns:
        The trials in battery order (contexts unique).

    Raises:
        ValueError: If the requested size is not positive, the seed is
            non-positive, or a slot cannot be filled within the cap.
    """
    if n_scenarios <= 0:
        raise ValueError(f"n_scenarios must be positive, got {n_scenarios}")
    if seed <= 0:
        raise ValueError(f"seed must be positive, got {seed}")
    excluded = set(contrast_keys or ())
    trials: list[Trial] = []
    seen: set[tuple[str, str, str]] = set()
    batch_size = max(1, min(64, n_scenarios))
    max_total = max_attempts_per_scenario * n_scenarios
    attempt = 0
    while len(trials) < n_scenarios:
        attempt += 1
        if attempt > max_total:
            raise ValueError(
                f"could not fill the battery: {len(trials)}/{n_scenarios} "
                f"unique contexts after {attempt} attempts (dictionary too "
                f"small or overlap too high)"
            )
        for trial in simulate_trials(coeffs, dictionary, batch_size, seed + 100_000 * attempt):
            key = trial_key(trial)
            if key in seen or key in excluded:
                continue
            seen.add(key)
            trials.append(trial)
            if len(trials) == n_scenarios:
                break
    return trials


def assert_zero_overlap(battery: list[Trial], contrast_keys: set[tuple[str, str, str]]) -> None:
    """Assert that no battery trial context occurs in the contrast set.

    Args:
        battery: The generated scenarios.
        contrast_keys: The contrast-set trial contexts.

    Raises:
        AssertionError: If any battery context overlaps the contrast set,
            naming the first offending triple.
    """
    overlap = sorted({trial_key(t) for t in battery} & contrast_keys)
    if overlap:
        raise AssertionError(
            f"leakage: {len(overlap)} battery scenario(s) overlap the "
            f"contrast set, e.g. {overlap[0]!r}"
        )


def write_battery(
    out_path: Path,
    battery: list[Trial],
    seed: int,
    contrast_path: Path,
    *,
    contrast_seed: int | None = None,
) -> dict:
    """Serialise the battery artifact and return its summary.

    The artifact records the generation seed, the contrast-set seed it
    must not collide with, the overlap statistics (asserted zero), and
    the scenarios with their event-1 state (so downstream runs do not
    need to re-simulate).

    Args:
        out_path: Destination JSON path.
        battery: The generated scenarios.
        seed: The generation seed.
        contrast_path: The contrast-set artifact path (recorded for
            provenance).

    Returns:
        The summary dict (also stored under ``"summary"``).
    """
    contrast_data = json.loads(contrast_path.read_text(encoding="utf-8"))
    contrast_keys = contrast_trial_keys(contrast_path)
    assert_zero_overlap(battery, contrast_keys)
    identity_pairs = {(t.user_term, t.agent_term) for t in battery}
    summary = {
        "seed": seed,
        "contrast_seed": contrast_seed if contrast_seed is not None else contrast_data.get("seed"),
        "n_scenarios": len(battery),
        "n_distinct_trial_contexts": len({trial_key(t) for t in battery}),
        "n_distinct_identity_pairs": len(identity_pairs),
        "contrast_overlap": 0,
        "contrast_pairs": len(contrast_keys),
    }
    payload = {
        "seed": seed,
        "contrast_set": str(contrast_path),
        "n_scenarios": len(battery),
        "summary": summary,
        "scenarios": [
            {
                "index": i,
                **asdict(trial),
                "context": CONTEXT_TEMPLATE.format(
                    user=trial.user_term,
                    agent=trial.agent_term,
                    message=trial.message_term,
                ),
            }
            for i, trial in enumerate(battery)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BATTERY_SEED)
    parser.add_argument("--data", type=Path, default=None, help="ACT dictionary data directory")
    parser.add_argument(
        "--contrast",
        type=Path,
        default=Path("artifacts/contrast_set.json"),
        help="Contrast-set artifact for the leakage check",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Battery artifact output path (default: artifacts/scenario_battery.json)",
    )
    args = parser.parse_args(argv)

    paths = DataPaths()
    paths.ensure_dirs()
    data = args.data or paths.data_dir
    dictionary = ACTDictionary.load(data)
    coeffs = ACTCoefficients.load(paths.coefficients_file)
    out = args.out if args.out is not None else paths.artifacts_dir / "scenario_battery.json"
    contrast = args.contrast
    if not contrast.is_absolute() and not contrast.exists():
        candidate = paths.artifacts_dir / args.contrast.name
        if candidate.exists():
            contrast = candidate
    battery = generate_battery(coeffs, dictionary, args.n_scenarios, args.seed)
    summary = write_battery(out, battery, args.seed, contrast)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
