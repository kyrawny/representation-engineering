"""Seeded conversation trial simulation and deflection contrast sets.

Builds distribution-matched low/high-deflection behaviour contrast sets used
to fit the deflection reader and the contrast steerer (S6). A trial is a
(user identity, agent identity, user message behaviour) triple sampled from
the MTurk-Interact dictionaries. Event 1 (the user's message) is evaluated
with ACT to obtain the post-event fundamentals; a batch of candidate
response behaviours is then scored by the event-2 deflection, and the
bottom/top quantiles form the low/high groups. Pairs share the trial
context (context invariance); only the response behaviour differs.

CLI (smoke mode with ``--quick``)::

    python -m act_steering.stimuli --n-pairs 1300 --seed 42
    python -m act_steering.stimuli --quick
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .act_core import ACTCoefficients
from .config import DataPaths
from .dictionary import ACTDictionary, EPATerm

__all__ = [
    "Trial",
    "ContrastItem",
    "ContrastPair",
    "RESPONSE_TEMPLATE",
    "simulate_trials",
    "build_contrast_set",
    "pairs_to_dicts",
    "dicts_to_pairs",
    "main",
]

_LOGGER = logging.getLogger(__name__)

#: Default candidate pool size per trial.
DEFAULT_CANDIDATES = 200

#: Fraction of the candidate pool forming the low-deflection group.
DEFAULT_LOW_QUANTILE = 0.2

#: Fraction of the candidate pool forming the high-deflection group.
DEFAULT_HIGH_QUANTILE = 0.2

#: Minimum U separation required for a pair to be kept.
DEFAULT_MIN_DELTA = 0.05

#: Trial context template (invariant within a pair; user/agent/message only).
CONTEXT_TEMPLATE = "You are {agent}. A person who is {user} says to you: {message}."

#: Response continuation template (group-varying part; appended in S6).
RESPONSE_TEMPLATE = "You respond by {behaviour}."

_MAX_INT = 2**31 - 1


@dataclass(frozen=True)
class Trial:
    """One simulated conversation trial after event 1.

    Attributes:
        user_term: Normalized user identity term.
        agent_term: Normalized agent identity term.
        message_term: Normalized user message behaviour term.
        message_epa: EPA of the message behaviour (e, p, a).
        user_after: User's post-event self-fundamental (e, p, a).
        agent_after: Agent's post-event self-fundamental (e, p, a).
        event1_deflection: Deflection U of event 1.
    """

    user_term: str
    agent_term: str
    message_term: str
    message_epa: tuple[float, float, float]
    user_after: tuple[float, float, float]
    agent_after: tuple[float, float, float]
    event1_deflection: float


@dataclass(frozen=True)
class ContrastItem:
    """One side of a contrast pair: a response behaviour in a trial context.

    Attributes:
        context: Formatted context text (identical within a pair).
        user_term: User identity term.
        agent_term: Agent identity term.
        message_term: User message behaviour term.
        behaviour_term: Response behaviour term (the group-varying part).
        behaviour_epa: EPA of the response behaviour (e, p, a).
        deflection: Event-2 deflection U for this behaviour.
    """

    context: str
    user_term: str
    agent_term: str
    message_term: str
    behaviour_term: str
    behaviour_epa: tuple[float, float, float]
    deflection: float


@dataclass(frozen=True)
class ContrastPair:
    """A distribution-matched low/high deflection pair (same context).

    Attributes:
        low: The low-deflection item.
        high: The high-deflection item.
        delta_u: Deflection separation ``U_high - U_low``.
    """

    low: ContrastItem
    high: ContrastItem
    delta_u: float


def _sample_distinct_indices(rng: np.random.Generator, size: int) -> tuple[int, int]:
    """Draw two distinct indices in ``[0, size)`` from the generator."""
    i = int(rng.integers(size))
    j = int(rng.integers(size))
    while j == i:
        j = int(rng.integers(size))
    return i, j


def _format_context(user_term: str, agent_term: str, message_term: str) -> str:
    """Format the trial context text (invariant within a pair)."""
    return CONTEXT_TEMPLATE.format(user=user_term, agent=agent_term, message=message_term)


def _simulate_one_trial(
    coeffs: ACTCoefficients, dictionary: ACTDictionary, rng: np.random.Generator
) -> Trial:
    """Sample and evaluate a single trial (identities from file order)."""
    i, j = _sample_distinct_indices(rng, len(dictionary.identities))
    k = int(rng.integers(len(dictionary.behaviors)))
    user = dictionary.identities[i]
    agent = dictionary.identities[j]
    message = dictionary.behaviors[k]
    u_after, a_after, u1 = _event1(coeffs, user, agent, message)
    return Trial(
        user_term=user.term,
        agent_term=agent.term,
        message_term=message.term,
        message_epa=message.as_vector(),
        user_after=u_after,
        agent_after=a_after,
        event1_deflection=u1,
    )


def _event1(
    coeffs: ACTCoefficients, user: EPATerm, agent: EPATerm, message: EPATerm
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Evaluate event 1: the user acts on the agent with the message behaviour."""
    s = np.array(user.as_vector(), dtype=np.float64)
    b = np.array(message.as_vector(), dtype=np.float64)
    o = np.array(agent.as_vector(), dtype=np.float64)
    w = coeffs.impressions(s, b, o)
    u1 = float(np.sum((w - np.concatenate([s, b, o])) ** 2))
    return tuple(map(float, w[0:3])), tuple(map(float, w[6:9])), u1


def simulate_trials(
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    n_trials: int,
    seed: int,
) -> list[Trial]:
    """Simulate ``n_trials`` conversation trials with a seeded RNG.

    Args:
        coeffs: The coefficient set.
        dictionary: The MTurk-Interact dictionaries.
        n_trials: Number of trials to simulate.
        seed: RNG seed (reproducible for a fixed file order).

    Returns:
        The trials in simulation order.
    """
    rng = np.random.default_rng(seed)
    return [_simulate_one_trial(coeffs, dictionary, rng) for _ in range(n_trials)]


def _score_candidates(
    coeffs: ACTCoefficients, trial: Trial, candidates: list[EPATerm]
) -> list[tuple[EPATerm, float]]:
    """Score candidate response behaviours by event-2 deflection."""
    s = np.array(trial.agent_after, dtype=np.float64)
    o = np.array(trial.user_after, dtype=np.float64)
    batch = np.array([c.as_vector() for c in candidates], dtype=np.float64)
    values = coeffs.deflection_batch(s, o, batch)
    return list(zip(candidates, values.tolist()))


def _make_item(trial: Trial, term: EPATerm, u: float) -> ContrastItem:
    """Build a ContrastItem for a (trial, behaviour) combination."""
    return ContrastItem(
        context=_format_context(trial.user_term, trial.agent_term, trial.message_term),
        user_term=trial.user_term,
        agent_term=trial.agent_term,
        message_term=trial.message_term,
        behaviour_term=term.term,
        behaviour_epa=term.as_vector(),
        deflection=float(u),
    )


def build_contrast_set(
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    n_pairs: int,
    seed: int,
    candidates_per_trial: int = DEFAULT_CANDIDATES,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
    min_delta: float = DEFAULT_MIN_DELTA,
    max_trials: int | None = None,
) -> list[ContrastPair]:
    """Build ``n_pairs`` distribution-matched low/high contrast pairs.

    Within each trial, candidates are scored by event-2 deflection and split
    by quantiles; the sorted low and high groups are paired rank-for-rank so
    that both items of a pair share the trial context.

    Args:
        coeffs: The coefficient set.
        dictionary: The MTurk-Interact dictionaries.
        n_pairs: Number of pairs to build.
        seed: Master RNG seed.
        candidates_per_trial: Candidate response behaviours scored per trial.
        low_quantile: Bottom quantile fraction for the low group.
        high_quantile: Top quantile fraction for the high group.
        min_delta: Minimum ``U_high - U_low`` for a pair to be kept.
        max_trials: Trial budget guard (default ``4 * n_pairs``).

    Returns:
        The contrast pairs.

    Raises:
        ValueError: If a quantile is not in (0, 0.5).
    """
    if not 0.0 < low_quantile <= 0.5 or not 0.0 < high_quantile <= 0.5:
        raise ValueError("quantiles must be in (0, 0.5]")
    rng = np.random.default_rng(seed)
    budget = max_trials if max_trials is not None else n_pairs * 4
    pairs: list[ContrastPair] = []
    for _ in range(budget):
        if len(pairs) >= n_pairs:
            break
        trial = _simulate_one_trial(coeffs, dictionary, rng)
        cand_seed = int(rng.integers(_MAX_INT))
        candidates = dictionary.sample(candidates_per_trial, seed=cand_seed, collection="behaviors")
        scored = _score_candidates(coeffs, trial, candidates)
        values = np.array([u for _, u in scored], dtype=np.float64)
        low_cut = float(np.quantile(values, low_quantile))
        high_cut = float(np.quantile(values, 1.0 - high_quantile))
        low = sorted(((t, u) for t, u in scored if u <= low_cut), key=lambda x: x[1])
        high = sorted(((t, u) for t, u in scored if u >= high_cut), key=lambda x: x[1])
        # Rank-match: i-th lowest with i-th highest (both sorted ascending).
        for (lo_term, lo_u), (hi_term, hi_u) in zip(low, reversed(high)):
            if lo_term.term == hi_term.term:
                continue
            delta = hi_u - lo_u
            if delta < min_delta:
                continue
            pairs.append(
                ContrastPair(
                    low=_make_item(trial, lo_term, lo_u),
                    high=_make_item(trial, hi_term, hi_u),
                    delta_u=float(delta),
                )
            )
            if len(pairs) >= n_pairs:
                break
    if len(pairs) < n_pairs:
        _LOGGER.warning(
            "built only %d of %d requested pairs (trial budget %d exhausted)",
            len(pairs),
            n_pairs,
            budget,
        )
    return pairs


def pairs_to_dicts(pairs: list[ContrastPair]) -> list[dict]:
    """Serialize contrast pairs to a JSON-friendly list of dicts."""
    return [asdict(pair) for pair in pairs]


def dicts_to_pairs(dicts: list[dict]) -> list[ContrastPair]:
    """Rebuild contrast pairs from the dict format written by the CLI.

    Inverse of :func:`pairs_to_dicts` (JSON round-trip helper for S6).
    """

    def item(d: dict) -> ContrastItem:
        return ContrastItem(
            context=d["context"],
            user_term=d["user_term"],
            agent_term=d["agent_term"],
            message_term=d["message_term"],
            behaviour_term=d["behaviour_term"],
            behaviour_epa=tuple(d["behaviour_epa"]),
            deflection=float(d["deflection"]),
        )

    return [
        ContrastPair(low=item(p["low"]), high=item(p["high"]), delta_u=float(p["delta_u"]))
        for p in dicts
    ]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build the contrast set and write it to JSON."""
    parser = argparse.ArgumentParser(description="Build ACT deflection contrast sets.")
    parser.add_argument("--n-pairs", type=int, default=1300, help="number of contrast pairs")
    parser.add_argument("--seed", type=int, default=42, help="master RNG seed")
    parser.add_argument(
        "--candidates", type=int, default=DEFAULT_CANDIDATES, help="candidates per trial"
    )
    parser.add_argument(
        "--min-delta", type=float, default=DEFAULT_MIN_DELTA, help="min U separation"
    )
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    parser.add_argument("--quick", action="store_true", help="smoke mode (20 pairs, 50 candidates)")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_pairs = 20
        args.candidates = 50
    paths = DataPaths()
    paths.ensure_dirs()
    out = args.out if args.out is not None else paths.artifacts_dir / "contrast_set.json"
    coeffs = ACTCoefficients.load(paths.coefficients_file)
    dictionary = ACTDictionary.load(paths.data_dir)
    pairs = build_contrast_set(
        coeffs, dictionary, args.n_pairs, args.seed, args.candidates, min_delta=args.min_delta
    )
    payload = {
        "seed": args.seed,
        "n_pairs_requested": args.n_pairs,
        "n_pairs": len(pairs),
        "candidates_per_trial": args.candidates,
        "min_delta": args.min_delta,
        "pairs": pairs_to_dicts(pairs),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
