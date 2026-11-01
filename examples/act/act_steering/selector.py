"""ACT best-of-k candidate selection (Mode 3 of the synthesis).

Given the post-event-1 conversation state, generate (or receive) ``k``
candidate responses, read the EPA of each candidate's behaviour with an
injectable reader, score each candidate by the post-event ACT deflection
it would cause, and select the candidate with the lowest deflection::

    for r in candidates:
        b_r = reader(r)
        U_r = deflection(agent_fundamentals, b_r, user_fundamentals)
    pick argmin U_r

Pure logic — torch-free. The candidate generator (model sampling) and the
EPA reader (e.g. the M-X reader over a response) are injected callables,
so the whole module is unit-testable on CPU.

Selection rule
--------------
The deflection is evaluated with the post-event-1 fundamentals of the
conversation state: the agent (actor) acts on the user (object) with the
candidate behaviour ``b_r``. The winner is the argmin of ``U``; any
candidate within ``tie_tolerance`` of the minimum is treated as tied,
and among the tied candidates the earliest index wins (generation order
is the priority). This keeps the selection reproducible for a fixed
generator and a fixed ``k``.

Degenerate / failure handling
-----------------------------
* ``k = 1``: the single candidate wins trivially (``degenerate=True``).
* A candidate whose EPA reading fails is marked unscorable (``error`` is
  set); the selection proceeds over the remaining candidates.
* If every candidate is unscorable, the selection has
  ``winner_index=None`` and a ``fallback_reason``; the caller falls back
  (e.g. to the unsteered first candidate or the b*-steered response).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from act_steering.act_core import EPA, ACTCoefficients
from act_steering.state import ConversationState

#: Callable mapping a candidate response text to the EPA of its behaviour.
BehaviourReader = Callable[[str], EPA | tuple[float, float, float] | np.ndarray]
#: Callable mapping the conversation state to a list of candidate texts.
CandidateGenerator = Callable[[ConversationState], list[str]]

__all__ = [
    "BestOfKSelection",
    "BestOfKSelector",
    "CandidateScore",
    "select_best_of_k",
]


def _epa_array(name: str, value: EPA | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Validate and coerce an EPA value to a float (3,) array."""
    if isinstance(value, EPA):
        value = value.as_array()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} EPA must have shape (3,), got {arr.shape}")
    return arr


@dataclass(frozen=True)
class CandidateScore:
    """The scoring outcome for one candidate response.

    Attributes:
        index: Candidate position in the generated list (0-based).
        text: The candidate response text.
        behaviour_epa: The read behaviour EPA (None when unscorable).
        deflection: The post-event deflection U (None when unscorable).
        error: Why the candidate is unscorable (None when scorable).
    """

    index: int
    text: str
    behaviour_epa: EPA | None
    deflection: float | None
    error: str | None = None

    @property
    def scorable(self) -> bool:
        """True when this candidate received a finite deflection."""
        return self.error is None and self.deflection is not None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "index": self.index,
            "text": self.text,
            "behaviour_epa": (
                None
                if self.behaviour_epa is None
                else [float(v) for v in self.behaviour_epa.as_array()]
            ),
            "deflection": None if self.deflection is None else float(self.deflection),
            "error": self.error,
        }


@dataclass(frozen=True)
class BestOfKSelection:
    """The outcome of a best-of-k selection.

    Attributes:
        winner_index: Index of the winning candidate (None on total
            scoring failure).
        k: Number of candidates considered.
        tie_tolerance: The absolute deflection tie tolerance used.
        degenerate: True when k == 1 (no comparison was possible).
        all_tied: True when every scorable candidate is within
            ``tie_tolerance`` of the minimum.
        n_scorable: Number of candidates that scored successfully.
        scores: Per-candidate scores, in generation order.
        fallback_reason: Why the selection needs a fallback (None when a
            winner exists).
    """

    winner_index: int | None
    k: int
    tie_tolerance: float
    degenerate: bool
    all_tied: bool
    n_scorable: int
    scores: tuple[CandidateScore, ...]
    fallback_reason: str | None = None

    @property
    def winner(self) -> CandidateScore | None:
        """The winning candidate score (None on total failure)."""
        return None if self.winner_index is None else self.scores[self.winner_index]

    @property
    def winner_text(self) -> str | None:
        """The winning candidate text (None on total failure)."""
        return None if self.winner is None else self.winner.text

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "winner_index": self.winner_index,
            "k": self.k,
            "tie_tolerance": float(self.tie_tolerance),
            "degenerate": self.degenerate,
            "all_tied": self.all_tied,
            "n_scorable": self.n_scorable,
            "scores": [s.to_dict() for s in self.scores],
            "fallback_reason": self.fallback_reason,
        }


def _select(scores: tuple[CandidateScore, ...], k: int, tie_tolerance: float) -> BestOfKSelection:
    """Pick the argmin over the scorable scores.

    Candidates within ``tie_tolerance`` of the minimum deflection are
    treated as tied; among the tied candidates the earliest index wins.
    """
    scorable = [s for s in scores if s.scorable]
    if not scorable:
        return BestOfKSelection(
            winner_index=None,
            k=k,
            tie_tolerance=tie_tolerance,
            degenerate=(k == 1),
            all_tied=False,
            n_scorable=0,
            scores=scores,
            fallback_reason=f"all {k} candidates failed scoring",
        )
    best_u = min(s.deflection for s in scorable)
    tied = [s for s in scorable if s.deflection <= best_u + tie_tolerance]
    # Earliest index among the tied candidates (generation order priority).
    best = min(tied, key=lambda s: s.index)
    all_tied = len(tied) == len(scorable)
    return BestOfKSelection(
        winner_index=best.index,
        k=k,
        tie_tolerance=tie_tolerance,
        degenerate=(k == 1),
        all_tied=all_tied,
        n_scorable=len(scorable),
        scores=scores,
    )


def select_best_of_k(
    coeffs: ACTCoefficients,
    actor: EPA | tuple[float, float, float] | np.ndarray,
    object_: EPA | tuple[float, float, float] | np.ndarray,
    candidates: Sequence[str],
    behaviours: Sequence[EPA | tuple[float, float, float] | np.ndarray | None],
    tie_tolerance: float = 1e-12,
) -> BestOfKSelection:
    """Pure best-of-k selection over pre-computed candidate behaviour EPAs.

    Args:
        coeffs: The impression-formation coefficient set.
        actor: The actor's (agent's) post-event-1 self-fundamental.
        object_: The object's (user's) post-event-1 self-fundamental.
        candidates: The candidate response texts.
        behaviours: The behaviour EPA per candidate (None marks a
            candidate as unscorable).
        tie_tolerance: Absolute deflection tolerance for tie-breaking.

    Returns:
        The selection with the per-candidate scores.

    Raises:
        ValueError: If candidates is empty, the lengths disagree, the
            tie tolerance is negative, or an EPA value is malformed.
    """
    k = len(candidates)
    if k == 0:
        raise ValueError("candidates must be non-empty")
    if len(behaviours) != k:
        raise ValueError("candidates and behaviours must have the same length")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0")
    actor_arr = _epa_array("actor", actor)
    object_arr = _epa_array("object_", object_)

    scores: list[CandidateScore] = []
    for i, (text, b) in enumerate(zip(candidates, behaviours)):
        if b is None:
            scores.append(
                CandidateScore(i, text, None, None, error="no behaviour EPA provided")
            )
            continue
        try:
            b_arr = _epa_array("behaviour", b)
            u = float(coeffs.deflection(actor_arr, b_arr, object_arr))
            if not np.isfinite(u):
                raise ValueError("deflection is not finite")
        except Exception as exc:  # noqa: BLE001 - any scoring failure marks the candidate
            scores.append(
                CandidateScore(i, text, None, None, error=f"{type(exc).__name__}: {exc}")
            )
            continue
        scores.append(CandidateScore(i, text, EPA.from_array(b_arr), u, None))
    return _select(tuple(scores), k, tie_tolerance)


class BestOfKSelector:
    """Best-of-k selection with injected candidate generator and reader.

    Args:
        coeffs: The impression-formation coefficient set (used for the
            event-2 deflection of each candidate).
        reader: Callable mapping a candidate response text to the EPA of
            its behaviour (e.g. the M-X reader over the response).
        generator: Optional callable mapping the conversation state to a
            list of ``k`` candidate texts (model sampling lives here).
        k: The number of candidates to generate.
        tie_tolerance: Absolute deflection tolerance for tie-breaking.

    Raises:
        ValueError: If ``k < 1`` or ``tie_tolerance < 0``.
    """

    def __init__(
        self,
        coeffs: ACTCoefficients,
        *,
        reader: BehaviourReader,
        generator: CandidateGenerator | None = None,
        k: int = 4,
        tie_tolerance: float = 1e-12,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be >= 0")
        self.coeffs = coeffs
        self.reader = reader
        self.generator = generator
        self.k = k
        self.tie_tolerance = tie_tolerance

    def score_candidates(
        self, state: ConversationState, candidates: Sequence[str]
    ) -> tuple[CandidateScore, ...]:
        """Score candidates against the post-event-1 state fundamentals.

        Args:
            state: The conversation state after the user message has been
                recorded (post-event-1 fundamentals).
            candidates: The candidate response texts.

        Returns:
            Per-candidate scores, in candidate order. Unscorable
            candidates carry an ``error``.

        Raises:
            ValueError: If candidates is empty or a reader output is
                malformed.
        """
        k = len(candidates)
        if k == 0:
            raise ValueError("candidates must be non-empty")
        scores: list[CandidateScore] = []
        for i, text in enumerate(candidates):
            try:
                b_arr = _epa_array("behaviour", self.reader(text))
                u = float(
                    self.coeffs.deflection(
                        state.agent_fundamentals.as_array(),
                        b_arr,
                        state.user_fundamentals.as_array(),
                    )
                )
                if not np.isfinite(u):
                    raise ValueError("deflection is not finite")
            except Exception as exc:  # noqa: BLE001 - reader failures mark the candidate
                scores.append(
                    CandidateScore(i, text, None, None, error=f"{type(exc).__name__}: {exc}")
                )
                continue
            scores.append(CandidateScore(i, text, EPA.from_array(b_arr), u, None))
        return tuple(scores)

    def select(self, state: ConversationState, candidates: Sequence[str]) -> BestOfKSelection:
        """Score and select among explicit candidate texts.

        Args:
            state: The conversation state (post-event-1 fundamentals).
            candidates: The candidate response texts.

        Returns:
            The best-of-k selection.
        """
        scores = self.score_candidates(state, candidates)
        return _select(scores, len(candidates), self.tie_tolerance)

    def generate_and_select(
        self, state: ConversationState
    ) -> tuple[tuple[str, ...], BestOfKSelection]:
        """Generate ``k`` candidates via the generator and select.

        Args:
            state: The conversation state (post-event-1 fundamentals).

        Returns:
            A tuple ``(candidates, selection)``.

        Raises:
            ValueError: If no generator is configured or the generator
                produces a different number of candidates than ``k``.
        """
        if self.generator is None:
            raise ValueError("no candidate generator configured")
        candidates = tuple(self.generator(state))
        if len(candidates) != self.k:
            msg = f"generator produced {len(candidates)} candidates, expected k={self.k}"
            raise ValueError(msg)
        return candidates, self.select(state, candidates)
