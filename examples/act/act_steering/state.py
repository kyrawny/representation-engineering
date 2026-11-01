"""Multi-turn conversation state with ACT deflection tracking.

Maintains the post-event self-fundamentals of the user and the agent across
turns, accumulates a per-turn deflection trace, applies the documented
sentiment-drift decay, and latches the unresolved-deflection trigger when
the event-1 deflection exceeds a threshold for a sustained number of
consecutive turns.

The drift rule is a documented heuristic, not a canonical ACT construct:
ACT impression formation has no built-in forgetting. With
``drift_decay < 1`` the deviation of each fundamental from its identity
baseline is multiplied by ``drift_decay`` after every completed turn, which
prevents long conversations from drifting into degenerate states. The
default ``drift_decay = 1.0`` is the pure ACT two-event update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .act_core import EPA, ACTCoefficients, optimal_response, two_event_state_update
from .config import StateConfig
from .dictionary import ACTDictionary

__all__ = ["StateConfig", "ConversationState"]


def _as_epa(value: EPA | tuple[float, float, float] | np.ndarray) -> EPA:
    """Coerce an EPA-like value (EPA, 3-tuple, or array) to an EPA."""
    if isinstance(value, EPA):
        return value
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected an EPA with 3 components, got shape {arr.shape}")
    return EPA.from_array(arr)


@dataclass
class ConversationState:
    """Multi-turn ACT conversation state.

    Attributes:
        user_term: User identity label (dictionary term).
        agent_term: Agent identity label (dictionary term).
        user_baseline: User identity EPA (drift anchor).
        agent_baseline: Agent identity EPA (drift anchor).
        user_fundamentals: User's current post-event self-fundamental.
        agent_fundamentals: Agent's current post-event self-fundamental.
        config: State configuration (drift, trigger).
        turn: Number of completed turns.
        messages: Number of user messages recorded.
        deflection_trace: Per-turn record dicts, appended in order.
        unresolved_streak: Consecutive unresolved user messages (U1 > threshold).
        trigger_fired: Latched unresolved-deflection trigger.
        trigger_turn: 1-based index of the user message that fired the trigger
            (None if unfired).
    """

    user_term: str
    agent_term: str
    user_baseline: EPA
    agent_baseline: EPA
    user_fundamentals: EPA
    agent_fundamentals: EPA
    config: StateConfig = field(default_factory=StateConfig)
    turn: int = 0
    messages: int = 0
    deflection_trace: list[dict] = field(default_factory=list)
    unresolved_streak: int = 0
    trigger_fired: bool = False
    trigger_turn: int | None = None

    _pending_message: tuple[str, EPA, float] | None = field(default=None, repr=False)

    @classmethod
    def from_ids(
        cls,
        dictionary: ACTDictionary,
        user_term: str,
        agent_term: str,
        config: StateConfig | None = None,
    ) -> ConversationState:
        """Create a state from dictionary identity terms."""
        user = dictionary.require(user_term)
        agent = dictionary.require(agent_term)
        return cls(
            user_term=user.term,
            agent_term=agent.term,
            user_baseline=EPA.from_array(np.array(user.as_vector(), dtype=np.float64)),
            agent_baseline=EPA.from_array(np.array(agent.as_vector(), dtype=np.float64)),
            user_fundamentals=EPA.from_array(np.array(user.as_vector(), dtype=np.float64)),
            agent_fundamentals=EPA.from_array(np.array(agent.as_vector(), dtype=np.float64)),
            config=config if config is not None else StateConfig(),
        )

    def record_user_message(
        self,
        coeffs: ACTCoefficients,
        message_epa: EPA | tuple[float, float, float] | np.ndarray,
        label: str = "",
    ) -> float:
        """Process event 1: the user acts on the agent with the message.

        Updates the post-event fundamentals, evaluates the trigger, and
        returns the event-1 deflection U.

        Args:
            coeffs: The coefficient set.
            message_epa: EPA of the user message behaviour.
            label: The message behaviour term (logged in the trace).

        Returns:
            The event-1 deflection U.
        """
        b = _as_epa(message_epa)
        user_after, agent_after, u1 = two_event_state_update(
            coeffs, self.user_fundamentals, self.agent_fundamentals, b
        )
        self.user_fundamentals = user_after
        self.agent_fundamentals = agent_after
        self._pending_message = (label, b, u1)
        self.messages += 1
        if u1 > self.config.deflection_threshold:
            self.unresolved_streak += 1
            if self.unresolved_streak >= self.config.sustained_turns and not self.trigger_fired:
                self.trigger_fired = True
                self.trigger_turn = self.messages
        else:
            self.unresolved_streak = 0
        return u1

    def optimal_response_target(self, coeffs: ACTCoefficients) -> EPA:
        """Return b* for the current state (event-2 target)."""
        return optimal_response(coeffs, self.user_fundamentals, self.agent_fundamentals)

    def record_agent_response(
        self,
        coeffs: ACTCoefficients,
        response_epa: EPA | tuple[float, float, float] | np.ndarray,
        label: str = "",
    ) -> float:
        """Process event 2 and complete the turn.

        Records the deflection trace entry, applies the drift decay, and
        increments the turn counter.

        Args:
            coeffs: The coefficient set.
            response_epa: EPA of the agent response behaviour.
            label: The response behaviour term (logged in the trace).

        Returns:
            The event-2 deflection U for the given response.

        Raises:
            RuntimeError: If no user message is pending for this turn.
        """
        if self._pending_message is None:
            raise RuntimeError("record_agent_response called without a pending user message")
        b = _as_epa(response_epa)
        b_star = self.optimal_response_target(coeffs)
        u2 = coeffs.deflection(
            self.agent_fundamentals.as_array(), b.as_array(), self.user_fundamentals.as_array()
        )
        w = coeffs.impressions(self.agent_fundamentals, b, self.user_fundamentals)
        self.agent_fundamentals = EPA.from_array(w[0:3])
        self.user_fundamentals = EPA.from_array(w[6:9])
        label_msg, b_msg, u1 = self._pending_message
        self.deflection_trace.append(
            {
                "turn": self.turn,
                "user_message": label_msg,
                "user_message_epa": [float(v) for v in b_msg.as_array()],
                "u1": float(u1),
                "b_star": [float(v) for v in b_star.as_array()],
                "response": label,
                "response_epa": [float(v) for v in b.as_array()],
                "u2": float(u2),
            }
        )
        self._apply_drift()
        self.turn += 1
        self._pending_message = None
        return u2

    def _apply_drift(self) -> None:
        """Apply the documented sentiment-drift decay to both fundamentals."""
        decay = self.config.drift_decay
        if decay == 1.0:
            return
        self.user_fundamentals = self._drifted(self.user_fundamentals, self.user_baseline, decay)
        self.agent_fundamentals = self._drifted(self.agent_fundamentals, self.agent_baseline, decay)

    @staticmethod
    def _drifted(fundamentals: EPA, baseline: EPA, decay: float) -> EPA:
        """Return ``baseline + decay * (fundamentals - baseline)``."""
        f = fundamentals.as_array()
        base = baseline.as_array()
        return EPA.from_array(base + decay * (f - base))
