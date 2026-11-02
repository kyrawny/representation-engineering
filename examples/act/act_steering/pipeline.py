"""Per-turn ACT deflection-steering control loop (S10).

Implements the synthesis's control loop for one user turn:

1. **Quantify the event** — read the EPA of the user message (event-1
   behaviour) and process it through the conversation state: post-event
   fundamentals, event-1 deflection ``U_pre``, unresolved-deflection
   trigger.
2. **Generate k candidates** — the injected candidate generator applies the
   steering (Mode 1/2/3 wiring lives in the generator).
3. **Score post-event deflection** — the best-of-k selector scores each
   candidate as ``deflection(agent_fundamentals, b_r, user_fundamentals)``
   and picks the argmin.
4. **Monitor** — score the winning response with the per-token deflection
   trace (optional).
5. **Repair (re-identification)** — when the turn is flagged (``U_post``
   above the repair threshold, a monitor-trace fire, or the latched state
   trigger) and a repair policy is configured, try up to
   ``max_repair_attempts`` repair candidates and adopt the best one if it
   scores strictly lower than the selected response.
6. **Record the response** — event 2 through the state (drift included)
   and a JSON-safe turn entry appended to the trace log.

Pure logic — torch-free. The model-dependent components (message reader,
candidate generator, behaviour reader, monitor, repair policy) are
injected callables, so the loop is unit-testable on CPU.
``build_mock_components`` provides a deterministic mock used by the tests
and the CLI ``--mock`` mode.

CLI
---
::

    python -m act_steering.pipeline --mock --script script.json --out trace.json --k 4
    python -m act_steering.pipeline --quick --out trace.json

``script.json`` is ``{"user": "<identity term>", "agent": "<identity
term>", "messages": ["...", "..."]}``. ``--quick`` runs a built-in two-message
smoke script with the mock components. The real model wiring (RepE steerer
as generator, M-X reader as message/behaviour reader, monitor over hidden
states) is added in the GPU phase on top of this loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .act_core import EPA, ACTCoefficients, optimal_response
from .config import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DataPaths,
    MonitorConfig,
    PipelineConfig,
    SteeringConfig,
)
from .dictionary import ACTDictionary, EPATerm
from .directions import DirectionsBundle
from .monitor import (
    DeflectionMonitor,
    TokenTrace,
    make_hidden_provider,
    resolve_reader_layers,
)
from .readers import build_epa_reader
from .selector import BehaviourReader, BestOfKSelector
from .state import ConversationState
from .steerer import DeflectionSteerer, proportional_gain
from .stimuli import CONTEXT_TEMPLATE

__all__ = [
    "ACTSteeringPipeline",
    "MonitorFn",
    "PipelineComponents",
    "RepairPolicy",
    "ANTI_SYCOPHANCY_LINE",
    "PromptSpec",
    "build_messages",
    "build_mock_components",
    "build_prompt",
    "build_real_components",
    "chat_user_prompt",
    "descriptors_for",
    "render_prompt",
    "response_after",
    "role_system_prompt",
    "system_prompt_for",
    "main",
    "SteerCondition",
    "compute_target_bstar",
    "random_direction",
    "shuffled_identity_pair",
]

#: Callable mapping the user message text to its behaviour EPA.
MessageReader = Callable[[str], EPA | tuple[float, float, float] | np.ndarray]
#: Callable mapping the state to the list of candidate response texts.
CandidateGenerator = Callable[[ConversationState], Sequence[str]]
#: Callable mapping (state, chosen response) to repair candidate texts.
RepairPolicy = Callable[[ConversationState, str], Sequence[str]]
#: Callable mapping a response text to its per-token deflection trace.
MonitorFn = Callable[[str], TokenTrace]

_NEUTRAL = np.zeros(3, dtype=np.float64)


@dataclass
class PipelineComponents:
    """Injectable model-dependent components of the control loop.

    Attributes:
        message_reader: Reads the EPA of the user message (event-1
            behaviour).
        candidate_generator: Generates the k candidate responses (steering
            applied internally); receives the post-event-1 state.
        behaviour_reader: Reads the EPA of a response behaviour.
        monitor: Optional per-token deflection monitor over a response text.
        repair_policy: Optional re-identification repair policy; returns
            repair candidate texts for a flagged turn.
    """

    message_reader: MessageReader
    candidate_generator: CandidateGenerator
    behaviour_reader: BehaviourReader
    monitor: MonitorFn | None = None
    repair_policy: RepairPolicy | None = None


def _epa_array(name: str, value: EPA | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Validate and coerce an EPA value to a float (3,) array."""
    if isinstance(value, EPA):
        value = value.as_array()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} EPA must have shape (3,), got {arr.shape}")
    return arr


class ACTSteeringPipeline:
    """The per-turn read-compute-steer-select-monitor-repair loop.

    Args:
        coeffs: The impression-formation coefficient set.
        state: The multi-turn conversation state (updated in place).
        components: The injected model-dependent components.
        config: Control-loop settings.
        selector: Optional pre-built selector (defaults to a best-of-k
            selector over ``components.behaviour_reader``).

    Raises:
        ValueError: If the config is invalid.
    """

    def __init__(
        self,
        coeffs: ACTCoefficients,
        state: ConversationState,
        components: PipelineComponents,
        config: PipelineConfig,
        *,
        selector: BestOfKSelector | None = None,
    ) -> None:
        self.coeffs = coeffs
        self.state = state
        self.components = components
        self.config = config
        # The reader is read through the components at call time, so
        # swapping ``components.behaviour_reader`` (tests, hot reloads) is
        # picked up without rebuilding the selector.
        self.selector = selector or BestOfKSelector(
            coeffs,
            reader=lambda text: self.components.behaviour_reader(text),
            k=config.k,
        )
        self.turn_log: list[dict] = []

    # -- helpers ------------------------------------------------------------

    def _read(self, reader: Callable[[str], Any], text: str) -> tuple[EPA, bool]:
        """Read an EPA with the given reader; failures map to neutral EPA.

        Returns:
            A tuple ``(epa, failed)``; ``failed`` is True when the reader
            raised or returned a malformed value (the EPA is then the
            neutral (0, 0, 0) placeholder and must be logged as such).
        """
        try:
            return EPA.from_array(_epa_array("epa", reader(text))), False
        except Exception:  # noqa: BLE001 - reader failures degrade to neutral
            return EPA.from_array(_NEUTRAL.copy()), True

    def _deflection(self, behaviour: np.ndarray) -> float | None:
        """Post-event-1 deflection for a candidate behaviour (None on error)."""
        try:
            u = float(
                self.coeffs.deflection(
                    self.state.agent_fundamentals.as_array(),
                    behaviour,
                    self.state.user_fundamentals.as_array(),
                )
            )
            return u if np.isfinite(u) else None
        except Exception:  # noqa: BLE001 - degenerate deflections degrade to None
            return None

    # -- main loop ------------------------------------------------------------

    def run_turn(self, user_text: str) -> dict:
        """Run one user turn through the control loop.

        Args:
            user_text: The user's message text.

        Returns:
            The JSON-safe turn log entry (also appended to ``turn_log``).

        Raises:
            ValueError: If the candidate generator produces no candidates.
        """
        state = self.state
        # 1. event 1: the user acts on the agent with the message
        b_u, message_read_failed = self._read(self.components.message_reader, user_text)
        u1 = state.record_user_message(self.coeffs, b_u, label=user_text)

        # 2. generate k candidates (steering applied inside the generator)
        candidates = tuple(self.components.candidate_generator(state))
        if not candidates:
            raise ValueError("candidate generator produced no candidates")

        # 3. score post-event deflection of each candidate; pick the argmin
        selection = self.selector.select(state, candidates)

        response: str | None = None
        b_final = np.zeros(3, dtype=np.float64)
        u_post: float | None = None
        fallback: str | None = None
        if selection.winner is not None:
            response = selection.winner.text
            b_final = selection.winner.behaviour_epa.as_array()
            u_post = selection.winner.deflection
        elif self.config.fallback_response is not None:
            fallback = "candidate-scoring-failed"
            fb = self.config.fallback_response
            b_fb, read_failed = self._read(self.components.behaviour_reader, fb)
            u_fb = None if read_failed else self._deflection(b_fb)
            if u_fb is None:
                fallback = "fallback-epa-read-failed"
                b_final = _NEUTRAL.copy()
            else:
                response = self.config.fallback_response
                b_final, u_post = b_fb, u_fb
        else:
            fallback = "all-candidates-unscorable"
            u_post = self._deflection(_NEUTRAL)

        # 4. monitor the chosen response
        trace: TokenTrace | None = None
        if (
            self.components.monitor is not None
            and self.config.monitor_positions > 0
            and response is not None
        ):
            trace = self.components.monitor(response)

        # 5. repair (re-identification) when the turn is flagged
        flagged = (
            (u_post is not None and u_post > self.config.repair_threshold)
            or (trace is not None and trace.fired)
            or state.trigger_fired
        )
        repair_log: dict | None = None
        if (
            self.config.repair == "reidentify"
            and self.components.repair_policy is not None
            and flagged
            and response is not None
        ):
            repairs = list(self.components.repair_policy(state, response))
            repairs = repairs[: self.config.max_repair_attempts]
            best: tuple[str, np.ndarray, float] | None = None
            for text in repairs:
                b_r, read_failed = self._read(self.components.behaviour_reader, text)
                if read_failed:
                    continue
                u_r = self._deflection(b_r)
                if u_r is None:
                    continue
                if best is None or u_r < best[2]:
                    best = (text, b_r, u_r)
            if best is not None and u_post is not None and best[2] < u_post:
                repair_log = {
                    "adopted": True,
                    "text": best[0],
                    "from_u": float(u_post),
                    "to_u": float(best[2]),
                }
                response, b_final, u_post = best
            else:
                repair_log = {
                    "adopted": False,
                    "from_u": None if u_post is None else float(u_post),
                }

        # 6. record event 2 and complete the turn
        label = response if response is not None else "<no response>"
        u2 = state.record_agent_response(self.coeffs, b_final, label=label)
        entry = {
            "turn": state.turn - 1,
            "user_message": user_text,
            "message_epa": None if message_read_failed else [float(v) for v in b_u.as_array()],
            "message_epa_read_failed": message_read_failed,
            "u_pre": float(u1),
            "trigger_fired": bool(state.trigger_fired),
            "candidates": [s.to_dict() for s in selection.scores],
            "winner_index": selection.winner_index,
            "u_post": None if u_post is None else float(u_post),
            "response": response,
            "response_u": None if u2 is None else float(u2),
            "monitor": None if trace is None else trace.to_dict(),
            "repair": repair_log,
            "fallback": fallback,
        }
        self.turn_log.append(entry)
        return entry

    def run_conversation(self, user_messages: Sequence[str]) -> dict:
        """Run a scripted conversation and return the full trace log.

        Args:
            user_messages: The user's messages, in order.

        Returns:
            The JSON-safe conversation log.
        """
        return {
            "user_term": self.state.user_term,
            "agent_term": self.state.agent_term,
            "mode": self.config.mode,
            "turns": [self.run_turn(message) for message in user_messages],
        }

    def save_log(self, log: dict, path: Path | str) -> Path:
        """Write the trace log as JSON (creating parent directories)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(log, indent=2), encoding="utf-8")
        return out


# --- deterministic CPU mock (CLI --mock and tests) ----------------------------


def _deterministic_epa(text: str, salt: int) -> EPA:
    """Map text to a deterministic EPA in [-2, 2]^3 via SHA-256."""
    digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
    values = np.array([(digest[3 * i] / 255.0 - 0.5) * 2.0 for i in range(3)], dtype=np.float64)
    return EPA.from_array(values)


def build_mock_components(k: int = 4) -> PipelineComponents:
    """Build a deterministic CPU mock of the model-dependent components.

    Args:
        k: The number of candidates the mock generator produces.

    Returns:
        Components with hash-based readers and a templated generator.
    """
    return PipelineComponents(
        message_reader=lambda text: _deterministic_epa(text, salt=1),
        candidate_generator=lambda _state: [f"candidate-{j}" for j in range(k)],
        behaviour_reader=lambda text: _deterministic_epa(text, salt=2),
        monitor=None,
        repair_policy=None,
    )


# --- chat-template prompt construction (H3 ablations) ------------------------

#: Prompt style: ``"chat"`` renders through the tokenizer's chat template
#: (real model runs); ``"raw"`` keeps the flat continuation template (mock
#: mode and AAAI-27 reproduction of the first GPU run).
PromptStyle = Literal["chat", "raw"]
#: System prompt variants for the condition matrix.
SystemVariant = Literal["role", "act_descriptors", "anti_sycophancy"]

#: Fixed anti-sycophancy instruction (condition B2).
ANTI_SYCOPHANCY_LINE = (
    "Be honest and helpful even when the user disagrees with you, pushes "
    "back, or suggests a wrong answer. Never change a correct answer just "
    "to please the user."
)

#: Qualitative descriptor buckets per EPA dimension (high / neutral / low),
#: with the bucket threshold at +/-1.0 EPA units.
_DESCRIPTOR_BUCKETS: dict[str, tuple[str, str, str]] = {
    "e": ("warm and kind", "even-tempered", "cool and blunt"),
    "p": ("assertive and confident", "calm and measured", "soft and deferential"),
    "a": ("lively and energetic", "steady-paced", "slow and quiet"),
}


def role_system_prompt(user_term: str, agent_term: str) -> str:
    """Identity-role framing for the system prompt (condition B0).

    Args:
        user_term: The user identity label (dictionary term).
        agent_term: The agent identity label (dictionary term).

    Returns:
        A single system-prompt sentence fixing the agent's role.
    """
    return (
        f"You are a {agent_term} interacting with a person who is a "
        f"{user_term}. Respond naturally, in character, and keep your "
        "responses concise."
    )


def descriptors_for(
    epa: EPA | tuple[float, float, float] | np.ndarray,
) -> str:
    """Qualitative descriptor phrase for an EPA vector (condition B1).

    Each dimension maps to a high / neutral / low bucket (threshold
    +/-1.0 EPA units) and the three phrases join with commas.

    Args:
        epa: The EPA vector in (E, P, A) order.

    Returns:
        A phrase such as ``"warm and kind, calm and measured, lively and
        energetic"``.
    """
    e, p, a = _epa_array("epa", epa)

    def _bucket(key: str, value: float) -> str:
        if value > 1.0:
            return _DESCRIPTOR_BUCKETS[key][0]
        if value < -1.0:
            return _DESCRIPTOR_BUCKETS[key][2]
        return _DESCRIPTOR_BUCKETS[key][1]

    return ", ".join((_bucket("e", e), _bucket("p", p), _bucket("a", a)))


@dataclass(frozen=True)
class PromptSpec:
    """Prompt construction settings for one ablation condition.

    Attributes:
        style: ``"chat"`` (tokenizer chat template; the default for real
            model runs) or ``"raw"`` (flat continuation template; mock
            mode and AAAI-27 reproduction). The system fields are ignored
            for ``"raw"``.
        variant: System prompt variant: ``"role"`` (identity framing only,
            B0), ``"act_descriptors"`` (role framing plus qualitative
            descriptors of the ACT target, B1) or ``"anti_sycophancy"``
            (role framing plus a fixed anti-sycophancy instruction, B2).
        target: The b* target in (E, P, A) order used by the
            ``"act_descriptors"`` variant; ignored otherwise. ``None``
            falls back to the neutral EPA.
        system: Explicit system prompt override; replaces the
            variant-derived text when set.
    """

    style: PromptStyle = "chat"
    variant: SystemVariant = "role"
    target: tuple[float, float, float] | None = None
    system: str | None = None


def system_prompt_for(spec: PromptSpec, user_term: str, agent_term: str) -> str | None:
    """Resolve the system prompt text for one condition (pure).

    Args:
        spec: The prompt spec (variant / override / target).
        user_term: The user identity label.
        agent_term: The agent identity label.

    Returns:
        The system prompt text, or ``None`` when ``style`` is ``"raw"``.
    """
    if spec.style == "raw":
        return None
    if spec.system is not None:
        return spec.system
    base = role_system_prompt(user_term, agent_term)
    if spec.variant == "act_descriptors":
        target = spec.target if spec.target is not None else (0.0, 0.0, 0.0)
        return f"{base} Respond in a {descriptors_for(target)} way."
    if spec.variant == "anti_sycophancy":
        return f"{base} {ANTI_SYCOPHANCY_LINE}"
    return base


def build_messages(
    state: ConversationState, spec: PromptSpec | None = None
) -> list[dict[str, str]]:
    """Chat messages for the current turn (pure; no tokenizer).

    The system message (per ``spec``) comes first when the style is
    ``"chat"``; then the conversation: the first user message, each
    completed turn's agent reply (when recorded), and the pending
    (current) user message last. On the first turn the pending message
    *is* the first user message, so it appears only once.

    Args:
        state: The conversation state (a user message must be pending).
        spec: The prompt spec (defaults to chat style, role variant).

    Returns:
        The message list for ``tokenizer.apply_chat_template``.

    Raises:
        RuntimeError: If no user message is pending.
    """
    if state._pending_message is None:
        raise RuntimeError("build_messages called without a pending user message")
    pending = state._pending_message[0]
    trace = state.deflection_trace
    s = spec or PromptSpec()
    messages: list[dict[str, str]] = []
    system = system_prompt_for(s, state.user_term, state.agent_term)
    if system:
        messages.append({"role": "system", "content": system})
    first = trace[0]["user_message"] if trace else pending
    messages.append({"role": "user", "content": first})
    for i, entry in enumerate(trace):
        response = entry.get("response")
        if response:
            messages.append({"role": "assistant", "content": response})
        if i + 1 < len(trace):
            messages.append({"role": "user", "content": trace[i + 1]["user_message"]})
    if not trace:
        # The pending message is the first user message (already added).
        return messages
    messages.append({"role": "user", "content": pending})
    return messages


def render_prompt(messages: Sequence[dict[str, str]], tokenizer: Any) -> str:
    """Render chat messages through the tokenizer's chat template.

    Args:
        messages: The message list from :func:`build_messages`.
        tokenizer: A tokenizer exposing ``apply_chat_template``.

    Returns:
        The rendered prompt ending at the generation (assistant) position.
    """
    return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)


def chat_user_prompt(text: str, tokenizer: Any) -> str:
    """Render a single user turn through the tokenizer chat template.

    For evaluation scripts (sycophancy dialogues, dose-response probes)
    that keep fixed raw prompt strings: the text becomes the user message
    and the prompt ends at the assistant generation position, so an
    instruct model answers the turn instead of continuing a flat
    transcript.

    Args:
        text: The raw prompt text (e.g., a QA question).
        tokenizer: A tokenizer exposing ``apply_chat_template``.

    Returns:
        The rendered prompt ending at the generation (assistant) position.

    Raises:
        ValueError: If ``tokenizer`` is None.
    """
    if tokenizer is None:
        raise ValueError("chat_user_prompt requires a tokenizer")
    return render_prompt([{"role": "user", "content": text}], tokenizer)


def build_prompt(
    state: ConversationState,
    tokenizer: Any | None = None,
    spec: PromptSpec | None = None,
) -> str:
    """The generation prompt for the current turn.

    With ``spec=None`` (the historical default) the raw flat continuation
    template is used: the first user message fills
    :data:`CONTEXT_TEMPLATE`, each completed turn contributes the agent
    reply (when recorded) followed by the next user message, and the
    pending (current) message comes last. With a chat-style spec the
    messages are rendered through ``tokenizer.apply_chat_template`` with
    the system role (``tokenizer`` is then required); a raw-style spec
    ignores the system fields.

    Args:
        state: The conversation state (a user message must be pending).
        tokenizer: Required for chat-style specs.
        spec: The prompt spec (defaults to the raw template).

    Returns:
        The prompt text to generate the agent response from.

    Raises:
        RuntimeError: If no user message is pending.
        ValueError: If a chat-style spec is given without a tokenizer.
    """
    if spec is not None and spec.style == "chat":
        if tokenizer is None:
            raise ValueError("chat-style build_prompt requires a tokenizer")
        return render_prompt(build_messages(state, spec), tokenizer)
    if state._pending_message is None:
        raise RuntimeError("build_prompt called without a pending user message")
    pending = state._pending_message[0]
    trace = state.deflection_trace
    if not trace:
        return CONTEXT_TEMPLATE.format(
            user=state.user_term, agent=state.agent_term, message=pending
        )
    parts = [
        CONTEXT_TEMPLATE.format(
            user=state.user_term, agent=state.agent_term, message=trace[0]["user_message"]
        )
    ]
    for i, entry in enumerate(trace):
        response = entry.get("response")
        if response:
            parts.append(f"You replied: {response}.")
        if i + 1 < len(trace):
            parts.append(f"The person says: {trace[i + 1]['user_message']}.")
    parts.append(f"The person says: {pending}.")
    return " ".join(parts)


#: Trailing stop tokens that tokenizers can leave at the end of decoded
#: generations (chat end-of-turn, base end-of-text).
_TRAILING_STOP_TOKENS = ("<|eot_id|>", "<|end_of_text|>")


def response_after(full: Any, prompt: str) -> str:
    """Extract the generated response from a generation-pipeline output.

    The prompt prefix is stripped when present, surrounding whitespace is
    removed, and trailing end-of-turn special tokens (``<|eot_id|>``,
    ``<|end_of_text|>``) are dropped, so the returned text contains only
    the model's reply.

    Args:
        full: The pipeline output (text, list of texts, or text dicts).
        prompt: The prompt the response continues.

    Returns:
        The cleaned response text.
    """
    text = full[0] if isinstance(full, (list, tuple)) else full
    if isinstance(text, dict):
        text = text.get("generated_text", "")
    text = str(text)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    text = text.strip()
    for token in _TRAILING_STOP_TOKENS:
        if text.endswith(token):
            text = text[: -len(token)]
    return text.strip()


# --- steer conditions (H3 ablation matrix) ----------------------------------

#: Steering axes for the ablation matrix.
SteerAxis = Literal["none", "deflection", "epa_target", "random"]
#: b* target kinds for the ``"epa_target"`` axis.
TargetKind = Literal["situation", "identity_only", "shuffled", "constant"]

#: Large odd prime mixing the scenario index into condition seeds so
#: different scenarios draw different (deterministic) pairs/directions.
_SCENARIO_SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class SteerCondition:
    """Steering mechanism for one ablation condition (real components).

    Attributes:
        name: The condition label (matches the ablation matrix).
        axis: ``"none"`` (unsteered baseline), ``"deflection"``
            (deflection-contrast direction with proportional gain — the
            synthesis's primary axis), ``"epa_target"`` (AAAI-27 style:
            per-dimension EPA directions scaled to the b* target
            components) or ``"random"`` (seeded random direction, offset
            norm matched to the deflection gain at the same U_pre).
        target_kind: For the ``"epa_target"`` axis — which b* target to
            steer toward: ``"situation"`` (full two-event ACT target;
            M1), ``"identity_only"`` (b* from the identity baselines, i.e.
            the user message is not read; B6), ``"shuffled"`` (b* from a
            seeded random identity pair; B7) or ``"constant"`` (one fixed
            EPA; B8). Ignored by the other axes.
        constant_target: Fixed (E, P, A) for ``target_kind="constant"``.
        seed: Master seed for the shuffled pair / random direction.
        scenario_index: Index of the current scenario; added to the seed
            (via :data:`_SCENARIO_SEED_STRIDE`) so each scenario gets a
            distinct deterministic draw.

    Raises:
        ValueError: If the axis/target_kind combination is invalid.
    """

    name: str
    axis: SteerAxis = "none"
    target_kind: TargetKind | None = None
    constant_target: tuple[float, float, float] | None = None
    seed: int = 0
    scenario_index: int = 0

    def __post_init__(self) -> None:
        if self.axis == "epa_target":
            if self.target_kind not in ("situation", "identity_only", "shuffled", "constant"):
                raise ValueError(
                    f"epa_target axis requires a target kind, got {self.target_kind!r}"
                )
            if self.target_kind == "constant" and self.constant_target is None:
                raise ValueError("constant target kind requires constant_target")
        elif self.target_kind is not None:
            raise ValueError(
                f"target_kind {self.target_kind!r} is only valid for the "
                f"epa_target axis (got axis={self.axis!r})"
            )
        if self.scenario_index < 0:
            raise ValueError("scenario_index must be >= 0")


def _scenario_seed(seed: int, scenario_index: int) -> int:
    """Deterministic per-scenario seed from the master seed."""
    return seed + _SCENARIO_SEED_STRIDE * (scenario_index + 1)


def shuffled_identity_pair(
    dictionary: ACTDictionary, seed: int, scenario_index: int = 0
) -> tuple[EPATerm, EPATerm]:
    """Draw a deterministic random (user, agent) identity pair (condition B7).

    The pair is drawn from ``dictionary.identities`` with a seed derived
    from the master seed and the scenario index, so each scenario gets a
    distinct reproducible pair and the identity correspondence with the
    scenario is broken while the target distribution is preserved.

    Args:
        dictionary: The ACT dictionary (identity source).
        seed: Master seed.
        scenario_index: Scenario index (distinct draws per scenario).

    Returns:
        A ``(user, agent)`` pair of :class:`~act_steering.dictionary.EPATerm`
        (distinct terms).

    Raises:
        ValueError: If the dictionary holds fewer than two identities.
    """
    n = len(dictionary.identities)
    if n < 2:
        raise ValueError(f"need at least two identities, got {n}")
    rng = np.random.default_rng(_scenario_seed(seed, scenario_index))
    i = int(rng.integers(n))
    j = int(rng.integers(n - 1))
    if j >= i:
        j += 1
    return dictionary.identities[i], dictionary.identities[j]


def compute_target_bstar(
    coeffs: ACTCoefficients,
    state: ConversationState,
    kind: TargetKind = "situation",
    *,
    dictionary: ACTDictionary | None = None,
    seed: int = 0,
    scenario_index: int = 0,
    constant_target: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Compute the b* target for one condition (pure; ACT core only).

    - ``"situation"``: the full two-event ACT target from the current
      post-event-1 fundamentals (message read in; M1).
    - ``"identity_only"``: b* from the identity baseline fundamentals —
      equivalent to reading a neutral user message (B6).
    - ``"shuffled"``: b* from a seeded random identity pair drawn from
      ``dictionary.identities`` (seed + scenario index) — preserves the
      target *distribution* while breaking the identity correspondence
      (B7).
    - ``"constant"``: the fixed ``constant_target`` (B8).

    Args:
        coeffs: The impression-formation coefficient set.
        state: The conversation state (post event 1).
        kind: The target kind.
        dictionary: Required for ``"shuffled"``.
        seed: Master seed (``"shuffled"`` only).
        scenario_index: Scenario index (``"shuffled"`` only).
        constant_target: Fixed EPA (``"constant"`` only).

    Returns:
        The b* target as a float (3,) array in (E, P, A) order.

    Raises:
        ValueError: If a kind is missing its required arguments.
    """
    if kind == "situation":
        return state.optimal_response_target(coeffs).as_array()
    if kind == "identity_only":
        return optimal_response(coeffs, state.user_baseline, state.agent_baseline).as_array()
    if kind == "constant":
        if constant_target is None:
            raise ValueError("constant target kind requires constant_target")
        return np.asarray(constant_target, dtype=np.float64)
    if kind == "shuffled":
        if dictionary is None:
            raise ValueError("shuffled target kind requires the dictionary")
        user_term, agent_term = shuffled_identity_pair(dictionary, seed, scenario_index)
        user = EPA.from_array(np.asarray(user_term.as_vector(), dtype=np.float64))
        agent = EPA.from_array(np.asarray(agent_term.as_vector(), dtype=np.float64))
        return optimal_response(coeffs, user, agent).as_array()
    raise ValueError(f"unknown target kind {kind!r}")


def random_direction(hidden_size: int, seed: int, scenario_index: int = 0) -> np.ndarray:
    """A seeded unit direction (deterministic per seed / scenario).

    Args:
        hidden_size: The model's hidden dimension.
        seed: Master seed.
        scenario_index: Scenario index (distinct draws per scenario).

    Returns:
        A float (hidden_size,) unit vector.

    Raises:
        ValueError: If ``hidden_size`` is not positive.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    rng = np.random.default_rng(_scenario_seed(seed, scenario_index))
    v = rng.standard_normal(hidden_size)
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not np.isfinite(norm):
        # Practically unreachable for a Gaussian draw; stay total.
        v = np.zeros(hidden_size)
        v[0] = 1.0
        return v
    return v / norm


def build_real_components(
    model: Any,
    tokenizer: Any,
    bundle: DirectionsBundle,
    config: PipelineConfig,
    *,
    steering: SteeringConfig | None = None,
    monitor_config: MonitorConfig | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    steerer_cls: type = DeflectionSteerer,
    condition: SteerCondition | None = None,
    coeffs: ACTCoefficients | None = None,
    dictionary: ACTDictionary | None = None,
    prompt_spec: PromptSpec | None = None,
    read_format: str = "raw",
    scenario_index_provider: Callable[[], int] | None = None,
) -> PipelineComponents:
    """Build the real model-dependent components for the control loop.

    The message/behaviour reader projects the last-token hidden state
    onto the bundle's per-layer EPA directions; the candidate generator
    applies the configured :class:`SteerCondition` (by default the
    static deflection steerer with proportional gain driven by the
    pending message's event-1 deflection); the optional monitor scores
    responses with the per-token deflection trace; the optional repair
    policy resamples one candidate with sampling turned on.

    Args:
        model: HuggingFace causal LM (mutated by the steerer wrapper).
        tokenizer: Its tokenizer.
        bundle: The directions artifact (S6).
        config: The control-loop settings (``k``, monitor, repair).
        steering: Steerer settings (defaults to ``SteeringConfig()``).
        monitor_config: Monitor settings (defaults to ``MonitorConfig()``).
        max_new_tokens: Generation length cap per candidate.
        temperature: Sampling temperature for k > 1 candidates.
        steerer_cls: Steerer class (injectable for tests; defaults to
            :class:`~act_steering.steerer.DeflectionSteerer`).
        condition: The ablation steering condition (defaults to the
            deflection axis — the historical behavior).
        coeffs: Required when the condition's target kind or the prompt
            spec's ``act_descriptors`` variant needs ACT targets.
        dictionary: Required for the ``"shuffled"`` target kind.
        prompt_spec: Prompt construction settings (chat template +
            system-role variant); ``None`` keeps the raw flat prompt.
        read_format: Reader/monitor read format (``"raw"`` or
            ``"chat"``; must match the directions fit format).
        scenario_index_provider: Callable returning the current scenario
            index (battery mode: one steerer, many scenarios). When
            absent, ``condition.scenario_index`` is used for the
            per-scenario ``"random"`` / ``"shuffled"`` draws.

    Returns:
        The :class:`PipelineComponents` wired for real model runs.

    Raises:
        ValueError: If the condition/prompt settings are unsatisfied
            (missing ``coeffs`` / ``dictionary``).
    """
    cond = condition or SteerCondition(name="steer", axis="deflection")
    needs_act = cond.axis in ("epa_target",) or (
        prompt_spec is not None and prompt_spec.variant == "act_descriptors"
    )
    if needs_act and coeffs is None:
        raise ValueError(
            "coeffs is required for the epa_target axis or the act_descriptors prompt variant"
        )
    if cond.target_kind == "shuffled" and dictionary is None:
        raise ValueError("dictionary is required for the shuffled target kind")

    steerer = steerer_cls(model, tokenizer, bundle, steering or SteeringConfig())

    def _scenario_index() -> int:
        """Current scenario index (provider wins; condition is the fallback)."""
        if scenario_index_provider is not None:
            return int(scenario_index_provider())
        return int(cond.scenario_index)

    reader = build_epa_reader(model, tokenizer, bundle, read_format=read_format)
    steer_layers = list(getattr(steerer, "layers", bundle.hidden_layers))

    def _generate(state: ConversationState, *, sample: bool) -> str:
        u_pre = state._pending_message[2]
        kwargs: dict[str, Any] = (
            {
                "do_sample": True,
                "temperature": max(temperature, 0.2),
                "top_p": 0.95,
            }
            if sample
            else {"do_sample": False}
        )
        steer_target: np.ndarray | None = None
        if cond.axis == "epa_target":
            steer_target = compute_target_bstar(
                coeffs,
                state,
                cond.target_kind,
                dictionary=dictionary,
                seed=cond.seed,
                scenario_index=_scenario_index(),
                constant_target=cond.constant_target,
            )
        spec = prompt_spec
        if spec is not None and spec.style == "chat" and spec.variant == "act_descriptors":
            desc_target = steer_target
            if desc_target is None:
                desc_target = compute_target_bstar(coeffs, state, "situation")
            if tuple(desc_target) != tuple(spec.target or (0.0, 0.0, 0.0)):
                spec = PromptSpec(
                    style=spec.style,
                    variant=spec.variant,
                    target=tuple(desc_target),
                    system=spec.system,
                )
        prompt = build_prompt(state, tokenizer, spec)
        if cond.axis == "none":
            full = steerer.generate_plain(
                prompt, max_new_tokens=max_new_tokens, **kwargs, return_text=True
            )
        elif cond.axis == "deflection":
            full = steerer.generate(
                prompt,
                u_pre=u_pre,
                mode="deflection",
                max_new_tokens=max_new_tokens,
                **kwargs,
                return_text=True,
            )
        elif cond.axis == "epa_target":
            full = steerer.generate(
                prompt,
                u_pre=u_pre,
                mode="epa_target",
                target=steer_target,
                max_new_tokens=max_new_tokens,
                **kwargs,
                return_text=True,
            )
        else:  # "random": seeded direction, norm matched to the deflection gain
            gain = proportional_gain(u_pre, steerer.config)
            offsets = {
                layer: gain * random_direction(bundle.hidden_size, cond.seed, _scenario_index())
                for layer in steer_layers
            }
            full = steerer.generate_with_offsets(
                prompt,
                offsets,
                max_new_tokens=max_new_tokens,
                **kwargs,
                return_text=True,
            )
        return response_after(full, prompt)

    def generator(state: ConversationState) -> list[str]:
        if config.k == 1:
            return [_generate(state, sample=False)]
        return [_generate(state, sample=True) for _ in range(config.k)]

    monitor_fn: MonitorFn | None = None
    if config.monitor_positions > 0:
        mconfig = monitor_config or MonitorConfig()
        provider = make_hidden_provider(
            model, tokenizer, resolve_reader_layers(bundle, mconfig), read_format=read_format
        )
        monitor_impl = DeflectionMonitor(provider, bundle, mconfig)

        def monitor_fn(text: str) -> TokenTrace:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            return monitor_impl.trace(text, max(1, min(n_tokens, config.monitor_positions)))

    repair_policy: RepairPolicy | None = None
    if config.repair == "reidentify":

        def repair_policy(state: ConversationState, response: str) -> list[str]:
            return [_generate(state, sample=True)]

    return PipelineComponents(
        message_reader=reader,
        candidate_generator=generator,
        behaviour_reader=reader,
        monitor=monitor_fn,
        repair_policy=repair_policy,
    )


# --- CLI -----------------------------------------------------------------------

#: Built-in script for ``--quick`` smoke mode.
QUICK_SCRIPT: dict[str, object] = {
    "user": "student",
    "agent": "teacher",
    "messages": [
        "I think I can finish the whole course in one weekend.",
        "Can you just give me the answer to the last homework question?",
    ],
}


def _build_real_cli_components(
    args: argparse.Namespace,
    config: PipelineConfig,
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
) -> PipelineComponents:
    """Build the real (model-bound) components from parsed CLI arguments.

    The generation prompt uses the ``--prompt-style`` / ``--system-variant``
    settings (default: the tokenizer chat template with the role system
    prompt — required for instruct models, which otherwise continue the
    flat transcript instead of answering the turn).

    Args:
        args: Parsed CLI arguments (``--model``, ``--directions``, ...).
        config: The control-loop settings to wire.
        coeffs: ACT coefficients (needed by the ``act_descriptors``
            prompt variant).
        dictionary: The ACT dictionary (needed by the ``act_descriptors``
            prompt variant's target resolution).

    Returns:
        The real :class:`PipelineComponents`.

    Raises:
        SystemExit: If the directions artifact is missing.
    """
    from .directions import (
        DirectionsBundle,
        load_model_and_pipeline,
        resolve_directions_paths,
    )

    npz_path, json_path = resolve_directions_paths(args.directions, args.model)
    if not npz_path.exists():
        raise SystemExit(
            f"directions artifact {npz_path} not found; run "
            f"`python -m act_steering.directions --model {args.model}` first"
        )
    model, tokenizer, _ = load_model_and_pipeline(args.model, args.device, args.dtype)
    bundle = DirectionsBundle.load(npz_path, json_path)
    return build_real_components(
        model,
        tokenizer,
        bundle,
        config,
        steering=SteeringConfig(
            alpha=args.alpha,
            token_pos=None if args.token_pos == "none" else args.token_pos,
        ),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        monitor_config=MonitorConfig(threshold=args.monitor_threshold),
        coeffs=coeffs,
        dictionary=dictionary,
        prompt_spec=PromptSpec(style=args.prompt_style, variant=args.system_variant),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a scripted conversation through the control loop and log it.

    Mock mode (``--mock``/``--quick``) runs deterministic CPU components.
    Real mode (without ``--mock``) loads the model plus the directions
    artifact and wires the EPA reader, deflection steerer and monitor.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: When no script is available or the directions
            artifact is missing in real mode.
    """
    parser = argparse.ArgumentParser(
        description="ACT deflection-steering control loop over a scripted conversation."
    )
    parser.add_argument(
        "--script", type=Path, default=None, help="Conversation script JSON (omit with --quick)"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output trace log JSON path")
    parser.add_argument(
        "--mock", action="store_true", help="Run with deterministic CPU mock components"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: built-in two-message script with mock components",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Recorded in the log (mock is deterministic)",
    )
    parser.add_argument("--user", default=None, help="User identity term (overrides script)")
    parser.add_argument("--agent", default=None, help="Agent identity term (overrides script)")
    parser.add_argument("--k", type=int, default=4, help="Candidates per turn")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (real mode)")
    parser.add_argument(
        "--directions",
        type=Path,
        default=None,
        help="Directions .npz (default: artifacts/directions_<model>.npz)",
    )
    parser.add_argument("--device", default="cuda", help="torch device (real mode)")
    parser.add_argument("--dtype", default="float16", help="torch dtype (real mode)")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Steering base strength (real mode)"
    )
    parser.add_argument(
        "--token-pos",
        default="end",
        choices=("end", "start", "none"),
        help=(
            "Where the steering vector applies (real mode): 'end' = last "
            "prompt position plus every new token (CAA style, default); "
            "'none' = every position (AAAI-27 style)"
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (real mode)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Generation cap per candidate (real mode)"
    )
    parser.add_argument(
        "--prompt-style",
        default="chat",
        choices=("chat", "raw"),
        help=(
            "Generation prompt format (real mode): 'chat' renders the "
            "conversation through the tokenizer chat template with a "
            "system role (the correct framing for instruct models; "
            "default) or 'raw' uses the flat continuation template "
            "(AAAI-27 reproduction)"
        ),
    )
    parser.add_argument(
        "--system-variant",
        default="role",
        choices=("role", "act_descriptors", "anti_sycophancy"),
        help=(
            "System prompt variant for the chat style (real mode): 'role' "
            "(B0), 'act_descriptors' (B1: role plus qualitative ACT "
            "target descriptors) or 'anti_sycophancy' (B2)"
        ),
    )
    parser.add_argument(
        "--monitor-positions",
        type=int,
        default=0,
        help="Response tail positions scored by the monitor (0 disables; real mode)",
    )
    parser.add_argument(
        "--monitor-threshold",
        type=float,
        default=1.0,
        help=(
            "Per-token deflection exceedance threshold (real mode); "
            "recalibrate on an unsteered trace so baseline firing is ~5-10%"
        ),
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Enable the re-identification repair policy (real mode)",
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.mock = True
    script: dict[str, object] | None = QUICK_SCRIPT if args.quick else None
    if not args.quick and args.script is not None:
        script = json.loads(args.script.read_text(encoding="utf-8"))
    if script is None:
        parser.error("a --script file is required unless --quick is given")
    messages = script.get("messages", [])
    if not messages:
        parser.error("script must contain a non-empty 'messages' list")
    user = args.user or str(script.get("user", "student"))
    agent = args.agent or str(script.get("agent", "teacher"))

    data = DataPaths()
    coeffs = ACTCoefficients.load(data.coefficients_file)
    dictionary = ACTDictionary.load(data.data_dir)
    state = ConversationState.from_ids(dictionary, user, agent)

    if args.mock:
        components = build_mock_components(k=args.k)
        config = PipelineConfig(k=args.k, mode="mock")
    else:
        config = PipelineConfig(
            mode="static",
            k=args.k,
            monitor_positions=args.monitor_positions,
            repair="reidentify" if args.repair else "none",
        )
        components = _build_real_cli_components(args, config, coeffs, dictionary)
    pipeline = ACTSteeringPipeline(coeffs, state, components, config)
    log = pipeline.run_conversation(messages)
    log["seed"] = args.seed
    log["quick"] = bool(args.quick)
    if not args.mock:
        log["model"] = args.model
    pipeline.save_log(log, args.out)
    print(f"wrote {len(log['turns'])} turns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
