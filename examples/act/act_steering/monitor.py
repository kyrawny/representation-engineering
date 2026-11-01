"""Per-token deflection monitoring (S7).

Mode 1 of the design synthesis (``act-deflection-steering-via-repe``): a
*deflection trace* is computed for a generated response by scoring every
output token — the honesty-frontier pattern from the upstream repo — and
feeding the trace into the re-identification repair trigger.

Score definition: every generated token's hidden state is projected onto
the signed deflection-contrast direction stored in the directions bundle
(S6). The bundle directions already point toward *low* deflection with
the reader sign applied, so a *positive* projection means the token is
moving in the low-deflection direction and a *negative* projection means
it is moving toward high deflection. Per-token score::

    score_j = mean_L( h_j,L . d_L )        (d_L points toward low deflection)

Trace processing (the honesty-frontier pattern):

1. **Median outlier clipping**: scores more than ``clip_sigma`` standard
   deviations from the *median* are replaced by the median (a single
   degenerate token must not trigger a repair).
2. **Token trigger**: a token *exceeds* when its clipped score is strictly
   below ``-threshold`` (drifting toward high deflection by at least the
   threshold); a response *fires* when at least one token exceeds.
3. **Sustained (re-identification) trigger**: the pipeline collects the
   per-turn fired flags; a repair is requested only after
   ``trigger_window`` consecutive fired turns, so one noisy response does
   not trigger a rewrite.

Positioning (per the WS2 measurement stack): the trace is a
*diagnostic* — a reader score on the steered model (M-S family) — used
for repair triggers and for logging stress accumulation across turns. It
is never the headline evaluation metric.

Cost: one full forward pass per scored position (the same pattern as the
upstream honesty notebook), so monitor short responses only.

The pure-logic half (clipping, trace building, trigger objects) is
torch-free and CPU-testable; :func:`make_hidden_provider` is the thin
torch glue for real models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import MonitorConfig
from .directions import DEFLECTION_DIRECTION, DirectionsBundle

__all__ = [
    "TokenTrace",
    "ReidentificationTrigger",
    "clip_outliers",
    "token_scores",
    "build_token_trace",
    "sustained_trigger",
    "resolve_reader_layers",
    "DeflectionMonitor",
    "make_hidden_provider",
]

#: A hidden-state provider maps ``(text, rep_token)`` to
#: ``{decoder_layer: (hidden_size,) float array}`` for the single
#: reading position (HF-style negative indexing, clamped).
HiddenProvider = Callable[[str, int], dict[int, np.ndarray]]


# ---------------------------------------------------------------------------
# Pure core (torch-free)
# ---------------------------------------------------------------------------


def clip_outliers(scores: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """Replace extreme scores with the median (honesty-frontier pattern).

    A score is an outlier when it lies more than ``clip_sigma`` standard
    deviations (population std of the input) away from the *median*.
    Non-positive ``clip_sigma`` disables clipping.

    Args:
        scores: 1-D array of per-token scores.
        clip_sigma: Outlier distance in standard deviations.

    Returns:
        Clipped copy (same shape, float64); input is not modified.
    """
    out = np.asarray(scores, dtype=np.float64).copy()
    if out.size == 0 or clip_sigma <= 0.0:
        return out
    med = float(np.median(out))
    sd = float(out.std())
    if sd > 0.0:
        mask = (out > med + clip_sigma * sd) | (out < med - clip_sigma * sd)
        out[mask] = med
    return out


def token_scores(
    provider: HiddenProvider,
    bundle: DirectionsBundle,
    layers: list[int],
    text: str,
    n_positions: int,
) -> np.ndarray:
    """Score the last ``n_positions`` tokens of ``text``.

    Position ``j`` (0 = first scored token, ``n_positions - 1`` = last
    token) is read at ``rep_token = -(n_positions - j)`` and scored as the
    mean projection over ``layers`` onto the deflection-contrast
    direction.

    Args:
        provider: Hidden-state provider (see :data:`HiddenProvider`).
        bundle: Directions artifact bundle.
        layers: Decoder layers to average over.
        text: The full text containing the scored tail (e.g. context
            plus the generated response).
        n_positions: Number of tail tokens to score (usually the
            generated response length).

    Returns:
        ``(n_positions,)`` float64 array; positive scores point toward
        low deflection.

    Raises:
        ValueError: If ``n_positions`` is negative or a layer is missing
            from the bundle.
    """
    if n_positions < 0:
        raise ValueError(f"n_positions must be >= 0, got {n_positions}")
    for layer in layers:
        if layer not in bundle.hidden_layers:
            raise ValueError(
                f"layer {layer} not in the directions bundle "
                f"(layers: {bundle.hidden_layers})"
            )
    scores = np.empty(n_positions, dtype=np.float64)
    for j in range(n_positions):
        rep_token = -(n_positions - j)
        hidden = provider(text, rep_token)
        projections = np.array(
            [hidden[layer] @ bundle.direction(DEFLECTION_DIRECTION, layer) for layer in layers]
        )
        scores[j] = float(projections.mean())
    return scores


@dataclass(frozen=True)
class TokenTrace:
    """Processed per-token deflection trace for one generated response.

    Attributes:
        scores: Raw per-token projections (positive = low-deflection side).
        clipped: Scores after median outlier clipping.
        threshold: Exceedance threshold (positive magnitude).
        clip_sigma: Clip distance used (in standard deviations).
        n_tokens: Number of scored tokens.
        n_above: Tokens whose clipped score is below ``-threshold``.
        fired: Whether at least one token exceeds.
        mean, median, std: Statistics of the clipped scores.
        peak: Maximum clipped score.
        peak_index: Index of the maximum clipped score (-1 if empty).
    """

    scores: tuple[float, ...]
    clipped: tuple[float, ...]
    threshold: float
    clip_sigma: float
    n_above: int
    fired: bool
    mean: float
    median: float
    std: float
    peak: float
    peak_index: int

    @property
    def n_tokens(self) -> int:
        """Number of scored tokens."""
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for trace logs."""
        return {
            "threshold": self.threshold,
            "clip_sigma": self.clip_sigma,
            "n_tokens": self.n_tokens,
            "n_above": self.n_above,
            "fired": self.fired,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "peak": self.peak,
            "peak_index": self.peak_index,
            "scores": list(self.scores),
            "clipped": list(self.clipped),
        }


def build_token_trace(
    scores: np.ndarray, threshold: float, clip_sigma: float = 5.0
) -> TokenTrace:
    """Build a :class:`TokenTrace` from raw per-token scores.

    Args:
        scores: Raw per-token projections.
        threshold: Positive exceedance threshold; a token exceeds when its
            clipped score is strictly below ``-threshold``.
        clip_sigma: Median-outlier clip distance.

    Raises:
        ValueError: If the threshold is not positive.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    raw = np.atleast_1d(np.asarray(scores, dtype=np.float64))
    clipped = clip_outliers(raw, clip_sigma)
    n_above = int(np.sum(clipped < -threshold))
    return TokenTrace(
        scores=tuple(float(x) for x in raw),
        clipped=tuple(float(x) for x in clipped),
        threshold=float(threshold),
        clip_sigma=float(clip_sigma),
        n_above=n_above,
        fired=n_above > 0,
        mean=float(clipped.mean()) if clipped.size else 0.0,
        median=float(np.median(clipped)) if clipped.size else 0.0,
        std=float(clipped.std()) if clipped.size else 0.0,
        peak=float(clipped.max()) if clipped.size else 0.0,
        peak_index=int(np.argmax(clipped)) if clipped.size else -1,
    )


@dataclass(frozen=True)
class ReidentificationTrigger:
    """Sustained re-identification repair trigger (turn-level).

    The pipeline appends each turn's :attr:`TokenTrace.fired` flag to a
    history and re-evaluates this trigger after every turn.

    Attributes:
        fired: Whether ``window`` consecutive fired turns were reached.
        window: Required consecutive fired turns (``trigger_window``).
        consecutive: Fired-turn run length at the tail of the history.
        turns_recorded: Total turns in the history.
        turn_fired: The full fired-flag history (oldest first).
        reason: Human-readable decision rationale for trace logs.
    """

    fired: bool
    window: int
    consecutive: int
    turns_recorded: int
    turn_fired: tuple[bool, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for trace logs."""
        return {
            "fired": self.fired,
            "window": self.window,
            "consecutive": self.consecutive,
            "turns_recorded": self.turns_recorded,
            "turn_fired": list(self.turn_fired),
            "reason": self.reason,
        }


def sustained_trigger(turn_fired: Sequence[bool], window: int) -> ReidentificationTrigger:
    """Evaluate the sustained re-identification trigger.

    The trigger fires when the *last* ``window`` turns in the history all
    fired. A shorter history can never fire.

    Args:
        turn_fired: Per-turn fired flags, oldest first.
        window: Required consecutive fired turns.

    Returns:
        The trigger object with its decision rationale.

    Raises:
        ValueError: If ``window`` is below 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    flags = tuple(bool(f) for f in turn_fired)
    consecutive = 0
    for flag in reversed(flags):
        if not flag:
            break
        consecutive += 1
    if len(flags) < window:
        reason = f"only {len(flags)} turn(s) recorded; need {window}"
    elif consecutive >= window:
        reason = f"{consecutive} consecutive fired turns (>= {window})"
    else:
        reason = f"tail run of {consecutive} fired turn(s) < {window}"
    return ReidentificationTrigger(
        fired=len(flags) >= window and consecutive >= window,
        window=int(window),
        consecutive=consecutive,
        turns_recorded=len(flags),
        turn_fired=flags,
        reason=reason,
    )


def resolve_reader_layers(bundle: DirectionsBundle, config: MonitorConfig) -> list[int]:
    """Return the monitor's reader layers from config or the bundle.

    An empty ``config.reader_layers`` auto-selects the bundle's last
    layer.

    Raises:
        ValueError: If a configured layer is not carried by the bundle.
    """
    layers = list(config.reader_layers)
    if not layers:
        layers = [max(bundle.hidden_layers)]
    for layer in layers:
        if layer not in bundle.hidden_layers:
            raise ValueError(
                f"layer {layer} not in the directions bundle "
                f"(layers: {bundle.hidden_layers})"
            )
    return sorted(set(layers))


# ---------------------------------------------------------------------------
# Model-bound glue (lazy torch import)
# ---------------------------------------------------------------------------


class DeflectionMonitor:
    """Per-token deflection monitor over a hidden-state provider.

    Args:
        hidden_provider: Maps ``(text, rep_token)`` to per-layer hidden
            states; use :func:`make_hidden_provider` for a real model.
        bundle: The directions artifact bundle (S6).
        config: Monitor settings.
    """

    def __init__(
        self,
        hidden_provider: HiddenProvider,
        bundle: DirectionsBundle,
        config: MonitorConfig,
    ) -> None:
        self.provider = hidden_provider
        self.bundle = bundle
        self.config = config
        self.layers = resolve_reader_layers(bundle, config)

    def trace(self, text: str, n_positions: int) -> TokenTrace:
        """Score the last ``n_positions`` tokens of ``text`` and process.

        Args:
            text: Full text with the generated response at the tail.
            n_positions: Number of tail tokens (the response length).
        """
        scores = token_scores(
            self.provider, self.bundle, self.layers, text, n_positions
        )
        return build_token_trace(scores, self.config.threshold, self.config.clip_sigma)


def make_hidden_provider(
    model,
    tokenizer,
    decoder_layers: list[int],
    max_length: int = 512,
    read_format: str = "raw",
) -> HiddenProvider:
    """Build a hidden-state provider from a HuggingFace causal LM.

    One forward pass per call (no KV cache); the reading position is
    clamped into the tokenised input.

    Args:
        model: HuggingFace causal LM (``output_hidden_states`` support).
        tokenizer: Its tokenizer.
        decoder_layers: 0-indexed decoder layer indices to read.
        max_length: Tokenisation cap for forward passes.
        read_format: ``"raw"`` (default) or ``"chat"`` (chat template,
            text in the assistant position — must match the format the
            directions were fit in; see
            :func:`act_steering.readers.format_for_reading`).

    Returns:
        A :data:`HiddenProvider` callable.
    """
    import torch

    from .readers import format_for_reading

    device = getattr(model, "device", None)

    def provider(text: str, rep_token: int) -> dict[int, np.ndarray]:
        enc = tokenizer(
            format_for_reading(text, tokenizer, read_format),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        n_tokens = int(enc["input_ids"].shape[1])
        idx = rep_token if rep_token >= 0 else n_tokens + rep_token
        idx = max(0, min(idx, n_tokens - 1))
        return {
            layer: out.hidden_states[layer + 1][0, idx].detach().cpu().numpy()
            for layer in decoder_layers
        }

    return provider
