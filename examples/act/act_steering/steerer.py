"""Steering execution for deflection steering (S7).

Applies per-turn steering to a generation model along the
deflection-contrast direction (primary axis) or, as a baseline condition,
along the EPA dimension directions scaled to the ACT optimal behaviour
b* (the AAAI-27 steering style).

Two actuation modes (see the design synthesis
``act-deflection-steering-via-repe`` in the wiki):

* **Static mode** (default, :class:`DeflectionSteerer`): the direction is
  loaded once from the directions artifact (S6). Per turn the magnitude
  is a *proportional gain* on the event-1 deflection U_pre::

      alpha_t = clip(alpha * kappa * U_pre / u_ref, alpha_min, alpha_max)
      c_t     = alpha_t * d

  so benign turns run effectively unsteered (no quality tax) while
  deflection-generating turns get a harder push, and the gain can never
  exceed ``alpha_max`` (which the dose-response sweep keeps below the
  degeneration threshold). This closes the loop the AAAI-27 paper left
  open: the objective (deflection) is in the control law.

* **In-loop contrast mode** (:class:`InLoopContrastSteerer`): a *new*
  vector is computed at every decode step from the pos/neg contrast
  continuations of the *current* context (upstream ``ContrastVec*``
  path). The situation stays situation-dependent even after topic drift,
  at ~3x the per-step compute and with ``use_cache=False``.

The AAAI-27 baseline (EPA-target steering) is kept as a condition, not
the default: ``epa_target_offsets`` computes ``sum_d c_d * t_d * dhat_d``
with fixed coefficients (0.5, 0.2, 0.2) for (E, P, A) — the single
coefficient definition chosen per revision plan WS1.3 (the contradictory
sigmoid clamp is dropped).

Application mechanics (verified against upstream ``WrappedBlock``):
the offset is a 1-D ``(hidden_size,)`` vector set through
``wrapped_model.set_controller(layers, {layer: vec}, "decoder_block",
token_pos=..., masks=...)``. ``WrappedBlock.set_controller`` squeezes the
activation, so only 1-D vectors (or batch >= 2) work through the
pipeline path. With ``token_pos="end"`` the vector hits the *last prompt
position* in prefill and then the *new token at every decode step*, i.e.
the whole response is conditioned on the steering. Because a 1-D
controller makes the upstream ``len_token = controller.shape[1]`` equal
1, the ``"end"``/``"start"`` branches need the explicit all-ones
``(1, 1, 1)`` mask from :func:`controller_masks` (the ``position_ids``
mask the upstream code would otherwise use is ``(1, seq, 1)`` and cannot
broadcast against the ``(1, 1, hidden)`` target). ``token_pos=None``
instead steers *every* position (prefill and decode), which is what the
AAAI-27 implementation used.

The pure-logic half (gain, offsets, validation) is torch-free and
CPU-testable; the model-bound classes lazy-import ``torch`` / ``repe``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .config import SteeringConfig
from .directions import DEFLECTION_DIRECTION, EPA_DIMENSIONS, DirectionsBundle

__all__ = [
    "STEERING_MODES",
    "EPA_TARGET_COEFFICIENTS",
    "proportional_gain",
    "validate_steering_config",
    "resolve_layers",
    "resolve_model_layers",
    "deflection_offsets",
    "epa_target_offsets",
    "controller_masks",
    "DeflectionSteerer",
    "InLoopContrastSteerer",
]

#: Actuation modes understood by the steerer.
STEERING_MODES: tuple[str, ...] = ("static", "reextract", "inloop")

#: Fixed per-dimension steering coefficients for (E, P, A) in the
#: AAAI-27 baseline condition (revision plan WS1.3: one definition,
#: no sigmoid clamp).
EPA_TARGET_COEFFICIENTS: tuple[float, ...] = (0.5, 0.2, 0.2)


def proportional_gain(u_pre: float, config: SteeringConfig) -> float:
    """Return the per-turn steering gain ``alpha_t`` for a deflection.

    ``alpha_t = clip(alpha * kappa * U_pre / u_ref, alpha_min,
    alpha_max)``; at ``U_pre == u_ref`` the gain equals the base
    ``alpha`` (with ``kappa = 1``).

    Args:
        u_pre: Event-1 deflection of the current user message.
        config: Actuation settings.

    Returns:
        The clamped per-turn gain (>= 0).

    Raises:
        ValueError: If ``u_pre`` is negative or the config is invalid.
    """
    if u_pre < 0.0:
        raise ValueError(f"u_pre must be non-negative, got {u_pre}")
    validate_steering_config(config)
    raw = config.alpha * config.kappa * (u_pre / config.u_ref)
    return float(np.clip(raw, config.alpha_min, config.alpha_max))


def validate_steering_config(config: SteeringConfig) -> None:
    """Check a :class:`SteeringConfig` for internally consistent values.

    Raises:
        ValueError: If any field is out of range or mutually
            inconsistent.
    """
    if config.mode not in STEERING_MODES:
        raise ValueError(f"mode must be one of {STEERING_MODES}, got {config.mode!r}")
    if config.alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {config.alpha}")
    if config.alpha_min < 0.0:
        raise ValueError(f"alpha_min must be >= 0, got {config.alpha_min}")
    if config.alpha_min > config.alpha_max:
        raise ValueError(
            f"alpha_min ({config.alpha_min}) must be <= alpha_max ({config.alpha_max})"
        )
    if config.u_ref <= 0.0:
        raise ValueError(f"u_ref must be > 0, got {config.u_ref}")
    if config.kappa < 0.0:
        raise ValueError(f"kappa must be >= 0, got {config.kappa}")
    if config.token_pos not in (None, "end", "start") and not isinstance(
        config.token_pos, int
    ):
        raise ValueError(
            f"token_pos must be None, 'end', 'start', or an int, got {config.token_pos!r}"
        )
    if config.mode == "inloop" and config.contrast_tokens >= 0:
        raise ValueError(
            "contrast_tokens must be negative (tail positions), "
            f"got {config.contrast_tokens}"
        )


def resolve_layers(directions: DirectionsBundle, config: SteeringConfig) -> list[int]:
    """Return the steering layer indices from config or the bundle.

    An empty ``config.layers`` auto-selects the bundle's last layer.

    Raises:
        ValueError: If a configured layer is not carried by the bundle.
    """
    layers = list(config.layers)
    if not layers:
        layers = [max(directions.hidden_layers)]
    for layer in layers:
        if layer not in directions.hidden_layers:
            raise ValueError(
                f"layer {layer} not in the directions bundle "
                f"(layers: {directions.hidden_layers})"
            )
    return sorted(set(layers))


def resolve_model_layers(model, config: SteeringConfig) -> list[int]:
    """Return the steering layer indices for a model without a bundle.

    Used by the in-loop contrast steerer; an empty ``config.layers``
    auto-selects the model's last decoder layer.

    Raises:
        ValueError: If a configured layer is out of range for the model.
    """
    num_layers = int(model.config.num_hidden_layers)
    layers = list(config.layers)
    if not layers:
        layers = [num_layers - 1]
    for layer in layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer {layer} out of range [0, {num_layers})")
    return sorted(set(layers))


def deflection_offsets(
    directions: DirectionsBundle,
    config: SteeringConfig,
    u_pre: float,
) -> dict[int, np.ndarray]:
    """Compute the per-layer offsets for static deflection-contrast steering.

    Returns:
        ``{layer: (hidden_size,) float64 array}`` with the per-turn
        offset ``alpha_t * d`` at each resolved layer.
    """
    alpha_t = proportional_gain(u_pre, config)
    layers = resolve_layers(directions, config)
    return {
        layer: alpha_t * directions.direction(DEFLECTION_DIRECTION, layer)
        for layer in layers
    }


def epa_target_offsets(
    directions: DirectionsBundle,
    config: SteeringConfig,
    target: np.ndarray | tuple[float, float, float] | list[float],
) -> dict[int, np.ndarray]:
    """Compute the per-layer offsets for the AAAI-27 EPA-target baseline.

    The offset at each layer is ``sum_d c_d * t_d * dhat_d`` over the
    EPA dimensions in E, P, A order with fixed coefficients
    ``EPA_TARGET_COEFFICIENTS``.

    Args:
        directions: The directions artifact bundle.
        config: Actuation settings (layers only).
        target: The ACT optimal behaviour b* (E, P, A order).

    Returns:
        ``{layer: (hidden_size,) float64 array}``.
    """
    t = np.asarray(target, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError(f"target b* must have 3 components, got shape {t.shape}")
    layers = resolve_layers(directions, config)
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        acc = np.zeros(directions.hidden_size)
        for i, dim in enumerate(EPA_DIMENSIONS):
            acc += EPA_TARGET_COEFFICIENTS[i] * t[i] * directions.direction(
                f"epa.{dim}", layer
            )
        out[layer] = acc
    return out


def _to_tensor(vec: np.ndarray, model):
    """Convert an offset to a model-dtype/device torch tensor."""
    import torch

    dtype = getattr(model, "dtype", torch.float32)
    return torch.as_tensor(np.asarray(vec, dtype=np.float32), dtype=dtype, device=model.device)


def controller_masks(token_pos, device=None, dtype=None):
    """Return the explicit mask a 1-D controller needs at ``"end"``/``"start"``.

    The upstream ``WrappedBlock`` computes ``len_token =
    controller.shape[1]`` for the ``"end"``/``"start"`` token positions.
    A 1-D direction vector is squeezed to ``(hidden,)`` and reshaped to
    ``(1, 1, hidden)``, so ``len_token`` becomes 1 — but the padding mask
    the upstream code then derives from ``position_ids`` has shape
    ``(1, seq, 1)``, and ``controller * mask`` cannot be assigned into the
    ``(1, 1, hidden)`` target (RuntimeError: expanded size mismatch).
    Passing an explicit all-ones ``(1, 1, 1)`` mask applies the vector to
    exactly the last (or first) position in prefill and to the single new
    token at every decode step — CAA-style tail steering.

    Args:
        token_pos: The configured application position.
        device: Device for the mask (``None`` for CPU).
        dtype: Dtype for the mask (should match the model dtype).

    Returns:
        A ones tensor of shape ``(1, 1, 1)`` for ``"end"``/``"start"``,
        else ``None`` (the upstream ``position_ids``-derived mask
        applies, which is correct for whole-sequence steering).
    """
    if token_pos not in ("end", "start"):
        return None
    import torch

    return torch.ones((1, 1, 1), device=device, dtype=dtype)


class DeflectionSteerer:
    """Static-mode steerer: cached direction, per-turn proportional gain.

    Wraps the configured decoder blocks of the model with the upstream
    ``RepControlPipeline``. Per turn, :meth:`generate` computes the
    offsets from the event-1 deflection U_pre and applies them through
    ``set_controller`` with the configured ``token_pos``.

    Args:
        model: HuggingFace causal LM (mutated: decoder blocks wrapped).
        tokenizer: Its tokenizer.
        directions: The directions artifact bundle (S6).
        config: Actuation settings (``mode`` must be ``"static"``).

    Raises:
        ValueError: If the config is invalid or the mode is not static.
    """

    def __init__(
        self,
        model,
        tokenizer,
        directions: DirectionsBundle,
        config: SteeringConfig,
    ) -> None:
        validate_steering_config(config)
        if config.mode != "static":
            raise ValueError(
                "DeflectionSteerer implements mode='static' only; use "
                f"InLoopContrastSteerer for mode={config.mode!r}"
            )
        from repe.rep_control_pipeline import RepControlPipeline

        self.model = model
        self.tokenizer = tokenizer
        self.directions = directions
        self.config = config
        self.layers = resolve_layers(directions, config)
        self.pipeline = RepControlPipeline(model, tokenizer, layers=self.layers)

    def activations(
        self,
        u_pre: float,
        mode: str = "deflection",
        target: np.ndarray | tuple[float, float, float] | list[float] | None = None,
    ) -> dict[int, np.ndarray]:
        """Return the per-layer numpy offsets for one turn.

        Args:
            u_pre: Event-1 deflection (deflection mode).
            mode: ``"deflection"`` (primary) or ``"epa_target"`` (AAAI-27
                baseline).
            target: b* in E, P, A order (required for ``epa_target``).
        """
        if mode == "deflection":
            return deflection_offsets(self.directions, self.config, u_pre)
        if mode == "epa_target":
            if target is None:
                raise ValueError("epa_target mode requires the b* target")
            return epa_target_offsets(self.directions, self.config, target)
        raise ValueError(f"unknown steering mode {mode!r}")

    def generate_with_offsets(
        self,
        prompt: str,
        offsets: Mapping[int, np.ndarray],
        **gen_kwargs,
    ):
        """Generate with explicit per-layer offsets (any condition).

        The offsets are applied at the wrapped decoder blocks for the
        duration of the generation (the pipeline resets the controllers
        afterwards).

        Args:
            prompt: The generation prompt.
            offsets: ``{decoder_layer: (hidden_size,) offset vector}``;
                keys must be among the steerer's layers.
        """
        if not offsets:
            raise ValueError("generate_with_offsets requires non-empty offsets")
        unknown = set(offsets) - set(self.layers)
        if unknown:
            raise ValueError(f"offset layers {sorted(unknown)} not in {self.layers}")
        tensors = {
            layer: _to_tensor(vec, self.model) for layer, vec in offsets.items()
        }
        self.pipeline.wrapped_model.set_controller(
            list(self.layers),
            tensors,
            "decoder_block",
            token_pos=self.config.token_pos,
            masks=controller_masks(
                self.config.token_pos,
                device=self.model.device,
                dtype=tensors[self.layers[0]].dtype,
            ),
        )
        try:
            return self.pipeline(prompt, **gen_kwargs)
        finally:
            self.pipeline.wrapped_model.reset()

    def generate_plain(self, prompt: str, **gen_kwargs):
        """Generate with no steering (unsteered baseline).

        The pipeline is reset first, so no controller is applied.
        """
        self.pipeline.wrapped_model.reset()
        return self.pipeline(prompt, **gen_kwargs)

    def generate(
        self,
        prompt: str,
        u_pre: float,
        mode: str = "deflection",
        target: np.ndarray | tuple[float, float, float] | list[float] | None = None,
        **gen_kwargs,
    ):
        """Generate one steered response for the current turn.

        The per-turn offsets (deflection-contrast direction with
        proportional gain, or the AAAI-27 EPA-target offsets) are applied
        at the wrapped decoder blocks for the duration of the generation.
        """
        offsets = self.activations(u_pre, mode=mode, target=target)
        return self.generate_with_offsets(prompt, offsets, **gen_kwargs)


class InLoopContrastSteerer:
    """In-loop contrast steerer: a new vector at every decode step.

    The model must be an upstream ``ContrastVecLlamaForCausalLM`` or
    ``ContrastVecMistralForCausalLM`` instance. At each decode step the
    pos/neg contrast streams (which must be *continuations of the current
    context*, not standalone texts) are forwarded through the same layers
    and ``alpha * (h_pos - h_neg)`` over the last ``|contrast_tokens|``
    positions is added to the main stream and to both contrast streams
    (cascading). Requires ``use_cache=False``; ~3x per-step compute.

    Args:
        model: A ContrastVec causal-LM instance.
        tokenizer: Its tokenizer.
        config: Actuation settings (``mode`` must be ``"inloop"``).

    Raises:
        TypeError: If the model is not a ContrastVec instance.
        ValueError: If the config is invalid.
    """

    def __init__(self, model, tokenizer, config: SteeringConfig) -> None:
        validate_steering_config(config)
        if config.mode != "inloop":
            raise ValueError(
                f"InLoopContrastSteerer requires mode='inloop', got {config.mode!r}"
            )
        from repe.rep_control_contrast_vec import (
            ContrastVecLlamaForCausalLM,
            ContrastVecMistralForCausalLM,
        )

        if not isinstance(
            model, (ContrastVecLlamaForCausalLM, ContrastVecMistralForCausalLM)
        ):
            raise TypeError(
                "InLoopContrastSteerer requires a ContrastVecLlamaForCausalLM "
                "or ContrastVecMistralForCausalLM model; load the model with "
                "that class instead of AutoModelForCausalLM"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.layers = resolve_model_layers(model, config)

    def generate(
        self,
        prompt: str,
        pos_text: str,
        neg_text: str,
        alpha: float | None = None,
        max_length: int | None = None,
        **gen_kwargs,
    ):
        """Generate with per-step contrast steering.

        Args:
            prompt: The conversation context to continue.
            pos_text: Low-deflection continuation of ``prompt``.
            neg_text: High-deflection continuation of ``prompt``.
            alpha: Contrast strength; defaults to ``config.alpha``.
            max_length: Optional generation length cap.
        """
        if gen_kwargs.get("use_cache", False):
            raise ValueError("in-loop contrast generation requires use_cache=False")
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        pos = self.tokenizer(pos_text, return_tensors="pt", padding=True, truncation=True)
        neg = self.tokenizer(neg_text, return_tensors="pt", padding=True, truncation=True)
        device = self.model.device
        kwargs = dict(
            compute_contrast=True,
            alpha=alpha if alpha is not None else self.config.alpha,
            contrast_tokens=self.config.contrast_tokens,
            control_layer_ids=list(self.layers),
            use_cache=False,
            pos_input_ids=pos["input_ids"].to(device),
            pos_attention_mask=pos["attention_mask"].to(device),
            neg_input_ids=neg["input_ids"].to(device),
            neg_attention_mask=neg["attention_mask"].to(device),
            **gen_kwargs,
        )
        if max_length is not None:
            kwargs["max_length"] = max_length
        return self.model.generate(
            **{key: value for key, value in inputs.items() if key != "token_type_ids"},
            **kwargs,
        )
