"""
EPA Steering Module.

Provides representation-engineering-based EPA steering for text generation.
Supports two usage patterns:

1. **Simple steering** — use ``steer_generation()`` directly with target EPA
   values and uniform coefficients.
2. **Calibrated steering** — use ``EPASteerer`` with per-dimension configs
   derived from reading tuning results, allowing per-layer weights, signs,
   and tuned coefficients.

Typical usage::

    steerer = EPASteerer.from_reader(reader, rep_readers, model, tokenizer)
    response = steerer.generate(prompt, target_epa={"e": 1.5, "p": 0.0, "a": -0.5})
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import math

import numpy as np
import torch

from .act_core import EPA


# =========================================================================
# Steering magnitude limiter
# =========================================================================

def _sigmoid_clamp(x: float, limit: float = 1.0) -> float:
    """Smoothly clamp a scalar to [-limit, +limit] using tanh.

    Maps the raw steering magnitude through ``limit * tanh(x / limit)``
    so that small values pass through approximately unchanged while
    large values asymptote smoothly at ±limit.  This prevents extreme
    target EPA values from producing oversized perturbations that cause
    text degeneration.

    Args:
        x: Raw steering magnitude (coeff * target_value).
        limit: Maximum absolute value of the output (default 1.0).

    Returns:
        Clamped magnitude in [-limit, +limit].
    """
    if limit <= 0:
        return 0.0
    return limit * math.tanh(x / limit)


from .prompt_formatting import DIMENSION_NAMES, DIM_KEY


# =========================================================================
# Low-level activation helpers
# =========================================================================

def make_epa_activations(
    rep_readers: Dict[str, Any],
    layers: List[int],
    e_coeff: float = 0.0,
    p_coeff: float = 0.0,
    a_coeff: float = 0.0,
    device=None,
    dtype=None,
    normalize: bool = True,
    clamp_limit: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """
    Create combined EPA activation dictionary for the control pipeline.

    Direction vectors are L2-normalized by default so that the coefficient
    directly controls the perturbation magnitude (a coefficient of 1.0
    adds a unit vector).

    Args:
        rep_readers: Dict mapping dimension name to ``RepReader``.
        layers: Layer indices to create activations for.
        e_coeff: Evaluation steering coefficient.
        p_coeff: Potency steering coefficient.
        a_coeff: Activity steering coefficient.
        device: Target device.
        clamp_limit: Maximum absolute steering magnitude per dimension
            (default 1.0).  Set to 0 or ``float('inf')`` to disable.
        dtype: Target dtype.
        normalize: If True, L2-normalize direction vectors.

    Returns:
        Dict mapping layer index to activation tensor.
    """
    # Determine hidden dimension
    hidden_dim = None
    for reader in rep_readers.values():
        for _layer, direction in reader.directions.items():
            hidden_dim = direction.shape[-1]
            break
        if hidden_dim is not None:
            break

    # Pre-populate with zeros
    activations: Dict[int, torch.Tensor] = {}
    if hidden_dim is not None:
        for layer in layers:
            t = torch.zeros(1, hidden_dim)
            if dtype is not None:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            activations[layer] = t

    for dim, coeff in [("evaluation", e_coeff), ("potency", p_coeff), ("activity", a_coeff)]:
        if coeff == 0.0 or dim not in rep_readers:
            continue

        reader = rep_readers[dim]
        for layer in layers:
            if layer not in reader.directions:
                continue

            sign = reader.direction_signs.get(layer, 1)
            if hasattr(sign, "item"):
                sign = sign.item()
            sign = float(sign)

            direction = torch.tensor(reader.directions[layer])

            if normalize:
                norm = direction.norm(p=2)
                if norm > 0:
                    direction = direction / norm

            if dtype is not None:
                direction = direction.to(dtype)
            if device is not None:
                direction = direction.to(device)

            magnitude = _sigmoid_clamp(coeff * sign, limit=clamp_limit) if clamp_limit > 0 and clamp_limit != float('inf') else coeff * sign
            activation = magnitude * direction

            if layer in activations:
                activations[layer] = activations[layer] + activation
            else:
                activations[layer] = activation

    return activations


def steer_generation(
    model,
    tokenizer,
    rep_readers: Dict[str, Any],
    layers: List[int],
    prompt: str,
    target_epa: Tuple[float, float, float],
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rep_control_pipeline=None,
    normalize: bool = True,
    do_sample: bool = True,
    **generation_kwargs,
) -> str:
    """
    Generate text with EPA steering (simple interface).

    Uses ``RepControlPipeline`` from the ``repe`` library, applying
    L2-normalized direction vectors by default.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        rep_readers: Dict mapping dimension name to ``RepReader``.
        layers: Layer indices to steer.
        prompt: Formatted input prompt.
        target_epa: Target (E, P, A) values.
        steering_coefficient: Global multiplier for steering strength.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        rep_control_pipeline: Optional pre-configured pipeline.
        normalize: L2-normalize directions (recommended).
        do_sample: Whether to use sampling.
        **generation_kwargs: Extra generation arguments.

    Returns:
        Generated response text (prompt stripped).
    """
    from transformers import pipeline as hf_pipeline

    e_target, p_target, a_target = target_epa

    activations = make_epa_activations(
        rep_readers=rep_readers,
        layers=layers,
        e_coeff=e_target * steering_coefficient,
        p_coeff=p_target * steering_coefficient,
        a_coeff=a_target * steering_coefficient,
        device=model.device,
        dtype=model.dtype,
        normalize=normalize,
    )

    if rep_control_pipeline is None:
        rep_control_pipeline = hf_pipeline(
            "rep-control",
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            control_method="reading_vec",
        )

    outputs = rep_control_pipeline(
        prompt,
        activations=activations,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        **generation_kwargs,
    )

    generated_text = outputs[0]["generated_text"]
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


# =========================================================================
# Calibrated steerer class
# =========================================================================

class EPASteerer:
    """
    EPA steering with per-dimension layer configs and tuned coefficients.

    Derives its configuration from an ``EPAReader`` so that the same layers
    and signs used for reading are also used for steering.
    """

    def __init__(
        self,
        model,
        tokenizer,
        rep_readers: Dict[str, Any],
        steering_configs: Dict[str, Dict],
        all_layers: Optional[List[int]] = None,
        clamp_limit: float = 1.0,
    ):
        """
        Args:
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            rep_readers: Dict of ``RepReader`` objects.
            steering_configs: Per-dimension config dict with keys:
                ``'layers'``, ``'signs'``, ``'base_coeff'``.
            all_layers: All available layers for the control pipeline.
                If None, derived from the first ``RepReader``.
            clamp_limit: Maximum absolute steering magnitude per
                dimension after multiplying all coefficients (default
                1.0).  Uses a smooth tanh clamp.  Set to
                ``float('inf')`` to disable.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.rep_readers = rep_readers
        self.steering_configs = steering_configs
        self.clamp_limit = clamp_limit

        if all_layers is None:
            first_reader = next(iter(rep_readers.values()))
            all_layers = sorted(first_reader.directions.keys())
        self.all_layers = all_layers

        # Lazy-initialised control pipeline
        self._control_pipeline = None

    @property
    def control_pipeline(self):
        """Lazy-create the ``rep-control`` pipeline."""
        if self._control_pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._control_pipeline = hf_pipeline(
                "rep-control",
                model=self.model,
                tokenizer=self.tokenizer,
                layers=self.all_layers,
                control_method="reading_vec",
            )
        return self._control_pipeline

    # -----------------------------------------------------------------
    # Build activations
    # -----------------------------------------------------------------

    def build_activations(
        self,
        dim: str,
        target_value: float,
        coeff_override: Optional[Dict[int, float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Build activations for steering a single dimension.

        The perturbation magnitude is ``coeff * |target_value|`` applied
        along the (sign-corrected, L2-normalised) direction vector.  This
        makes the steering strength proportional to how far the target is
        from the EPA origin.

        Args:
            dim: One of ``'evaluation'``, ``'potency'``, ``'activity'``.
            target_value: Target EPA value for this dimension.
            coeff_override: Optional per-layer coefficient overrides.

        Returns:
            Dict mapping every layer to an activation tensor (zero for
            non-steering layers).
        """
        config = self.steering_configs[dim]
        reader = self.rep_readers[dim]
        activations: Dict[int, torch.Tensor] = {}

        for layer in self.all_layers:
            if layer in config["layers"]:
                if coeff_override and layer in coeff_override:
                    coeff = coeff_override[layer]
                else:
                    coeff = config["base_coeff"]

                # Layer sign from reader (aligns direction to positive=high EPA)
                dir_sign = float(reader.direction_signs.get(layer, 1))
                if hasattr(reader.direction_signs.get(layer, 1), "item"):
                    dir_sign = float(reader.direction_signs[layer].item())

                direction = torch.tensor(reader.directions[layer], dtype=torch.float16)

                # L2-normalise so coeff directly controls magnitude
                norm = direction.norm(p=2)
                if norm > 0:
                    direction = direction / norm

                # Perturbation = clamp(coeff * target_value) * sign * normalised_dir
                # target_value carries both sign and magnitude;
                # sigmoid clamp prevents extreme values from causing degeneration
                raw_magnitude = coeff * target_value * dir_sign
                clamped = _sigmoid_clamp(raw_magnitude, limit=self.clamp_limit)
                activations[layer] = clamped * direction
            else:
                direction = torch.tensor(reader.directions[layer], dtype=torch.float16)
                activations[layer] = torch.zeros_like(direction)

        return activations

    def build_combined_activations(
        self,
        target_epa: Dict[str, float],
        coeff_overrides: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Build combined activations for steering all three dimensions at once.

        Args:
            target_epa: Dict with ``'evaluation'``, ``'potency'``,
                ``'activity'`` targets (or subset).
            coeff_overrides: Optional per-dimension per-layer overrides.

        Returns:
            Dict mapping every layer to a combined activation tensor.
        """
        # Initialise with zeros
        first_reader = next(iter(self.rep_readers.values()))
        sample_dir = next(iter(first_reader.directions.values()))
        hidden_dim = sample_dir.shape[-1]

        combined: Dict[int, torch.Tensor] = {}
        for layer in self.all_layers:
            combined[layer] = torch.zeros(1, hidden_dim, dtype=torch.float16)

        for dim in DIMENSION_NAMES:
            if dim not in target_epa or target_epa[dim] == 0.0:
                continue
            override = (coeff_overrides or {}).get(dim)
            act = self.build_activations(dim, target_epa[dim], coeff_override=override)
            for layer in self.all_layers:
                combined[layer] = combined[layer] + act[layer]

        return combined

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        target_epa: Union[Dict[str, float], Tuple[float, float, float]],
        max_new_tokens: int = 128,
        do_sample: bool = False,
        repetition_penalty: float = 1.2,
        coeff_overrides: Optional[Dict[str, Dict[int, float]]] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate text steered toward the target EPA values.

        Args:
            prompt: Formatted input prompt.
            target_epa: Target EPA — either ``{'evaluation': ..., 'potency': ..., 'activity': ...}``
                or ``(e, p, a)`` tuple.
            max_new_tokens: Maximum tokens.
            do_sample: Whether to sample.
            repetition_penalty: Repetition penalty.
            coeff_overrides: Per-dimension per-layer coefficient overrides.
            **generation_kwargs: Extra generation parameters.

        Returns:
            Generated response text (prompt stripped).
        """
        if isinstance(target_epa, (list, tuple)):
            target_epa = {
                "evaluation": target_epa[0],
                "potency": target_epa[1],
                "activity": target_epa[2],
            }

        activations = self.build_combined_activations(target_epa, coeff_overrides)

        outputs = self.control_pipeline(
            prompt,
            activations=activations,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            **generation_kwargs,
        )

        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------

    @classmethod
    def from_reader(
        cls,
        reader: "EPAReader",
        rep_readers: Dict[str, Any],
        model,
        tokenizer,
        base_coeff: float = 2.0,
        phase1: Optional[Dict] = None,
    ) -> "EPASteerer":
        """
        Construct an ``EPASteerer`` from an ``EPAReader`` config.

        Uses the reader's selected layers and signs as the steering config.

        Args:
            reader: Configured ``EPAReader`` instance.
            rep_readers: Dict of ``RepReader`` objects.
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            base_coeff: Default steering coefficient per layer.
            phase1: Phase-1 correlations (used to derive signs if not in
                the reader config).  Optional.

        Returns:
            Configured ``EPASteerer``.
        """
        # Avoid circular import at module level
        from .epa_reader import EPAReader as _EPAReader

        steering_configs: Dict[str, Dict] = {}
        for dim in DIMENSION_NAMES:
            dim_cfg = reader.config.dimensions[dim]
            layers = sorted(dim_cfg.selected_layers.keys())
            signs = dict(dim_cfg.layer_signs)

            # Fallback: derive signs from phase1 if available
            if not signs and phase1 and dim in phase1:
                for layer in layers:
                    rho = phase1[dim].get(layer, {}).get("rho", 0)
                    signs[layer] = 1.0 if rho > 0 else -1.0

            steering_configs[dim] = {
                "layers": layers,
                "signs": signs,
                "base_coeff": base_coeff,
            }

        return cls(
            model=model,
            tokenizer=tokenizer,
            rep_readers=rep_readers,
            steering_configs=steering_configs,
        )
