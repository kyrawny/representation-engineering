"""Model-bound reading side shared by the ACT steering modules.

Two reader families over a HuggingFace causal LM plus a
:class:`~act_steering.directions.DirectionsBundle`:

- **EPA reader** — projects the reading position's hidden state onto the
  per-layer EPA direction set (averaged across layers) and returns an
  EPA-style score triple. Positive scores mark the positive side of each
  dimension. Used as the loop's message/behaviour reader (pipeline real
  mode), the dose-response probe's shift measurement, and the
  ``reader_validation`` model reader.
- **Reference scorer** — per-token cross-entropy (nats) under a
  reference model, the perplexity-style quality metric used by the
  guardrails and dose-response sweeps.

The projection math is torch-free (pure numpy) so the scoring core is
unit-testable on CPU; the model-bound builders import ``torch`` lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .act_core import EPA
from .directions import DEFAULT_MAX_LENGTH, EPA_DIMENSIONS

__all__ = [
    "EpaReader",
    "NEUTRAL_READ_CONTEXT",
    "ReferenceScorer",
    "available_reader_layers",
    "build_epa_reader",
    "build_reference_scorer",
    "epa_scores",
    "format_for_reading",
]

#: Neutral user context for chat-format EPA reading (AAAI-27 pattern):
#: the text to read sits in the assistant position after this user
#: message, with no system prompt. The directions must have been fit in
#: the same format for the reader to be valid.
NEUTRAL_READ_CONTEXT = "Pretend you're a person making a statement in a conversation."

#: Reading formats: ``"raw"`` (flat text; mock mode and legacy artifacts)
#: or ``"chat"`` (chat template, text in the assistant position).
ReadFormat = str


def format_for_reading(text: str, tokenizer: Any, read_format: str = "raw") -> str:
    """Format response text for EPA reading (shared by reader and monitor).

    Args:
        text: The response / message text to read.
        tokenizer: A tokenizer exposing ``apply_chat_template`` (required
            for the ``"chat"`` format).
        read_format: ``"raw"`` returns the text unchanged; ``"chat"``
            renders the chat template with :data:`NEUTRAL_READ_CONTEXT`
            as the user turn and *text* as the assistant turn (no system
            prompt, no generation prompt — the reading position is the
            last token of *text* itself).

    Returns:
        The formatted input text.

    Raises:
        ValueError: For an unknown format, or ``"chat"`` without a
            tokenizer.
    """
    if read_format == "raw":
        return text
    if read_format == "chat":
        if tokenizer is None:
            raise ValueError("chat read format requires a tokenizer")
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": NEUTRAL_READ_CONTEXT},
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    raise ValueError(f"unknown read format {read_format!r}")

#: A reader maps text to an EPA score triple.
EpaReader = Callable[[str], EPA]
#: A scorer maps text to a per-token cross-entropy in nats.
ReferenceScorer = Callable[[str], float]


def epa_scores(
    hiddens: Mapping[int, np.ndarray],
    directions: Mapping[str, Mapping[int, np.ndarray]],
    layers: Sequence[int] | None = None,
    prefer_raw: bool = True,
) -> tuple[float, float, float]:
    """Project one reading position onto the per-layer EPA directions.

    For each dimension the projection of ``hiddens[layer]`` onto the
    dimension's direction at ``layer`` is averaged over the layers
    available in both mappings (and the ``layers`` filter, if given).
    The direction vectors already carry the reader's sign, so a
    positive score means the positive side of the dimension.

    Args:
        hiddens: ``{decoder_layer: (hidden_size,) hidden state}``.
        directions: The directions bundle mapping (keys ``"epa.evaluation"``,
            ``"epa.potency"``, ``"epa.activity"``; optionally the
            pre-orthonormalisation ``"epa.{dim}.raw"`` keys).
        layers: Optional layer filter; defaults to all common layers.
        prefer_raw: If True (default) and the mapping carries
            ``"epa.{dim}.raw"``, those pre-orthonormalisation signed unit
            directions are used instead of the (possibly Løwdin-
            orthonormalised) canonical ones — orthonormality is a
            steering-side requirement; raw unit directions are less
            distorted for measurement.

    Returns:
        The (E, P, A) score triple.

    Raises:
        ValueError: If a dimension is missing or no layer is available.
    """
    keys: dict[str, str] = {}
    for dim in EPA_DIMENSIONS:
        raw_key = f"epa.{dim}.raw"
        key = raw_key if prefer_raw and raw_key in directions else f"epa.{dim}"
        if key not in directions:
            raise ValueError(f"directions missing {key!r}")
        keys[dim] = key
    common = _common_layers(hiddens, directions, layers)
    out: list[float] = []
    for dim in EPA_DIMENSIONS:
        key = keys[dim]
        values = [
            float(np.dot(hiddens[layer], directions[key][layer])) for layer in common
        ]
        out.append(float(np.mean(values)))
    return (out[0], out[1], out[2])


def available_reader_layers(
    hiddens: Mapping[int, np.ndarray],
    directions: Mapping[str, Mapping[int, np.ndarray]],
    layers: Sequence[int] | None = None,
) -> list[int]:
    """The layer indices readable for every EPA dimension (sorted)."""
    return _common_layers(hiddens, directions, layers)


def _common_layers(
    hiddens: Mapping[int, np.ndarray],
    directions: Mapping[str, Mapping[int, np.ndarray]],
    layers: Sequence[int] | None,
) -> list[int]:
    """Intersect hidden-state, direction and filter layer sets."""
    pool = set(hiddens)
    for dim in EPA_DIMENSIONS:
        pool &= set(directions[f"epa.{dim}"])
    if layers is not None:
        pool &= set(layers)
    if not pool:
        raise ValueError("no common reader layers between hiddens and directions")
    return sorted(pool)


def build_epa_reader(
    model: Any,
    tokenizer: Any,
    bundle: Any,
    layers: Sequence[int] | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    read_format: str = "raw",
) -> EpaReader:
    """Build a model-bound EPA reader over a directions bundle.

    One forward pass per call; the reading position is the last token of
    the (truncated) input.

    Args:
        model: HuggingFace causal LM with ``output_hidden_states`` support.
        tokenizer: Its tokenizer.
        bundle: A :class:`~act_steering.directions.DirectionsBundle`.
        layers: Optional reader-layer filter; defaults to all layers
            carrying the full EPA direction set.
        max_length: Tokenisation cap for the forward pass.
        read_format: ``"raw"`` (default; flat text) or ``"chat"`` (chat
            template, text in the assistant position — must match the
            format the directions were fit in).

    Returns:
        A callable mapping text to an :class:`~act_steering.act_core.EPA`.
    """
    import torch

    if layers is None:
        layers = sorted(bundle.directions["epa.evaluation"])
    device = getattr(model, "device", None)

    def reader(text: str) -> EPA:
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
        hiddens = {
            layer: out.hidden_states[layer + 1][0, n_tokens - 1]
            .detach()
            .cpu()
            .numpy()
            for layer in layers
        }
        return EPA(*epa_scores(hiddens, bundle.directions, layers))

    return reader


def build_reference_scorer(
    model: Any,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> ReferenceScorer:
    """Build a reference-model scorer (per-token cross-entropy in nats).

    Args:
        model: HuggingFace causal LM used as the reference.
        tokenizer: Its tokenizer.
        max_length: Tokenisation cap for the forward pass.

    Returns:
        A callable mapping text to the mean cross-entropy (nats/token).
    """
    import torch
    import torch.nn.functional as F

    device = getattr(model, "device", None)

    def scorer(text: str) -> float:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[:, :-1, :].float()
        target = input_ids[:, 1:]
        if target.numel() == 0:
            return 0.0
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="mean"
        )
        return float(loss)

    return scorer
