"""Tests for act_steering.readers (projection core, model-bound builders)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from act_steering import directions as d
from act_steering.act_core import EPA
from act_steering.readers import (
    available_reader_layers,
    build_epa_reader,
    build_reference_scorer,
    epa_scores,
)

HIDDEN_SIZE = 4
LAYERS = [2, 3]


def _bundle() -> d.DirectionsBundle:
    """A bundle with hand-crafted directions of known projections."""
    directions: dict[str, dict[int, np.ndarray]] = {
        "epa.evaluation": {2: np.array([1.0, 0.0, 0.0, 0.0]), 3: np.array([0.0, 1.0, 0.0, 0.0])},
        "epa.potency": {2: np.array([0.0, 0.0, 1.0, 0.0]), 3: np.array([0.0, 0.0, 0.0, 1.0])},
        "epa.activity": {2: np.array([0.6, 0.0, 0.0, 0.0]), 3: np.array([0.0, 0.8, 0.0, 0.0])},
        "deflection": {2: np.array([1.0, 0.0, 0.0, 0.0]), 3: np.array([0.0, 1.0, 0.0, 0.0])},
    }
    return d.DirectionsBundle(
        model_name="test/readers-fake",
        hidden_size=HIDDEN_SIZE,
        hidden_layers=LAYERS,
        directions=directions,
        created="2026-10-30T00:00:00+00:00",
        seed=42,
        lowdin=False,
        contrast_set_path="artifacts/contrast_set.json",
        n_contrast_pairs=2,
        epa_items_per_side=2,
        stats={},
    )


HIDDENS = {2: np.array([1.0, 1.0, 1.0, 1.0]), 3: np.array([2.0, 2.0, 2.0, 2.0])}


def test_epa_scores_averages_across_layers() -> None:
    e, p, a = epa_scores(HIDDENS, _bundle().directions)
    assert e == pytest.approx(1.5)
    assert p == pytest.approx(1.5)
    assert a == pytest.approx(1.1)


def test_epa_scores_layer_filter() -> None:
    e, p, a = epa_scores(HIDDENS, _bundle().directions, layers=[2])
    assert e == pytest.approx(1.0)
    assert p == pytest.approx(1.0)
    assert a == pytest.approx(0.6)


def test_available_reader_layers() -> None:
    assert available_reader_layers(HIDDENS, _bundle().directions) == LAYERS
    assert available_reader_layers(HIDDENS, _bundle().directions, [3, 9]) == [3]


def test_epa_scores_no_common_layers() -> None:
    with pytest.raises(ValueError, match="no common reader layers"):
        epa_scores({9: np.zeros(HIDDEN_SIZE)}, _bundle().directions)


def test_epa_scores_missing_dimension() -> None:
    directions = _bundle().directions
    del directions["epa.activity"]
    with pytest.raises(ValueError, match="epa.activity"):
        epa_scores(HIDDENS, directions)


def test_epa_scores_prefers_raw_when_present() -> None:
    """Bundles carrying epa.{dim}.raw are read through the raw (pre-
    orthonormalisation) directions by default."""
    bundle = _bundle()
    directions = bundle.directions
    # Raw directions point along the first axis for every dimension/layer:
    # with HIDDENS, every dimension scores the same known value.
    raw = {2: np.array([1.0, 0.0, 0.0, 0.0]), 3: np.array([0.0, 1.0, 0.0, 0.0])}
    for dim in ("evaluation", "potency", "activity"):
        directions[f"epa.{dim}.raw"] = {layer: v.copy() for layer, v in raw.items()}
    e, p, a = epa_scores(HIDDENS, directions)
    # canonical would give (1.5, 1.5, 1.1); raw gives (1.5, 1.5, 1.5)
    assert (e, p, a) == pytest.approx((1.5, 1.5, 1.5))


def test_epa_scores_can_opt_out_of_raw() -> None:
    bundle = _bundle()
    directions = bundle.directions
    raw = {2: np.array([1.0, 0.0, 0.0, 0.0]), 3: np.array([0.0, 1.0, 0.0, 0.0])}
    for dim in ("evaluation", "potency", "activity"):
        directions[f"epa.{dim}.raw"] = {layer: v.copy() for layer, v in raw.items()}
    e, p, a = epa_scores(HIDDENS, directions, prefer_raw=False)
    assert (e, p, a) == pytest.approx((1.5, 1.5, 1.1))


def test_epa_scores_missing_raw_falls_back_per_dimension() -> None:
    """A partial .raw set (one dim only) falls back per dimension."""
    directions = _bundle().directions
    diag = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    directions["epa.evaluation.raw"] = {2: diag.copy(), 3: diag.copy()}
    e, p, a = epa_scores(HIDDENS, directions)
    # evaluation: raw diag -> (sqrt(2) + 2*sqrt(2))/2 = 1.5*sqrt(2)
    # potency/activity: canonical -> 1.5, 1.1
    assert (e, p, a) == pytest.approx((1.5 * np.sqrt(2.0), 1.5, 1.1))


class _FakeTokenizer:
    """Minimal tokenizer returning fixed-length fake inputs."""

    def __init__(self, n_tokens: int = 3) -> None:
        self.n_tokens = n_tokens

    def __call__(
        self,
        text: str,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, Any]:
        import torch

        n = 1 if not text else self.n_tokens
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "attention_mask": torch.ones(1, n),
        }


class _FakeModel:
    """Minimal causal-LM fake: hidden states per layer plus logits."""

    def __init__(self, hidden_size: int, n_layers: int, vocab: int = 7, n_tokens: int = 3) -> None:
        import torch

        self.device = "cpu"
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._vocab = vocab
        self._n_tokens = n_tokens
        self._torch = torch

    def __call__(
        self,
        input_ids: Any,
        attention_mask: Any,
        output_hidden_states: bool = False,
    ) -> Any:
        torch = self._torch
        hidden = torch.zeros(1, self._n_tokens, self._hidden_size)
        hidden[:, :, 0] = 1.0
        layers = (hidden,) * (self._n_layers + 1)

        class _Out:
            pass

        out = _Out()
        out.hidden_states = layers
        out.logits = torch.zeros(1, self._n_tokens, self._vocab)
        out.logits[:, :, 1] = 5.0
        return out


def _fake_bundle_pair() -> tuple[d.DirectionsBundle, _FakeModel, _FakeTokenizer]:
    bundle = _bundle()
    model = _FakeModel(HIDDEN_SIZE, max(LAYERS) + 1)
    tokenizer = _FakeTokenizer()
    return bundle, model, tokenizer


def test_build_epa_reader_projects_last_token() -> None:
    bundle, model, tokenizer = _fake_bundle_pair()
    reader = build_epa_reader(model, tokenizer, bundle, layers=LAYERS)
    epa = reader("hello world")
    # All hidden states are [1, 0, 0, 0]; projections: E = layer2 dot = 1.0,
    # layer3 dot = 0.0 -> mean 0.5; P = 0.0; A = 0.6 / 0.0 -> mean 0.3.
    assert isinstance(epa, EPA)
    assert epa.e == pytest.approx(0.5)
    assert epa.p == pytest.approx(0.0)
    assert epa.a == pytest.approx(0.3)


def test_build_epa_reader_default_layers() -> None:
    bundle, model, tokenizer = _fake_bundle_pair()
    reader = build_epa_reader(model, tokenizer, bundle)
    epa = reader("hello world")
    assert epa.e == pytest.approx(0.5)


def test_build_reference_scorer_returns_nats() -> None:
    bundle, model, tokenizer = _fake_bundle_pair()
    scorer = build_reference_scorer(model, tokenizer)
    value = scorer("hello world")
    # Logits: 5.0 at class 1, 0.0 elsewhere -> CE for target token 1 is
    # -log(softplus-ish); every other target has the same loss, all > 0.
    assert value > 0.0


def test_build_reference_scorer_empty_target() -> None:
    bundle, model, tokenizer = _fake_bundle_pair()
    scorer = build_reference_scorer(model, tokenizer)
    assert scorer("") == 0.0
