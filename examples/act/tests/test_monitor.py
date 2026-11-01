"""CPU mock tests for the per-token deflection monitor (S7).

The hidden-state provider is faked (no torch, no model), so the tests
cover trace length, the rep_token sweep, median outlier clipping, trigger
firing at the threshold, and silence below the threshold.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from act_steering import monitor as m
from act_steering.config import MonitorConfig
from act_steering.directions import (
    DEFLECTION_DIRECTION,
    EPA_DIMENSIONS,
    DirectionsBundle,
    l2_normalize,
)


def _bundle(rng: np.random.Generator) -> DirectionsBundle:
    """A synthetic directions bundle (hidden_size=16, layers 1/5/7)."""
    hidden_size = 16
    layers = [1, 5, 7]
    names = [DEFLECTION_DIRECTION, *(f"epa.{dim}" for dim in EPA_DIMENSIONS)]
    directions = {
        name: {
            layer: l2_normalize(rng.normal(size=hidden_size)) for layer in layers
        }
        for name in names
    }
    return DirectionsBundle(
        model_name="test/fake-model",
        hidden_size=hidden_size,
        hidden_layers=layers,
        directions=directions,
        seed=42,
    )


@pytest.fixture()
def bundle() -> DirectionsBundle:
    return _bundle(np.random.default_rng(0))


def _scaled_provider(bundle: DirectionsBundle, scale):
    """Provider whose hidden state is ``scale(rep_token) * d`` at every layer.

    The projection onto d is then exactly ``scale(rep_token)`` at each
    layer, so the mean projection equals the scale.
    """

    def provider(text: str, rep_token: int) -> dict[int, np.ndarray]:
        return {
            layer: scale(rep_token) * bundle.direction(DEFLECTION_DIRECTION, layer)
            for layer in bundle.hidden_layers
        }

    return provider


# --- rep_token sweep and trace length ---------------------------------------


def test_token_scores_length_and_values(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: float(-rt))
    scores = m.token_scores(provider, bundle, [1, 5], "some text", n_positions=5)
    # rep_token sweep is -5..-1, so scores are 5, 4, 3, 2, 1
    assert scores.shape == (5,)
    assert np.allclose(scores, [5.0, 4.0, 3.0, 2.0, 1.0])


def test_token_scores_sweeps_rep_tokens(bundle: DirectionsBundle) -> None:
    seen: list[int] = []

    def provider(text: str, rep_token: int) -> dict[int, np.ndarray]:
        seen.append(rep_token)
        return {layer: np.zeros(16) for layer in bundle.hidden_layers}

    m.token_scores(provider, bundle, [1], "x", n_positions=4)
    assert seen == [-4, -3, -2, -1]


def test_token_scores_zero_positions(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: 1.0)
    scores = m.token_scores(provider, bundle, [1], "x", n_positions=0)
    assert scores.shape == (0,)


def test_token_scores_negative_positions(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: 1.0)
    with pytest.raises(ValueError, match="n_positions"):
        m.token_scores(provider, bundle, [1], "x", n_positions=-1)


def test_token_scores_rejects_missing_layer(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: 1.0)
    with pytest.raises(ValueError, match="not in the directions bundle"):
        m.token_scores(provider, bundle, [3], "x", n_positions=1)


# --- median outlier clipping --------------------------------------------------


def test_clip_outliers_replaces_extreme() -> None:
    scores = np.zeros(100)
    scores[99] = -1000.0
    clipped = m.clip_outliers(scores, clip_sigma=5.0)
    assert clipped[99] == pytest.approx(0.0)  # replaced by the median
    assert np.all(clipped[:99] == 0.0)


def test_clip_outliers_keeps_moderate_values() -> None:
    scores = np.array([0.0, 0.1, 0.2, -0.2, -0.1])
    clipped = m.clip_outliers(scores, clip_sigma=5.0)
    assert np.allclose(clipped, scores)


def test_clip_outliers_disabled_at_sigma_zero() -> None:
    scores = np.array([0.0, 0.0, 0.0, -1000.0])
    clipped = m.clip_outliers(scores, clip_sigma=0.0)
    assert np.allclose(clipped, scores)


def test_clip_outliers_empty_and_constant() -> None:
    assert m.clip_outliers(np.array([])).shape == (0,)
    constant = np.array([3.0, 3.0, 3.0])
    assert np.allclose(m.clip_outliers(constant), constant)


# --- token trigger: firing and silence ---------------------------------------


def test_trace_fires_when_scores_cross_threshold() -> None:
    scores = np.full(10, -1.1)  # below -threshold (threshold = 1.0)
    trace = m.build_token_trace(scores, threshold=1.0)
    assert trace.fired is True
    assert trace.n_above == 10
    assert trace.n_tokens == 10


def test_trace_silent_below_trigger_level() -> None:
    scores = np.full(10, 0.0)  # no drift toward high deflection
    trace = m.build_token_trace(scores, threshold=1.0)
    assert trace.fired is False
    assert trace.n_above == 0


def test_trace_boundary_exactly_at_threshold_does_not_fire() -> None:
    scores = np.full(4, -1.0)
    trace = m.build_token_trace(scores, threshold=1.0)
    assert trace.fired is False


def test_trace_single_extreme_token_is_clipped_away() -> None:
    scores = np.zeros(100)
    scores[50] = -1000.0  # one degenerate token
    trace = m.build_token_trace(scores, threshold=1.0, clip_sigma=5.0)
    assert trace.fired is False
    assert trace.clipped[50] == pytest.approx(0.0)


def test_trace_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        m.build_token_trace(np.array([0.0]), threshold=0.0)


def test_token_trace_to_dict_is_json_serialisable() -> None:
    trace = m.build_token_trace(np.array([0.5, -2.0, 1.0]), threshold=1.0)
    payload = json.dumps(trace.to_dict())
    data = json.loads(payload)
    assert data["n_tokens"] == 3
    assert data["n_above"] == 1
    assert data["fired"] is True
    assert len(data["scores"]) == 3
    assert len(data["clipped"]) == 3


# --- sustained re-identification trigger --------------------------------------


def test_sustained_trigger_fires_after_window() -> None:
    trigger = m.sustained_trigger([False, False, True, True, True], window=3)
    assert trigger.fired is True
    assert trigger.consecutive == 3
    assert trigger.turns_recorded == 5
    assert trigger.window == 3


def test_sustained_trigger_short_history() -> None:
    trigger = m.sustained_trigger([True, True], window=3)
    assert trigger.fired is False
    assert "only 2 turn" in trigger.reason


def test_sustained_trigger_broken_run() -> None:
    trigger = m.sustained_trigger([True, True, True, False, True, True], window=3)
    assert trigger.fired is False
    assert trigger.consecutive == 2


def test_sustained_trigger_empty_history() -> None:
    trigger = m.sustained_trigger([], window=3)
    assert trigger.fired is False
    assert trigger.turns_recorded == 0


def test_sustained_trigger_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window"):
        m.sustained_trigger([True], window=0)


def test_reidentification_trigger_to_dict_is_json_serialisable() -> None:
    trigger = m.sustained_trigger([True, True, True], window=3)
    data = json.loads(json.dumps(trigger.to_dict()))
    assert data["fired"] is True
    assert data["turn_fired"] == [True, True, True]
    assert data["reason"]


# --- layer resolution and monitor glue ----------------------------------------


def test_resolve_reader_layers_auto_selects_last(bundle: DirectionsBundle) -> None:
    assert m.resolve_reader_layers(bundle, MonitorConfig()) == [7]


def test_resolve_reader_layers_explicit_and_dedup(bundle: DirectionsBundle) -> None:
    assert m.resolve_reader_layers(
        bundle, MonitorConfig(reader_layers=[5, 1, 5])
    ) == [1, 5]


def test_resolve_reader_layers_rejects_missing(bundle: DirectionsBundle) -> None:
    with pytest.raises(ValueError, match="not in the directions bundle"):
        m.resolve_reader_layers(bundle, MonitorConfig(reader_layers=[3]))


def test_monitor_trace_end_to_end(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: float(-rt) * 0.5)
    monitor = m.DeflectionMonitor(provider, bundle, MonitorConfig(threshold=1.0))
    trace = monitor.trace("context response", n_positions=4)
    # scores = 0.5 * (4, 3, 2, 1) = 2.0, 1.5, 1.0, 0.5 -> all above -1.0
    assert np.allclose(trace.scores, [2.0, 1.5, 1.0, 0.5])
    assert trace.fired is False


def test_monitor_trace_fires_with_negative_drift(bundle: DirectionsBundle) -> None:
    provider = _scaled_provider(bundle, scale=lambda rt: -2.0)
    monitor = m.DeflectionMonitor(provider, bundle, MonitorConfig(threshold=1.0))
    trace = monitor.trace("context response", n_positions=3)
    assert np.allclose(trace.scores, [-2.0, -2.0, -2.0])
    assert trace.fired is True
    assert trace.n_above == 3


def test_monitor_uses_resolved_layers(bundle: DirectionsBundle) -> None:
    """Layer averaging: with per-layer scales the trace is their mean."""

    def provider(text: str, rep_token: int) -> dict[int, np.ndarray]:
        return {
            1: 2.0 * bundle.direction(DEFLECTION_DIRECTION, 1),
            5: -4.0 * bundle.direction(DEFLECTION_DIRECTION, 5),
            7: 0.0 * bundle.direction(DEFLECTION_DIRECTION, 7),
        }

    monitor = m.DeflectionMonitor(
        provider, bundle, MonitorConfig(threshold=0.5, reader_layers=[1, 5])
    )
    trace = monitor.trace("x", n_positions=2)
    assert np.allclose(trace.scores, [-1.0, -1.0])  # mean of 2 and -4
    assert trace.fired is True
    assert trace.n_above == 2


def test_make_hidden_provider_moves_inputs_to_model_device() -> None:
    """The provider must forward on ``model.device``.

    Regression test for the "Expected all tensors to be on the same
    device, but found cuda:0 and cpu" crash: the fake model advertises
    the ``meta`` device (which exists on machines without CUDA) and
    records where it actually receives ``input_ids``.
    """
    from types import SimpleNamespace

    import torch

    hidden_size = 4
    layers = [1]

    class _Tok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
            n = max(1, len(text.split()))
            return {
                "input_ids": torch.arange(n, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
            }

    class _Model:
        device = torch.device("meta")

        def __init__(self) -> None:
            self.seen_devices: list = []

        def __call__(self, input_ids, attention_mask, output_hidden_states=None, **kw):
            assert output_hidden_states
            self.seen_devices.append(input_ids.device)
            seq = int(input_ids.shape[1])
            return SimpleNamespace(
                hidden_states=tuple(
                    torch.full((1, seq, hidden_size), float(i))
                    for i in range(max(layers) + 2)
                )
            )

    model = _Model()
    provider = m.make_hidden_provider(model, _Tok(), layers)
    out = provider("hello world", -1)
    assert model.seen_devices == [torch.device("meta")]
    assert set(out) == {1}
    # hidden_states[layer + 1] -> element 2 -> filled with 2.0
    assert out[1].shape == (hidden_size,)
    assert np.allclose(out[1], 2.0)
