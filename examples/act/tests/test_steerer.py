"""CPU tests for the deflection steerer (S7).

Covers the torch-free core (proportional gain, offset computation, layer
resolution, config validation) and the model-bound glue through a
monkeypatched ``RepControlPipeline`` (no GPU, no real model).
"""

from __future__ import annotations

import numpy as np
import pytest

from act_steering import steerer as s
from act_steering.config import SteeringConfig
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


@pytest.fixture()
def config() -> SteeringConfig:
    return SteeringConfig(layers=[1, 5], alpha=0.5, u_ref=10.0)


# --- proportional gain ------------------------------------------------------


def test_proportional_gain_at_reference_point() -> None:
    config = SteeringConfig(alpha=0.5, kappa=1.0, u_ref=10.0)
    assert s.proportional_gain(10.0, config) == pytest.approx(0.5)


def test_proportional_gain_clamps() -> None:
    config = SteeringConfig(alpha=0.5, alpha_min=0.0, alpha_max=1.5, u_ref=10.0)
    assert s.proportional_gain(0.0, config) == pytest.approx(0.0)
    assert s.proportional_gain(1000.0, config) == pytest.approx(1.5)
    config2 = SteeringConfig(alpha=0.5, alpha_min=0.2, alpha_max=1.5, u_ref=10.0)
    assert s.proportional_gain(0.0, config2) == pytest.approx(0.2)


def test_proportional_gain_kappa_scales() -> None:
    base = SteeringConfig(alpha=0.5, kappa=1.0, u_ref=10.0)
    double = SteeringConfig(alpha=0.5, kappa=2.0, u_ref=10.0)
    assert s.proportional_gain(4.0, double) == pytest.approx(
        2.0 * s.proportional_gain(4.0, base)
    )


def test_proportional_gain_rejects_negative() -> None:
    with pytest.raises(ValueError, match="u_pre"):
        s.proportional_gain(-1.0, SteeringConfig())


# --- config validation ------------------------------------------------------


def test_validate_steering_config_invalid_fields() -> None:
    with pytest.raises(ValueError, match="mode"):
        s.validate_steering_config(SteeringConfig(mode="bogus"))
    with pytest.raises(ValueError, match="alpha_min"):
        s.validate_steering_config(SteeringConfig(alpha_min=2.0, alpha_max=1.0))
    with pytest.raises(ValueError, match="u_ref"):
        s.validate_steering_config(SteeringConfig(u_ref=0.0))
    with pytest.raises(ValueError, match="token_pos"):
        s.validate_steering_config(SteeringConfig(token_pos="middle"))
    with pytest.raises(ValueError, match="contrast_tokens"):
        s.validate_steering_config(SteeringConfig(mode="inloop", contrast_tokens=8))


def test_validate_steering_config_accepts_defaults() -> None:
    s.validate_steering_config(SteeringConfig())
    s.validate_steering_config(SteeringConfig(mode="inloop", contrast_tokens=-8))


# --- layer resolution -------------------------------------------------------


def test_resolve_layers_auto_selects_last(bundle: DirectionsBundle) -> None:
    assert s.resolve_layers(bundle, SteeringConfig(layers=[])) == [7]


def test_resolve_layers_explicit_and_dedup(bundle: DirectionsBundle) -> None:
    assert s.resolve_layers(bundle, SteeringConfig(layers=[5, 1, 5])) == [1, 5]


def test_resolve_layers_rejects_missing(bundle: DirectionsBundle) -> None:
    with pytest.raises(ValueError, match="not in the directions bundle"):
        s.resolve_layers(bundle, SteeringConfig(layers=[3]))


def test_resolve_model_layers() -> None:
    class FakeConfig:
        num_hidden_layers = 32

    class FakeModel:
        config = FakeConfig()

    assert s.resolve_model_layers(FakeModel(), SteeringConfig(layers=[])) == [31]
    assert s.resolve_model_layers(FakeModel(), SteeringConfig(layers=[3, 31])) == [3, 31]
    with pytest.raises(ValueError, match="out of range"):
        s.resolve_model_layers(FakeModel(), SteeringConfig(layers=[32]))


# --- offset computation -----------------------------------------------------


def test_deflection_offsets_are_gained_directions(
    bundle: DirectionsBundle, config: SteeringConfig
) -> None:
    offsets = s.deflection_offsets(bundle, config, u_pre=20.0)
    gain = s.proportional_gain(20.0, config)  # 0.5 * 20/10 = 1.0
    assert set(offsets) == {1, 5}
    for layer in (1, 5):
        assert np.allclose(offsets[layer], gain * bundle.direction(DEFLECTION_DIRECTION, layer))
        assert np.linalg.norm(offsets[layer]) == pytest.approx(gain)


def test_deflection_offsets_zero_at_alpha_min(
    bundle: DirectionsBundle,
) -> None:
    config = SteeringConfig(layers=[1], alpha=0.5, alpha_min=0.0, u_ref=10.0)
    offsets = s.deflection_offsets(bundle, config, u_pre=0.0)
    assert np.all(offsets[1] == 0.0)


def test_epa_target_offsets_single_dimension(bundle: DirectionsBundle) -> None:
    config = SteeringConfig(layers=[1], alpha=0.5)
    offsets = s.epa_target_offsets(bundle, config, target=(2.0, 0.0, 0.0))
    expected = s.EPA_TARGET_COEFFICIENTS[0] * 2.0 * bundle.direction("epa.evaluation", 1)
    assert np.allclose(offsets[1], expected)


def test_epa_target_offsets_zero_target(bundle: DirectionsBundle) -> None:
    offsets = s.epa_target_offsets(bundle, SteeringConfig(layers=[1]), target=(0, 0, 0))
    assert np.all(offsets[1] == 0.0)


def test_epa_target_offsets_rejects_bad_target(bundle: DirectionsBundle) -> None:
    with pytest.raises(ValueError, match="3 components"):
        s.epa_target_offsets(bundle, SteeringConfig(layers=[1]), target=(1.0, 2.0))


# --- static steerer glue (monkeypatched pipeline) ---------------------------


class _FakeModel:
    def __init__(self) -> None:
        import torch

        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.config = None


class _RecordingPipeline:
    """Stands in for RepControlPipeline and records every interaction."""

    def __init__(self, model, tokenizer, layers) -> None:
        self.wrapped_model = self
        self.layers = layers
        self.controller = None
        self.reset_count = 0

    def set_controller(
        self, layers, activations, block_name, token_pos, masks=None
    ) -> None:
        self.controller = {
            "layers": layers,
            "activations": activations,
            "block_name": block_name,
            "token_pos": token_pos,
            "masks": masks,
        }

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return [{"generated_text": "steered"}]

    def reset(self) -> None:
        self.reset_count += 1


@pytest.fixture()
def recording_pipeline(monkeypatch: pytest.MonkeyPatch):
    import repe.rep_control_pipeline as rpc

    created: dict = {}

    def factory(model, tokenizer, layers):
        pipeline = _RecordingPipeline(model, tokenizer, layers)
        created["pipeline"] = pipeline
        return pipeline

    monkeypatch.setattr(rpc, "RepControlPipeline", factory)
    return created


def test_steerer_generate_applies_per_turn_offsets(
    bundle: DirectionsBundle, config: SteeringConfig, recording_pipeline: dict
) -> None:
    model = _FakeModel()
    steerer = s.DeflectionSteerer(model, None, bundle, config)
    out = steerer.generate("a prompt", u_pre=20.0, max_new_tokens=16)
    pipeline = recording_pipeline["pipeline"]
    assert out == [{"generated_text": "steered"}]
    assert pipeline.controller["block_name"] == "decoder_block"
    assert pipeline.controller["token_pos"] == "end"
    assert pipeline.controller["layers"] == [1, 5]
    assert pipeline.prompt == "a prompt"
    assert pipeline.kwargs == {"max_new_tokens": 16}
    assert pipeline.reset_count == 1
    gain = s.proportional_gain(20.0, config)
    for layer in (1, 5):
        expected = gain * bundle.direction(DEFLECTION_DIRECTION, layer)
        actual = pipeline.controller["activations"][layer].numpy()
        assert actual.shape == (bundle.hidden_size,)
        assert np.allclose(actual, expected, atol=1e-6)


def test_steerer_epa_target_mode(
    bundle: DirectionsBundle, config: SteeringConfig, recording_pipeline: dict
) -> None:
    steerer = s.DeflectionSteerer(_FakeModel(), None, bundle, config)
    steerer.generate("p", u_pre=0.0, mode="epa_target", target=(1.0, 0.0, 0.0))
    pipeline = recording_pipeline["pipeline"]
    expected = s.EPA_TARGET_COEFFICIENTS[0] * 1.0 * bundle.direction("epa.evaluation", 1)
    actual = pipeline.controller["activations"][1].numpy()
    assert np.allclose(actual, expected, atol=1e-6)


def test_controller_masks_end_and_start() -> None:
    import torch

    for pos in ("end", "start"):
        masks = s.controller_masks(pos, device=torch.device("cpu"), dtype=torch.float32)
        assert masks is not None
        assert masks.shape == (1, 1, 1)
        assert torch.all(masks == 1)
    assert s.controller_masks(None) is None
    assert s.controller_masks(3) is None


def test_steerer_generate_passes_tail_mask(
    bundle: DirectionsBundle, recording_pipeline: dict
) -> None:
    import torch

    config = SteeringConfig(layers=[1, 5], alpha=0.5, u_ref=10.0, token_pos="end")
    steerer = s.DeflectionSteerer(_FakeModel(), None, bundle, config)
    steerer.generate("p", u_pre=10.0)
    masks = recording_pipeline["pipeline"].controller["masks"]
    assert masks is not None
    assert masks.shape == (1, 1, 1)
    assert torch.all(masks == 1)


def test_steerer_generate_no_mask_when_token_pos_none(
    bundle: DirectionsBundle, recording_pipeline: dict
) -> None:
    config = SteeringConfig(layers=[1, 5], alpha=0.5, u_ref=10.0, token_pos=None)
    steerer = s.DeflectionSteerer(_FakeModel(), None, bundle, config)
    steerer.generate("p", u_pre=10.0)
    pipeline = recording_pipeline["pipeline"]
    assert pipeline.controller["token_pos"] is None
    assert pipeline.controller["masks"] is None


def test_end_branch_needs_explicit_mask_on_upstream_wrapped_block() -> None:
    """The installed upstream ``WrappedBlock`` 'end' branch with a 1-D
    controller.

    Without an explicit mask, the ``position_ids``-derived mask
    ``(1, seq, 1)`` cannot broadcast against the ``(1, 1, hidden)``
    target and the in-place add raises. With the ``(1, 1, 1)`` mask from
    :func:`act_steering.steerer.controller_masks`, only the last position
    (prefill) / the single new token (decode) is steered.
    """
    import torch
    from repe.rep_control_reading_vec import WrappedBlock

    class _PassthroughBlock(torch.nn.Module):
        def forward(self, x, **kwargs):  # noqa: ARG002 - decoder kwargs ignored
            return x

    hidden, seq = 8, 26
    position_ids = torch.arange(seq).unsqueeze(0)

    block = WrappedBlock(_PassthroughBlock())
    block.set_controller(torch.ones(hidden), token_pos="end")
    with pytest.raises(RuntimeError, match="expanded size"):
        block(torch.zeros(1, seq, hidden), position_ids=position_ids)

    block.reset()
    block.set_controller(
        torch.ones(hidden),
        token_pos="end",
        masks=s.controller_masks("end", dtype=torch.float32),
    )
    hidden_states = torch.zeros(1, seq, hidden)
    out = block(hidden_states, position_ids=position_ids)
    assert torch.all(out[:, :-1] == 0.0)
    assert torch.all(out[:, -1] == 1.0)

    # Decode step: a single new token at a non-zero position works with
    # the same mask.
    block.reset()
    block.set_controller(
        torch.ones(hidden),
        token_pos="end",
        masks=s.controller_masks("end", dtype=torch.float32),
    )
    out = block(torch.zeros(1, 1, hidden), position_ids=torch.tensor([[seq]]))
    assert torch.all(out == 1.0)


def test_steerer_rejects_non_static_mode(bundle: DirectionsBundle) -> None:
    with pytest.raises(ValueError, match="static"):
        s.DeflectionSteerer(
            _FakeModel(), None, bundle, SteeringConfig(mode="inloop", contrast_tokens=-8)
        )


def test_steerer_activations_unknown_mode(
    bundle: DirectionsBundle, config: SteeringConfig, recording_pipeline: dict
) -> None:
    steerer = s.DeflectionSteerer(_FakeModel(), None, bundle, config)
    with pytest.raises(ValueError, match="unknown steering mode"):
        steerer.activations(1.0, mode="bogus")


def test_steerer_activations_epa_requires_target(
    bundle: DirectionsBundle,
    config: SteeringConfig,
    recording_pipeline: dict,
) -> None:
    steerer = s.DeflectionSteerer(_FakeModel(), None, bundle, config)
    with pytest.raises(ValueError, match="requires the b\\* target"):
        steerer.activations(1.0, mode="epa_target")


# --- in-loop contrast steerer glue ------------------------------------------


class _FakeTokenizer:
    """Minimal callable tokenizer returning CPU tensors."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.calls: list[str] = []

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        self.calls.append(text)
        n = len(text)
        torch = self.torch
        return {
            "input_ids": torch.arange(1, n + 1, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }


class _FakeContrastModel:
    def __init__(self) -> None:
        import torch

        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.generated: dict = {}

        class _Cfg:
            num_hidden_layers = 32

        self.config = _Cfg()

    def generate(self, **kwargs):
        self.generated = kwargs
        return [[42]]


def test_inloop_steerer_requires_inloop_mode() -> None:
    with pytest.raises(ValueError, match="inloop"):
        s.InLoopContrastSteerer(
            _FakeContrastModel(), _FakeTokenizer(), SteeringConfig(mode="static")
        )


def test_inloop_steerer_rejects_non_contrast_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import repe.rep_control_contrast_vec as rcv

    # Point the isinstance check at a class our fake does not inherit from.
    monkeypatch.setattr(rcv, "ContrastVecLlamaForCausalLM", type("NotLlama", (), {}))
    monkeypatch.setattr(rcv, "ContrastVecMistralForCausalLM", type("NotMistral", (), {}))
    inloop_config = SteeringConfig(mode="inloop", contrast_tokens=-8)
    with pytest.raises(TypeError, match="ContrastVec"):
        s.InLoopContrastSteerer(_FakeContrastModel(), _FakeTokenizer(), inloop_config)


def _unvalidated_inloop() -> s.InLoopContrastSteerer:
    """Build an InLoopContrastSteerer bypassing __init__ (no model check)."""
    steerer = s.InLoopContrastSteerer.__new__(s.InLoopContrastSteerer)
    steerer.model = _FakeContrastModel()
    steerer.tokenizer = _FakeTokenizer()
    steerer.config = SteeringConfig(mode="inloop", contrast_tokens=-8, alpha=0.25, layers=[31])
    steerer.layers = [31]
    return steerer


def test_inloop_generate_rejects_use_cache() -> None:
    steerer = _unvalidated_inloop()
    with pytest.raises(ValueError, match="use_cache=False"):
        steerer.generate("ctx", "low continuation", "high continuation", use_cache=True)


def test_inloop_generate_wires_contrast_kwargs() -> None:
    steerer = _unvalidated_inloop()
    out = steerer.generate(
        "ctx", "low continuation", "high continuation", max_length=32
    )
    assert out == [[42]]
    kwargs = steerer.model.generated
    assert kwargs["compute_contrast"] is True
    assert kwargs["alpha"] == pytest.approx(0.25)  # config default
    assert kwargs["contrast_tokens"] == -8
    assert kwargs["control_layer_ids"] == [31]
    assert kwargs["use_cache"] is False
    assert kwargs["max_length"] == 32
    # Prompt ids from the main call, pos/neg ids from their own calls.
    assert steerer.tokenizer.calls == [
        "ctx",
        "low continuation",
        "high continuation",
    ]
    assert kwargs["input_ids"].shape[1] == 3  # len("ctx")
    assert kwargs["pos_input_ids"].shape[1] == 16  # len("low continuation")
    assert kwargs["neg_input_ids"].shape[1] == 17  # len("high continuation")


def test_inloop_generate_explicit_alpha() -> None:
    steerer = _unvalidated_inloop()
    steerer.generate("ctx", "p", "n", alpha=0.9)
    assert steerer.model.generated["alpha"] == pytest.approx(0.9)
