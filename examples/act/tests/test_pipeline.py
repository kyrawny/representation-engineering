"""CPU mock tests for the per-turn control loop (S10).

The synthetic coefficient set gives ``U = 4 * bE^2`` for both events, so
every deflection in the loop is hand-computable. The conversation state
still uses the real identity dictionary (student/teacher).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from act_steering.act_core import ACTCoefficients
from act_steering.config import DEFAULT_SEED, PipelineConfig, StateConfig
from act_steering.dictionary import ACTDictionary
from act_steering.monitor import TokenTrace
from act_steering.pipeline import ACTSteeringPipeline, PipelineComponents, main
from act_steering.state import ConversationState

DATA_DIR = r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act"

#: Behaviour table with hand-computable deflections: U = 4 * bE^2.
BEHAVIOUR_TABLE: dict[str, tuple[float, float, float]] = {
    "bad": (2.0, 0.0, 0.0),  # U = 16
    "bad2": (3.0, 0.0, 0.0),  # U = 36
    "ok": (0.5, 0.0, 0.0),  # U = 1
    "repair": (0.0, 0.0, 0.0),  # U = 0
    "fallback": (1.0, 0.0, 0.0),  # U = 4
}
MESSAGE_TABLE: dict[str, tuple[float, float, float]] = {
    "hello": (1.0, 0.0, 0.0),  # u1 = 4
    "again": (0.5, 0.0, 0.0),  # u1 = 1
}


def _synthetic_coeffs() -> ACTCoefficients:
    """Identity impressions plus one extra bE term: U = (2*bE)^2 exactly."""
    matrix = np.zeros((15, 9))
    masks = np.zeros(15, dtype=np.int64)
    for j in range(9):
        masks[j] = 1 << j
        matrix[j, j] = 1.0
    masks[9] = 1 << 3
    matrix[9, 3] = 2.0
    return ACTCoefficients(matrix=matrix, masks=masks)


@pytest.fixture(scope="module")
def dictionary() -> ACTDictionary:
    return ACTDictionary.load(DATA_DIR)


@pytest.fixture()
def coeffs() -> ACTCoefficients:
    return _synthetic_coeffs()


def _reader(table: dict[str, tuple[float, float, float]]):
    def reader(text: str):
        return table[text]

    return reader


def _boom(text: str):
    raise RuntimeError("reader down")


def _make_pipeline(
    coeffs: ACTCoefficients,
    dictionary: ACTDictionary,
    *,
    generator=None,
    repair=None,
    monitor=None,
    config=None,
    behaviour_table=BEHAVIOUR_TABLE,
    message_table=MESSAGE_TABLE,
    state_config=None,
):
    state = ConversationState.from_ids(
        dictionary, "student", "teacher", state_config
    )
    components = PipelineComponents(
        message_reader=_reader(message_table),
        candidate_generator=generator or (lambda _s: ["bad", "bad2", "ok"]),
        behaviour_reader=_reader(behaviour_table),
        monitor=monitor,
        repair_policy=repair,
    )
    return ACTSteeringPipeline(coeffs, state, components, config or PipelineConfig(k=3))


# --- basic turn loop ------------------------------------------------------------


def test_run_turn_full_entry(coeffs: ACTCoefficients, dictionary: ACTDictionary) -> None:
    pipeline = _make_pipeline(coeffs, dictionary)
    entry = pipeline.run_turn("hello")
    assert entry["turn"] == 0
    assert entry["u_pre"] == pytest.approx(4.0)  # 4 * 1^2
    assert len(entry["candidates"]) == 3
    assert [c["deflection"] for c in entry["candidates"]] == pytest.approx([16.0, 36.0, 1.0])
    assert entry["winner_index"] == 2
    assert entry["u_post"] == pytest.approx(1.0)
    assert entry["response"] == "ok"
    assert entry["response_u"] == pytest.approx(1.0)
    assert entry["fallback"] is None
    assert entry["repair"] is None
    assert entry["monitor"] is None
    assert entry["trigger_fired"] is False
    assert pipeline.state.turn == 1
    json.dumps(entry)  # must be JSON-safe


def test_run_conversation_and_save(
    coeffs: ACTCoefficients, dictionary: ACTDictionary, tmp_path
) -> None:
    pipeline = _make_pipeline(coeffs, dictionary)
    log = pipeline.run_conversation(["hello", "again"])
    assert log["user_term"] == "student"
    assert log["agent_term"] == "teacher"
    assert len(log["turns"]) == 2
    assert log["turns"][1]["turn"] == 1
    out = pipeline.save_log(log, tmp_path / "sub" / "trace.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["turns"]) == 2


# --- repair (re-identification) --------------------------------------------------


def test_repair_adopted_when_better(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    config = PipelineConfig(k=3, repair="reidentify", repair_threshold=0.5)
    pipeline = _make_pipeline(
        coeffs, dictionary, repair=lambda _s, _r: ["repair"], config=config
    )
    entry = pipeline.run_turn("hello")
    assert entry["repair"] is not None
    assert entry["repair"]["adopted"] is True
    assert entry["repair"]["from_u"] == pytest.approx(1.0)
    assert entry["repair"]["to_u"] == pytest.approx(0.0)
    assert entry["response"] == "repair"
    assert entry["u_post"] == pytest.approx(0.0)


def test_repair_not_adopted_when_worse(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    config = PipelineConfig(k=3, repair="reidentify", repair_threshold=0.5)
    pipeline = _make_pipeline(
        coeffs, dictionary, repair=lambda _s, _r: ["bad2"], config=config
    )
    entry = pipeline.run_turn("hello")
    assert entry["repair"]["adopted"] is False
    assert entry["response"] == "ok"
    assert entry["u_post"] == pytest.approx(1.0)


def test_repair_ignored_when_policy_disabled(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    # u_post = 1 > 0.5 would flag the turn, but repair policy is "none".
    config = PipelineConfig(k=3, repair="none", repair_threshold=0.5)
    pipeline = _make_pipeline(
        coeffs, dictionary, repair=lambda _s, _r: ["repair"], config=config
    )
    entry = pipeline.run_turn("hello")
    assert entry["repair"] is None
    assert entry["response"] == "ok"


def test_repair_limited_to_max_attempts(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    config = PipelineConfig(
        k=3, repair="reidentify", repair_threshold=0.5, max_repair_attempts=1
    )
    # Only the first repair ("bad2", U=36) may be tried; "repair" (U=0) is cut.
    pipeline = _make_pipeline(
        coeffs, dictionary, repair=lambda _s, _r: ["bad2", "bad", "repair"], config=config
    )
    entry = pipeline.run_turn("hello")
    assert entry["repair"]["adopted"] is False
    assert entry["response"] == "ok"


def test_repair_skipped_when_not_flagged(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    # u_post = 1 <= 10 (default threshold), no trigger, no monitor: unflagged.
    config = PipelineConfig(k=3, repair="reidentify")
    pipeline = _make_pipeline(
        coeffs, dictionary, repair=lambda _s, _r: ["repair"], config=config
    )
    entry = pipeline.run_turn("hello")
    assert entry["repair"] is None
    assert entry["response"] == "ok"


# --- fallbacks and failures -------------------------------------------------------


def test_fallback_response_on_total_failure(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    config = PipelineConfig(k=3, fallback_response="fallback")
    # The behaviour reader only knows the fallback text: candidates fail
    # scoring, the fallback reads fine (U = 4 * 1^2).
    pipeline = _make_pipeline(
        coeffs,
        dictionary,
        behaviour_table={"fallback": (1.0, 0.0, 0.0)},
        config=config,
    )
    entry = pipeline.run_turn("hello")
    assert entry["fallback"] == "candidate-scoring-failed"
    assert entry["response"] == "fallback"
    assert entry["u_post"] == pytest.approx(4.0)  # 4 * 1^2
    assert pipeline.state.turn == 1


def test_total_failure_without_fallback(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    pipeline = _make_pipeline(coeffs, dictionary)
    pipeline.components.behaviour_reader = _boom
    entry = pipeline.run_turn("hello")
    assert entry["fallback"] == "all-candidates-unscorable"
    assert entry["response"] is None
    assert entry["u_post"] == pytest.approx(0.0)  # neutral placeholder
    assert entry["response_u"] == pytest.approx(0.0)
    assert pipeline.state.turn == 1


def test_message_reader_failure_neutral_epa(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    pipeline = _make_pipeline(coeffs, dictionary)
    pipeline.components.message_reader = _boom
    entry = pipeline.run_turn("hello")
    assert entry["message_epa_read_failed"] is True
    assert entry["message_epa"] is None
    assert entry["u_pre"] == pytest.approx(0.0)  # neutral message: 4 * 0^2


def test_empty_candidates_raise(coeffs: ACTCoefficients, dictionary: ACTDictionary) -> None:
    pipeline = _make_pipeline(coeffs, dictionary, generator=lambda _s: [])
    with pytest.raises(ValueError, match="no candidates"):
        pipeline.run_turn("hello")


# --- trigger and monitor logging --------------------------------------------------


def test_trigger_latched_and_recorded(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state_config = StateConfig(deflection_threshold=1.0, sustained_turns=1)
    pipeline = _make_pipeline(coeffs, dictionary, state_config=state_config)
    entry = pipeline.run_turn("hello")  # u1 = 4 > 1
    assert entry["trigger_fired"] is True
    assert pipeline.state.trigger_fired is True


def test_monitor_trace_logged(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    trace = TokenTrace(
        scores=(0.1, 0.2),
        clipped=(0.1, 0.2),
        threshold=1.0,
        clip_sigma=5.0,
        n_above=0,
        fired=False,
        mean=0.15,
        median=0.15,
        std=0.05,
        peak=0.2,
        peak_index=1,
    )
    config = PipelineConfig(k=3, monitor_positions=4)
    pipeline = _make_pipeline(coeffs, dictionary, monitor=lambda _t: trace, config=config)
    entry = pipeline.run_turn("hello")
    assert entry["monitor"] is not None
    assert entry["monitor"]["fired"] is False
    assert entry["monitor"]["n_tokens"] == 2


def test_monitor_skipped_when_disabled(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    trace = TokenTrace(
        scores=(0.1,),
        clipped=(0.1,),
        threshold=1.0,
        clip_sigma=5.0,
        n_above=0,
        fired=False,
        mean=0.1,
        median=0.1,
        std=0.0,
        peak=0.1,
        peak_index=0,
    )
    config = PipelineConfig(k=3, monitor_positions=0)  # disabled
    pipeline = _make_pipeline(coeffs, dictionary, monitor=lambda _t: trace, config=config)
    entry = pipeline.run_turn("hello")
    assert entry["monitor"] is None


# --- CLI ---------------------------------------------------------------------------


def test_cli_mock_smoke(coeffs: ACTCoefficients, dictionary: ACTDictionary, tmp_path) -> None:
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps({"user": "student", "agent": "teacher", "messages": ["hello", "again"]}),
        encoding="utf-8",
    )
    out = tmp_path / "trace.json"
    rc = main(["--mock", "--script", str(script), "--out", str(out), "--k", "2"])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock"
    assert len(data["turns"]) == 2
    for turn in data["turns"]:
        for key in ("u_pre", "u_post", "response", "candidates", "winner_index"):
            assert key in turn


def test_cli_requires_directions_before_model_load(tmp_path) -> None:
    """Real mode exits fast on a missing directions artifact.

    The check runs before the model is loaded, so this test is hermetic:
    it never touches the GPU even on a machine that already has the
    runbook artifacts and model weights cached.
    """
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"messages": ["hello"]}), encoding="utf-8")
    missing = tmp_path / "missing" / "directions.npz"
    with pytest.raises(SystemExit):
        main(
            [
                "--script", str(script),
                "--out", str(tmp_path / "t.json"),
                "--directions", str(missing),
            ]
        )


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "quick.json"
    rc = main(["--quick", "--out", str(out), "--k", "2"])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock"
    assert data["quick"] is True
    assert data["seed"] == DEFAULT_SEED
    assert len(data["turns"]) == 2


def test_cli_requires_script_without_quick(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--mock", "--out", str(tmp_path / "t.json")])


# --- Real-mode wiring (S11.5) ---------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer returning a fixed number of tokens."""

    def __init__(self, n_tokens: int = 3) -> None:
        self.n_tokens = n_tokens

    def __call__(self, text, return_tensors, truncation, max_length):
        import torch

        n = 1 if not text else self.n_tokens
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "attention_mask": torch.ones(1, n),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(1 if not text else self.n_tokens))

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        """A minimal chat template for wiring tests."""
        parts = [f"<|{m['role']}|> {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|> ")
        return "\n".join(parts)


class _FakeModel:
    """Tiny fake causal LM: constant hidden state per token."""

    def __init__(self, hidden_size: int = 4, n_layers: int = 5) -> None:
        self._hidden_size = hidden_size
        self._n_layers = n_layers

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states: bool = False):
        import torch

        n = int(input_ids.shape[1]) if input_ids is not None else 1
        hidden = torch.zeros(1, n, self._hidden_size)
        hidden[:, :, 0] = 1.0

        class _Out:
            hidden_states = (hidden,) * (self._n_layers + 1)
            logits = torch.zeros(1, n, 4)

        return _Out()


class _FakeSteerer:
    """Records calls; always returns a canned continuation."""

    def __init__(self, model, tokenizer, directions, config) -> None:
        self.config = config
        self.layers = list(directions.hidden_layers)
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt, u_pre=None, **kwargs):
        self.calls.append(
            {"method": "generate", "prompt": prompt, "u_pre": u_pre, "kwargs": kwargs}
        )
        return [prompt + " fake response"]

    def generate_plain(self, prompt, **kwargs):
        self.calls.append({"method": "plain", "prompt": prompt, "kwargs": kwargs})
        return [prompt + " fake response"]

    def generate_with_offsets(self, prompt, offsets, **kwargs):
        self.calls.append(
            {"method": "offsets", "prompt": prompt, "offsets": offsets, "kwargs": kwargs}
        )
        return [prompt + " fake response"]


def _synthetic_bundle() -> object:
    from act_steering import directions as d

    directions = {
        "epa.evaluation": {
            2: np.array([1.0, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0, 0.0]),
        },
        "epa.potency": {
            2: np.array([0.0, 0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 0.0, 1.0]),
        },
        "epa.activity": {
            2: np.array([0.6, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 0.8, 0.0, 0.0]),
        },
        "deflection_contrast": {
            2: np.array([1.0, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0, 0.0]),
        },
    }
    return d.DirectionsBundle(
        model_name="test/pipeline-fake",
        hidden_size=4,
        hidden_layers=[2, 3],
        directions=directions,
        seed=42,
    )


def test_build_prompt_first_turn(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import build_prompt
    from act_steering.stimuli import CONTEXT_TEMPLATE

    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    prompt = build_prompt(state)
    assert prompt == CONTEXT_TEMPLATE.format(
        user=state.user_term, agent=state.agent_term, message="hello"
    )


def test_build_prompt_multi_turn(coeffs: ACTCoefficients, dictionary: ACTDictionary) -> None:
    from act_steering.pipeline import build_prompt

    pipeline = _make_pipeline(coeffs, dictionary)
    pipeline.run_turn("hello")
    state = pipeline.state
    state.record_user_message(coeffs, (0.5, 0.0, 0.0), label="again")
    prompt = build_prompt(state)
    assert "You replied:" in prompt
    assert "The person says: again." in prompt
    assert prompt.index("You replied:") < prompt.index("again.")


def test_build_prompt_raises_without_pending(dictionary: ACTDictionary) -> None:
    from act_steering.pipeline import build_prompt

    state = ConversationState.from_ids(dictionary, "student", "teacher")
    with pytest.raises(RuntimeError, match="pending"):
        build_prompt(state)


def test_build_real_components_wiring(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import build_real_components

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    bundle = _synthetic_bundle()
    components = build_real_components(
        model, tokenizer, bundle, PipelineConfig(k=1, mode="static"),
        steering=None, steerer_cls=_FakeSteerer,
    )
    assert components.candidate_generator is not None
    # reader projects the constant hidden state: E = (1.0 + 0.0)/2, etc.
    eps = components.message_reader("some text")
    assert eps.e == pytest.approx(0.5)
    assert eps.p == pytest.approx(0.0)
    assert eps.a == pytest.approx(0.3)
    # generator produces one candidate that continues the prompt
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    candidates = components.candidate_generator(state)
    assert candidates == ["fake response"]


def test_build_real_components_monitor_and_repair(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import build_real_components

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    bundle = _synthetic_bundle()
    config = PipelineConfig(k=1, mode="static", monitor_positions=2, repair="reidentify")
    components = build_real_components(
        model, tokenizer, bundle, config, steerer_cls=_FakeSteerer,
    )
    assert components.monitor is not None
    trace = components.monitor("abcd")
    assert isinstance(trace, TokenTrace)
    assert trace.n_tokens == 2
    assert components.repair_policy is not None
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    repaired = components.repair_policy(state, "some response")
    assert repaired == ["fake response"]


# --- Steer conditions (T3-S2) --------------------------------------------------


def _wired_components(
    dictionary: ACTDictionary,
    coeffs: ACTCoefficients | None = None,
    condition=None,
    prompt_spec=None,
    read_format: str = "raw",
    **kwargs,
):
    """Build real components with the recording fake steerer.

    Returns:
        A ``(components, steerer)`` pair.
    """
    from act_steering.pipeline import build_real_components

    instances: list[_FakeSteerer] = []

    class _RecordingSteerer(_FakeSteerer):
        def __init__(self, model, tokenizer, directions, config) -> None:
            super().__init__(model, tokenizer, directions, config)
            instances.append(self)

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    bundle = _synthetic_bundle()
    components = build_real_components(
        model,
        tokenizer,
        bundle,
        PipelineConfig(k=1, mode="static"),
        steerer_cls=_RecordingSteerer,
        coeffs=coeffs,
        dictionary=dictionary,
        condition=condition,
        prompt_spec=prompt_spec,
        read_format=read_format,
        **kwargs,
    )
    return components, instances[-1]


def _pending_state(dictionary: ACTDictionary, coeffs: ACTCoefficients):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.5, -0.5), label="hello")
    return state


def test_steer_condition_validation() -> None:
    from act_steering.pipeline import SteerCondition

    SteerCondition(name="b0", axis="none")
    SteerCondition(name="m0", axis="deflection")
    SteerCondition(name="m1", axis="epa_target", target_kind="situation")
    SteerCondition(name="b6", axis="epa_target", target_kind="identity_only")
    SteerCondition(
        name="b8",
        axis="epa_target",
        target_kind="constant",
        constant_target=(0.0, 0.5, -1.0),
    )
    SteerCondition(name="b5", axis="random", seed=7)
    with pytest.raises(ValueError):
        SteerCondition(name="x", axis="epa_target")  # missing target kind
    with pytest.raises(ValueError):
        SteerCondition(name="x", axis="epa_target", target_kind="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SteerCondition(name="x", axis="none", target_kind="situation")  # kind needs epa_target axis
    with pytest.raises(ValueError):
        SteerCondition(name="x", axis="epa_target", target_kind="constant")  # missing target
    with pytest.raises(ValueError):
        SteerCondition(name="x", axis="none", scenario_index=-1)


def test_compute_target_bstar_kinds(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.act_core import optimal_response
    from act_steering.pipeline import compute_target_bstar, shuffled_identity_pair

    state = _pending_state(dictionary, coeffs)
    situation = compute_target_bstar(coeffs, state, "situation")
    assert situation == pytest.approx(state.optimal_response_target(coeffs).as_array())
    identity_only = compute_target_bstar(coeffs, state, "identity_only")
    assert identity_only == pytest.approx(
        optimal_response(coeffs, state.user_baseline, state.agent_baseline).as_array()
    )
    constant = compute_target_bstar(
        coeffs, state, "constant", constant_target=(0.1, -0.2, 0.3)
    )
    assert constant == pytest.approx(np.array([0.1, -0.2, 0.3]))

    shuffled = compute_target_bstar(coeffs, state, "shuffled", dictionary=dictionary, seed=7)
    assert np.all(np.isfinite(shuffled))
    # deterministic for a fixed seed and scenario index
    again = compute_target_bstar(coeffs, state, "shuffled", dictionary=dictionary, seed=7)
    assert again == pytest.approx(shuffled)
    # the pair draw is deterministic and the scenario index moves it
    p0 = shuffled_identity_pair(dictionary, 7, 0)
    assert p0 == shuffled_identity_pair(dictionary, 7, 0)
    assert p0[0].term != p0[1].term
    pairs = {
        (a.term, b.term) for a, b in
        (shuffled_identity_pair(dictionary, 7, i) for i in range(8))
    }
    assert len(pairs) >= 2

    with pytest.raises(ValueError):
        compute_target_bstar(coeffs, state, "shuffled")
    with pytest.raises(ValueError):
        compute_target_bstar(coeffs, state, "constant")
    with pytest.raises(ValueError):
        compute_target_bstar(coeffs, state, "bogus")  # type: ignore[arg-type]


def test_random_direction_deterministic_unit() -> None:
    from act_steering.pipeline import random_direction

    v = random_direction(16, seed=3, scenario_index=1)
    assert v.shape == (16,)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v == pytest.approx(random_direction(16, seed=3, scenario_index=1))
    assert random_direction(16, seed=3, scenario_index=1) != pytest.approx(
        random_direction(16, seed=3, scenario_index=2), abs=1e-6
    )
    assert random_direction(16, seed=4, scenario_index=1) != pytest.approx(
        random_direction(16, seed=3, scenario_index=1), abs=1e-6
    )
    with pytest.raises(ValueError):
        random_direction(0, seed=3)


def test_condition_none_generates_plain(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import SteerCondition

    components, steerer = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(name="b0", axis="none"),
    )
    state = _pending_state(dictionary, coeffs)
    candidates = components.candidate_generator(state)
    assert candidates == ["fake response"]
    assert steerer.calls[-1]["method"] == "plain"


def test_condition_deflection_is_default(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    components, steerer = _wired_components(dictionary)
    state = _pending_state(dictionary, coeffs)
    components.candidate_generator(state)
    call = steerer.calls[-1]
    assert call["method"] == "generate"
    assert call["kwargs"].get("mode") == "deflection"
    assert "target" not in call["kwargs"]
    # the pending message's event-1 deflection drives the gain
    assert call["u_pre"] == pytest.approx(state._pending_message[2])


def test_condition_epa_target_situation_passes_bstar(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import SteerCondition

    components, steerer = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(name="m1", axis="epa_target", target_kind="situation"),
    )
    state = _pending_state(dictionary, coeffs)
    components.candidate_generator(state)
    call = steerer.calls[-1]
    assert call["method"] == "generate"
    assert call["kwargs"].get("mode") == "epa_target"
    target = call["kwargs"].get("target")
    assert target is not None
    assert np.asarray(target) == pytest.approx(
        state.optimal_response_target(coeffs).as_array()
    )


def test_condition_epa_target_constant(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import SteerCondition

    components, steerer = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(
            name="b8", axis="epa_target", target_kind="constant",
            constant_target=(0.25, -0.5, 1.0),
        ),
    )
    state = _pending_state(dictionary, coeffs)
    components.candidate_generator(state)
    target = steerer.calls[-1]["kwargs"].get("target")
    assert np.asarray(target) == pytest.approx(np.array([0.25, -0.5, 1.0]))


def test_condition_random_offsets_norm_and_seed(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import SteerCondition, random_direction
    from act_steering.steerer import proportional_gain

    components, steerer = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(name="b5", axis="random", seed=11),
    )
    state = _pending_state(dictionary, coeffs)
    components.candidate_generator(state)
    call = steerer.calls[-1]
    assert call["method"] == "offsets"
    offsets = call["offsets"]
    assert set(offsets) == {2, 3}
    u_pre = state._pending_message[2]
    gain = proportional_gain(u_pre, steerer.config)
    expected = gain * random_direction(4, seed=11, scenario_index=0)
    # seeded per scenario: one shared direction at every layer, norm = gain
    for vec in offsets.values():
        assert np.asarray(vec) == pytest.approx(expected)
    assert np.linalg.norm(expected) == pytest.approx(abs(gain))
    # a different scenario index draws a different direction
    components2, steerer2 = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(name="b5b", axis="random", seed=11, scenario_index=1),
    )
    state2 = _pending_state(dictionary, coeffs)
    components2.candidate_generator(state2)
    other = np.asarray(steerer2.calls[-1]["offsets"][2])
    assert other != pytest.approx(np.asarray(offsets[2]), abs=1e-6)


def test_condition_act_descriptors_prompt_uses_situation_bstar(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import (
        PromptSpec,
        SteerCondition,
        descriptors_for,
    )

    components, steerer = _wired_components(
        dictionary,
        coeffs=coeffs,
        condition=SteerCondition(name="b1", axis="none"),
        prompt_spec=PromptSpec(style="chat", variant="act_descriptors"),
    )
    state = _pending_state(dictionary, coeffs)
    components.candidate_generator(state)
    prompt = steerer.calls[-1]["prompt"]
    bstar = state.optimal_response_target(coeffs)
    expected_phrase = descriptors_for(bstar)
    assert expected_phrase in prompt
    # the chat template was applied (not the raw flat template)
    assert "<|system|>" in prompt and "<|assistant|>" in prompt


def test_build_real_components_requires_args(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    from act_steering.pipeline import PromptSpec, SteerCondition, build_real_components

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    bundle = _synthetic_bundle()
    config = PipelineConfig(k=1, mode="static")
    # epa_target axis without coeffs
    with pytest.raises(ValueError, match="coeffs"):
        build_real_components(
            model, tokenizer, bundle, config, steerer_cls=_FakeSteerer,
            condition=SteerCondition(name="m1", axis="epa_target", target_kind="situation"),
        )
    # shuffled target without the dictionary
    with pytest.raises(ValueError, match="dictionary"):
        build_real_components(
            model, tokenizer, bundle, config, steerer_cls=_FakeSteerer,
            coeffs=coeffs,
            condition=SteerCondition(name="b7", axis="epa_target", target_kind="shuffled"),
        )
    # act_descriptors prompt without coeffs
    with pytest.raises(ValueError, match="coeffs"):
        build_real_components(
            model, tokenizer, bundle, config, steerer_cls=_FakeSteerer,
            prompt_spec=PromptSpec(style="chat", variant="act_descriptors"),
        )
