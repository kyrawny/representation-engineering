"""Tests for act_steering.state (multi-turn ACT conversation state)."""

from __future__ import annotations

import numpy as np
import pytest

from act_steering.act_core import EPA, ACTCoefficients, optimal_response, two_event_state_update
from act_steering.dictionary import ACTDictionary
from act_steering.state import ConversationState, StateConfig

COEFFS_PATH = (
    r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act\2010impressionformation.csv"
)
DATA_DIR = r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act"


@pytest.fixture(scope="module")
def coeffs() -> ACTCoefficients:
    return ACTCoefficients.load(COEFFS_PATH)


@pytest.fixture(scope="module")
def dictionary() -> ACTDictionary:
    return ACTDictionary.load(DATA_DIR)


def _epa(values) -> EPA:
    return EPA.from_array(np.asarray(values, dtype=np.float64))


def test_initial_state_from_ids(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    assert state.turn == 0
    assert state.deflection_trace == []
    assert not state.trigger_fired
    assert state.unresolved_streak == 0
    user_expected = dictionary.require("student").as_vector()
    agent_expected = dictionary.require("teacher").as_vector()
    assert np.allclose(state.user_fundamentals.as_array(), user_expected)
    assert np.allclose(state.agent_fundamentals.as_array(), agent_expected)


def test_user_message_updates_fundamentals(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    user0, agent0 = state.user_fundamentals, state.agent_fundamentals
    message = dictionary.behaviors[3]
    u1 = state.record_user_message(coeffs, message.as_vector(), label=message.term)
    user_after, agent_after, u1_expected = two_event_state_update(
        coeffs, user0, agent0, _epa(message.as_vector())
    )
    assert u1 == pytest.approx(u1_expected)
    assert np.allclose(state.user_fundamentals.as_array(), user_after.as_array())
    assert np.allclose(state.agent_fundamentals.as_array(), agent_after.as_array())


def test_full_turn_trace_and_fundamentals(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    message, response = dictionary.behaviors[5], dictionary.behaviors[9]
    user0, agent0 = state.user_fundamentals, state.agent_fundamentals
    state.record_user_message(coeffs, message.as_vector(), message.term)
    b_star_expected = optimal_response(coeffs, state.user_fundamentals, state.agent_fundamentals)
    u2 = state.record_agent_response(coeffs, response.as_vector(), response.term)
    assert state.turn == 1
    assert len(state.deflection_trace) == 1
    entry = state.deflection_trace[0]
    assert entry["turn"] == 0
    assert entry["user_message"] == message.term
    assert entry["response"] == response.term
    assert np.allclose(entry["b_star"], b_star_expected.as_array())
    assert entry["u1"] >= 0.0 and entry["u2"] >= 0.0
    assert u2 == pytest.approx(entry["u2"])
    # Event-2 fundamentals checked against a manual impression evaluation.
    user_after, agent_after, _ = two_event_state_update(
        coeffs, user0, agent0, _epa(message.as_vector())
    )
    w = coeffs.impressions(agent_after, _epa(response.as_vector()), user_after)
    assert np.allclose(state.agent_fundamentals.as_array(), w[0:3])
    assert np.allclose(state.user_fundamentals.as_array(), w[6:9])


def test_b_star_target_matches_act_core(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, dictionary.behaviors[2].as_vector())
    expected = optimal_response(coeffs, state.user_fundamentals, state.agent_fundamentals)
    assert np.allclose(state.optimal_response_target(coeffs).as_array(), expected.as_array())


def test_drift_rule_pulls_toward_baseline(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    message, response = dictionary.behaviors[7], dictionary.behaviors[11]
    plain = ConversationState.from_ids(dictionary, "student", "teacher", StateConfig())
    drifted = ConversationState.from_ids(
        dictionary, "student", "teacher", StateConfig(drift_decay=0.5)
    )
    for s in (plain, drifted):
        s.record_user_message(coeffs, message.as_vector())
        s.record_agent_response(coeffs, response.as_vector())
    for role in ("user", "agent"):
        base = getattr(drifted, f"{role}_baseline")
        plain_f = getattr(plain, f"{role}_fundamentals")
        drifted_f = getattr(drifted, f"{role}_fundamentals")
        expected = base.as_array() + 0.5 * (plain_f.as_array() - base.as_array())
        assert np.allclose(drifted_f.as_array(), expected)


def test_trigger_fires_on_sustained_unresolved(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    config = StateConfig(deflection_threshold=0.0, sustained_turns=3)
    state = ConversationState.from_ids(dictionary, "student", "teacher", config)
    for i, idx in enumerate((1, 4, 8, 13)):
        state.record_user_message(coeffs, dictionary.behaviors[idx].as_vector())
        if i < 2:
            assert not state.trigger_fired
    assert state.trigger_fired
    assert state.trigger_turn == 3
    assert state.unresolved_streak == 4
    # Latched: later messages do not change the fire turn.
    state.record_user_message(coeffs, dictionary.behaviors[2].as_vector())
    assert state.trigger_turn == 3


def test_trigger_never_fires_on_low_deflection(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    config = StateConfig(deflection_threshold=1e12, sustained_turns=2)
    state = ConversationState.from_ids(dictionary, "student", "teacher", config)
    for idx in (1, 2, 3):
        state.record_user_message(coeffs, dictionary.behaviors[idx].as_vector())
        assert state.unresolved_streak == 0
    assert not state.trigger_fired


def test_trace_accumulation_over_turns(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    for i in range(3):
        state.record_user_message(coeffs, dictionary.behaviors[i].as_vector(), f"m{i}")
        state.record_agent_response(coeffs, dictionary.behaviors[i + 20].as_vector(), f"r{i}")
    assert [e["turn"] for e in state.deflection_trace] == [0, 1, 2]
    assert state.turn == 3
    for entry in state.deflection_trace:
        assert entry["u1"] >= 0.0 and entry["u2"] >= 0.0
        assert len(entry["b_star"]) == 3
        assert all(np.isfinite(v) for v in entry["b_star"])


def test_response_without_message_raises(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    with pytest.raises(RuntimeError):
        state.record_agent_response(coeffs, (0.5, 0.5, 0.5))


def test_config_validation():
    with pytest.raises(ValueError):
        StateConfig(drift_decay=0.0)
    with pytest.raises(ValueError):
        StateConfig(drift_decay=1.5)
    with pytest.raises(ValueError):
        StateConfig(deflection_threshold=-1.0)
    with pytest.raises(ValueError):
        StateConfig(sustained_turns=0)
