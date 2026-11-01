"""Tests for act_steering.stimuli (trial simulation + contrast sets)."""

from __future__ import annotations

import numpy as np
import pytest

from act_steering.act_core import ACTCoefficients
from act_steering.dictionary import ACTDictionary
from act_steering.stimuli import (
    RESPONSE_TEMPLATE,
    build_contrast_set,
    pairs_to_dicts,
    simulate_trials,
)

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


def _trial_keys(trials) -> list[tuple[str, str, str]]:
    return [(t.user_term, t.agent_term, t.message_term) for t in trials]


def test_simulate_trials_reproducible(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    a = simulate_trials(coeffs, dictionary, 30, seed=5)
    b = simulate_trials(coeffs, dictionary, 30, seed=5)
    assert _trial_keys(a) == _trial_keys(b)
    c = simulate_trials(coeffs, dictionary, 30, seed=6)
    assert _trial_keys(a) != _trial_keys(c)


def test_trial_fields_sane(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    trials = simulate_trials(coeffs, dictionary, 20, seed=1)
    assert len(trials) == 20
    for t in trials:
        assert t.user_term != t.agent_term
        assert t.event1_deflection >= 0.0
        for values in (t.message_epa, t.user_after, t.agent_after):
            assert all(np.isfinite(v) for v in values)
        assert dictionary.lookup(t.user_term) is not None
        assert dictionary.lookup(t.agent_term) is not None
        assert dictionary.lookup(t.message_term) is not None


def test_low_group_mean_u_below_high(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    pairs = build_contrast_set(coeffs, dictionary, 50, seed=7, candidates_per_trial=100)
    assert len(pairs) == 50
    mean_low = float(np.mean([p.low.deflection for p in pairs]))
    mean_high = float(np.mean([p.high.deflection for p in pairs]))
    assert mean_low < mean_high


def test_pairs_fixed_sizes_and_structure(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    pairs = build_contrast_set(
        coeffs, dictionary, 25, seed=9, candidates_per_trial=80, min_delta=0.05
    )
    assert len(pairs) == 25
    for p in pairs:
        # Context invariance across groups: both sides share the trial context.
        assert p.low.context == p.high.context
        assert p.low.user_term == p.high.user_term
        assert p.low.agent_term == p.high.agent_term
        assert p.low.message_term == p.high.message_term
        # Only the response behaviour differs.
        assert p.low.behaviour_term != p.high.behaviour_term
        # Delta is the deflection separation.
        assert p.delta_u == pytest.approx(p.high.deflection - p.low.deflection)
        assert p.delta_u >= 0.05


def _pair_keys(pairs) -> list[tuple[str, str]]:
    return [(p.low.behaviour_term, p.high.behaviour_term) for p in pairs]


def test_pairs_seed_reproducible(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    a = build_contrast_set(coeffs, dictionary, 20, seed=11, candidates_per_trial=60)
    b = build_contrast_set(coeffs, dictionary, 20, seed=11, candidates_per_trial=60)
    assert _pair_keys(a) == _pair_keys(b)
    c = build_contrast_set(coeffs, dictionary, 20, seed=12, candidates_per_trial=60)
    assert _pair_keys(a) != _pair_keys(c)


def test_behaviour_epa_matches_dictionary(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    pairs = build_contrast_set(coeffs, dictionary, 10, seed=3, candidates_per_trial=50)
    for p in pairs:
        assert p.low.behaviour_epa == dictionary.require(p.low.behaviour_term).as_vector()
        assert p.high.behaviour_epa == dictionary.require(p.high.behaviour_term).as_vector()


def test_context_contains_all_terms(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    pairs = build_contrast_set(coeffs, dictionary, 5, seed=2, candidates_per_trial=50)
    for p in pairs:
        for term in (p.low.user_term, p.low.agent_term, p.low.message_term):
            assert term in p.low.context
        # Context is invariant; the response text is built separately.
        assert p.low.context == p.high.context
        assert RESPONSE_TEMPLATE.format(behaviour=p.low.behaviour_term)
        assert RESPONSE_TEMPLATE.format(behaviour=p.high.behaviour_term)


def test_invalid_quantiles_raise(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    with pytest.raises(ValueError):
        build_contrast_set(coeffs, dictionary, 5, seed=1, low_quantile=0.9)


def test_pairs_serialize(coeffs: ACTCoefficients, dictionary: ACTDictionary):
    pairs = build_contrast_set(coeffs, dictionary, 3, seed=4, candidates_per_trial=50)
    dicts_ = pairs_to_dicts(pairs)
    assert len(dicts_) == 3
    assert set(dicts_[0]) == {"low", "high", "delta_u"}
    assert "behaviour_term" in dicts_[0]["low"]
