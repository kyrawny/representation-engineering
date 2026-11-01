"""Tests for scenario battery generation and the leakage check (T3-S3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from act_steering.act_core import ACTCoefficients
from act_steering.battery import (
    assert_zero_overlap,
    contrast_trial_keys,
    generate_battery,
    trial_key,
    write_battery,
)
from act_steering.dictionary import ACTDictionary, EPATerm
from act_steering.stimuli import Trial


def _small_dictionary() -> ACTDictionary:
    """Six identities and four behaviours: small enough for exact tests."""
    identities = [
        EPATerm(term=f"id_{i}", e=float(i - 2), p=float(i - 3), a=float(i - 4)) for i in range(6)
    ]
    behaviors = [
        EPATerm(term=f"beh_{i}", e=float(i - 1), p=float(i - 2), a=float(i - 3)) for i in range(4)
    ]
    return ACTDictionary(identities=identities, behaviors=behaviors, modifiers=[])


def _synthetic_coeffs() -> ACTCoefficients:
    """Identity impressions plus one extra bE term (U = (2*bE)^2)."""
    matrix = np.zeros((15, 9))
    masks = np.zeros(15, dtype=np.int64)
    for j in range(9):
        masks[j] = 1 << j
        matrix[j, j] = 1.0
    masks[9] = 1 << 3
    matrix[9, 3] = 2.0
    return ACTCoefficients(matrix=matrix, masks=masks)


def _trial(user: str, agent: str, message: str) -> Trial:
    return Trial(
        user_term=user,
        agent_term=agent,
        message_term=message,
        message_epa=(0.0, 0.0, 0.0),
        user_after=(0.0, 0.0, 0.0),
        agent_after=(0.0, 0.0, 0.0),
        event1_deflection=0.0,
    )


def test_generate_battery_unique_and_deterministic() -> None:
    dictionary = _small_dictionary()
    coeffs = _synthetic_coeffs()
    battery = generate_battery(coeffs, dictionary, n_scenarios=10, seed=7)
    assert len(battery) == 10
    keys = [trial_key(t) for t in battery]
    assert len(set(keys)) == 10
    again = generate_battery(coeffs, dictionary, n_scenarios=10, seed=7)
    assert [trial_key(t) for t in again] == keys


def test_generate_battery_excludes_contrast_keys() -> None:
    dictionary = _small_dictionary()
    coeffs = _synthetic_coeffs()
    first = generate_battery(coeffs, dictionary, n_scenarios=4, seed=3)
    excluded = {trial_key(first[0])}
    battery = generate_battery(coeffs, dictionary, n_scenarios=4, seed=3, contrast_keys=excluded)
    assert trial_key(first[0]) not in {trial_key(t) for t in battery}
    assert len({trial_key(t) for t in battery}) == 4


def test_generate_battery_validation() -> None:
    dictionary = _small_dictionary()
    coeffs = _synthetic_coeffs()
    with pytest.raises(ValueError):
        generate_battery(coeffs, dictionary, n_scenarios=0, seed=1)
    with pytest.raises(ValueError):
        generate_battery(coeffs, dictionary, n_scenarios=1, seed=0)
    # Degenerate dictionary: one context total, and it is excluded.
    tiny = ACTDictionary(
        identities=[
            EPATerm(term="a", e=0.0, p=0.0, a=0.0),
            EPATerm(term="b", e=1.0, p=1.0, a=1.0),
        ],
        behaviors=[EPATerm(term="x", e=0.0, p=0.0, a=0.0)],
        modifiers=[],
    )
    all_keys = {
        (t.user_term, t.agent_term, t.message_term)
        for t in generate_battery(coeffs, tiny, n_scenarios=2, seed=5)
    }
    with pytest.raises(ValueError, match="could not fill"):
        generate_battery(
            coeffs,
            tiny,
            n_scenarios=2,
            seed=5,
            contrast_keys=all_keys,
            max_attempts_per_scenario=5,
        )


def test_assert_zero_overlap() -> None:
    battery = [_trial("u1", "a1", "m1"), _trial("u2", "a2", "m2")]
    assert_zero_overlap(battery, {("u3", "a3", "m3")})
    with pytest.raises(AssertionError, match="leakage"):
        assert_zero_overlap(battery, {("u1", "a1", "m1")})


def test_contrast_trial_keys(tmp_path) -> None:
    contrast = {
        "seed": 42,
        "pairs": [
            {
                "low": {
                    "user_term": "u1",
                    "agent_term": "a1",
                    "message_term": "m1",
                },
                "high": {
                    "user_term": "u1",
                    "agent_term": "a1",
                    "message_term": "m1",
                },
            },
            {
                "low": {
                    "user_term": "u2",
                    "agent_term": "a2",
                    "message_term": "m2",
                },
                "high": {
                    "user_term": "u2",
                    "agent_term": "a2",
                    "message_term": "m2",
                },
            },
        ],
    }
    path = tmp_path / "contrast.json"
    path.write_text(json.dumps(contrast), encoding="utf-8")
    keys = contrast_trial_keys(path)
    assert keys == {("u1", "a1", "m1"), ("u2", "a2", "m2")}


def test_write_battery_roundtrip(tmp_path) -> None:
    contrast_path = tmp_path / "contrast.json"
    contrast_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "pairs": [
                    {
                        "low": {
                            "user_term": "cu",
                            "agent_term": "ca",
                            "message_term": "cm",
                        },
                        "high": {
                            "user_term": "cu",
                            "agent_term": "ca",
                            "message_term": "cm",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dictionary = _small_dictionary()
    coeffs = _synthetic_coeffs()
    battery = generate_battery(
        coeffs,
        dictionary,
        n_scenarios=5,
        seed=11,
        contrast_keys=contrast_trial_keys(contrast_path),
    )
    out = tmp_path / "battery.json"
    summary = write_battery(out, battery, seed=11, contrast_path=contrast_path)
    assert summary["n_scenarios"] == 5
    assert summary["contrast_overlap"] == 0
    assert summary["n_distinct_trial_contexts"] == 5

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["seed"] == 11
    assert data["n_scenarios"] == 5
    assert len(data["scenarios"]) == 5
    for i, scenario in enumerate(data["scenarios"]):
        assert scenario["index"] == i
        assert "context" in scenario
        assert scenario["user_term"] in scenario["context"]
        assert scenario["agent_term"] in scenario["context"]
        assert scenario["message_term"] in scenario["context"]
