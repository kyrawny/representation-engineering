"""Tests for ACT impression formation and deflection math."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from act_steering.act_core import (
    EPA,
    ACTCoefficients,
    optimal_response,
    optimal_response_behaviour,
    two_event_state_update,
)
from act_steering.dictionary import ACTDictionary, EPATerm

_OUTPUT_SLOTS = (
    "postAE",
    "postAP",
    "postAA",
    "postBE",
    "postBP",
    "postBA",
    "postOE",
    "postOP",
    "postOA",
)


def _epa(term: EPATerm) -> EPA:
    """Convert a dictionary term into an EPA vector."""
    return EPA(*term.as_vector())


@pytest.fixture(scope="module")
def coeffs(data_dir: Path) -> ACTCoefficients:
    """Load the shipped 15-term coefficient set."""
    return ACTCoefficients.load(data_dir / "2010impressionformation.csv")


@pytest.fixture(scope="module")
def dictionary(data_dir: Path) -> ACTDictionary:
    """Load the MTurk-Interact dictionaries."""
    return ACTDictionary.load(data_dir)


def _raw_matrix(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-read the raw CSV for independent recomputation in tests."""
    with (data_dir / "2010impressionformation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [[float(row[slot]) for slot in _OUTPUT_SLOTS] for row in rows]
    matrix = np.array(values, dtype=np.float64)
    masks = np.array([int(row["coef_name"][1:], 2) for row in rows], dtype=np.int64)
    return matrix, masks


def _raw_impressions(
    matrix: np.ndarray, masks: np.ndarray, s: np.ndarray, b: np.ndarray, o: np.ndarray
) -> np.ndarray:
    """Independent impression recomputation straight from the CSV."""
    basis = np.concatenate([s, b, o])
    terms = np.ones(len(masks))
    for bit in range(9):
        selected = (masks >> bit) & 1
        terms[selected.astype(bool)] *= basis[bit]
    return terms @ matrix


def test_coefficients_load_shape(coeffs: ACTCoefficients) -> None:
    """The shipped set has 15 terms x 9 slots and is linear in behaviour."""
    assert coeffs.matrix.shape == (15, 9)
    assert coeffs.masks.shape == (15,)
    assert 0 in coeffs.masks  # constant term
    assert coeffs.linear_in_behavior


def test_impressions_at_zero_equal_constant_row(coeffs: ACTCoefficients) -> None:
    """W(0, 0, 0) must equal the constant coefficient row."""
    zero = np.zeros(3)
    assert np.allclose(coeffs.impressions(zero, zero, zero), coeffs.matrix[0])


def test_impressions_match_raw_csv_recomputation(
    coeffs: ACTCoefficients, data_dir: Path
) -> None:
    """The module pipeline must match an independent recomputation from the CSV."""
    matrix, masks = _raw_matrix(data_dir)
    rng = np.random.default_rng(0)
    for _ in range(25):
        s = rng.uniform(-3, 3, 3)
        b = rng.uniform(-3, 3, 3)
        o = rng.uniform(-3, 3, 3)
        expected = _raw_impressions(matrix, masks, s, b, o)
        assert np.allclose(coeffs.impressions(s, b, o), expected, atol=1e-12)


def test_deflection_is_squared_distance(
    coeffs: ACTCoefficients, data_dir: Path
) -> None:
    """Deflection must equal the squared distance W - [s; b; o]."""
    matrix, masks = _raw_matrix(data_dir)
    rng = np.random.default_rng(1)
    for _ in range(25):
        s = rng.uniform(-3, 3, 3)
        b = rng.uniform(-3, 3, 3)
        o = rng.uniform(-3, 3, 3)
        w = _raw_impressions(matrix, masks, s, b, o)
        f = np.concatenate([s, b, o])
        expected = float(np.sum((w - f) ** 2))
        assert coeffs.deflection(s, b, o) == pytest.approx(expected, abs=1e-12)


def test_deflection_positive_for_real_scenario(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    s = _epa(dictionary.require("27_year_old")).as_array()
    b = _epa(dictionary.require("abandon")).as_array()
    o = _epa(dictionary.require("teacher")).as_array()
    assert coeffs.deflection(s, b, o) > 0


def test_epa_roundtrip_and_distance() -> None:
    epa = EPA(0.5, -1.2, 2.0)
    assert EPA.from_array(epa.as_array()) == epa
    assert EPA(1.0, 0.0, 0.0).distance_to(EPA(0.0, 1.0, 0.0)) == pytest.approx(np.sqrt(2.0))
    with pytest.raises(ValueError):
        EPA.from_array(np.zeros(2))


def test_behavior_affine_reproduces_impressions(coeffs: ACTCoefficients) -> None:
    """W = c + J b must hold exactly for random inputs."""
    rng = np.random.default_rng(3)
    for _ in range(25):
        s = rng.uniform(-3, 3, 3)
        b = rng.uniform(-3, 3, 3)
        o = rng.uniform(-3, 3, 3)
        c, jacobian = coeffs.behavior_affine(s, o)
        assert np.allclose(c + jacobian @ b, coeffs.impressions(s, b, o), atol=1e-12)


def _find_interior_scenario(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> tuple[EPA, EPA, EPA]:
    """Find an (actor, object) pair whose unconstrained b* lies inside the box."""
    actors = dictionary.sample(40, seed=0, collection="identities")
    objects = dictionary.sample(40, seed=1, collection="identities")
    for actor in actors:
        for object_ in objects:
            b = optimal_response_behaviour(
                coeffs,
                actor=_epa(actor),
                object_=_epa(object_),
                bounds=(-1e9, 1e9),
            )
            if np.all(np.abs(b.as_array()) < 4.2):
                return _epa(actor), _epa(object_), b
    raise AssertionError("no interior scenario found")


def test_closed_form_matches_lbfgsb_within_1e_6(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    """Interior closed-form solution and L-BFGS-B must agree below 1e-6."""
    actor, object_, b_closed = _find_interior_scenario(coeffs, dictionary)
    s, o = actor.as_array(), object_.as_array()
    c, jacobian = coeffs.behavior_affine(s, o)
    a_matrix = jacobian.copy()
    a_matrix[3:6, :] -= np.eye(3)
    d = np.concatenate([s, np.zeros(3), o]) - c

    def objective(x: np.ndarray) -> float:
        residual = a_matrix @ x - d
        return float(residual @ residual)

    result = minimize(
        objective,
        x0=b_closed.as_array(),
        method="L-BFGS-B",
        bounds=[(-4.3, 4.3)] * 3,
        tol=1e-14,
    )
    assert np.max(np.abs(result.x - b_closed.as_array())) < 1e-6


def test_optimal_response_is_global_box_minimum(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    """The returned b* must beat thousands of random in-box candidates."""
    user = _epa(dictionary.require("boss"))
    agent = _epa(dictionary.require("subordinate"))
    b = optimal_response(coeffs, user=user, agent=agent)
    s, o = agent.as_array(), user.as_array()
    u_opt = coeffs.deflection(s, b.as_array(), o)
    rng = np.random.default_rng(2)
    candidates = rng.uniform(-4.3, 4.3, size=(5000, 3))
    u_best_random = min(coeffs.deflection(s, cand, o) for cand in candidates)
    assert u_opt <= u_best_random + 1e-6
    assert np.all(b.as_array() >= -4.3 - 1e-9)
    assert np.all(b.as_array() <= 4.3 + 1e-9)


def test_two_event_targets_identity_dependent(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    """b* must depend on the identity pair (the situation-dependence claim)."""
    pairs = (
        ("boss", "subordinate"),
        ("subordinate", "boss"),
        ("stranger", "stranger"),
        ("teacher", "student"),
    )
    def _target(u: str, a: str) -> np.ndarray:
        user = _epa(dictionary.require(u))
        agent = _epa(dictionary.require(a))
        return optimal_response(coeffs, user=user, agent=agent).as_array()

    targets = np.array([_target(u, a) for u, a in pairs])
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            assert np.linalg.norm(targets[i] - targets[j]) > 0.05


def test_two_event_state_update(coeffs: ACTCoefficients, dictionary: ACTDictionary) -> None:
    """Event 1 must produce finite post-event fundamentals and a positive U."""
    user = _epa(dictionary.require("27_year_old"))
    agent = _epa(dictionary.require("teacher"))
    message = _epa(dictionary.require("abandon"))  # placeholder behaviour EPA
    user_after, agent_after, u1 = two_event_state_update(coeffs, user, agent, message)
    assert u1 > 0
    assert np.all(np.isfinite(user_after.as_array()))
    assert np.all(np.isfinite(agent_after.as_array()))
    again = two_event_state_update(coeffs, user, agent, message)
    assert again[0] == user_after
    assert again[1] == agent_after
