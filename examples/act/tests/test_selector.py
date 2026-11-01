"""CPU mock tests for ACT best-of-k candidate selection (S9).

The synthetic coefficient set makes the deflection analytic
(``U = bE^2``), so argmin and tie-break behaviour are verified against
hand-computed values. A second group uses the real coefficient CSV with
the conversation state for realism.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from act_steering.act_core import EPA, ACTCoefficients
from act_steering.dictionary import ACTDictionary
from act_steering.selector import BestOfKSelector, select_best_of_k
from act_steering.state import ConversationState

COEFFS_PATH = (
    r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act\2010impressionformation.csv"
)
DATA_DIR = r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act"


def _synthetic_coeffs() -> ACTCoefficients:
    """Identity impressions plus one extra bE term: U = (2*bE)^2 exactly.

    Rows 0-8 reproduce the basis slots (W = [s; b; o] exactly); row 9
    adds 2*bE to the behaviour-E slot (which already holds bE), so W[3]
    = 3*bE against the reference bE, and the only nonzero deflection
    contribution is (3*bE - bE)^2 = 4*bE^2.
    """
    matrix = np.zeros((15, 9))
    masks = np.zeros(15, dtype=np.int64)
    for j in range(9):
        masks[j] = 1 << j
        matrix[j, j] = 1.0
    masks[9] = 1 << 3
    matrix[9, 3] = 2.0
    return ACTCoefficients(matrix=matrix, masks=masks)


@pytest.fixture(scope="module")
def coeffs() -> ACTCoefficients:
    return ACTCoefficients.load(COEFFS_PATH)


@pytest.fixture(scope="module")
def dictionary() -> ACTDictionary:
    return ACTDictionary.load(DATA_DIR)


# --- pure selection: argmin and tie-breaking ---------------------------------


def test_argmin_picks_lowest_deflection() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(1.0, 0.5, -0.2),
        object_=(0.3, 0.1, 0.9),
        candidates=["a", "b", "c"],
        behaviours=[(0.5, 0.0, 0.0), (0.0, 1.0, 1.0), (2.0, 0.0, 0.0)],
    )
    assert [s.deflection for s in sel.scores] == pytest.approx([1.0, 0.0, 16.0])
    assert sel.winner_index == 1
    assert sel.winner_text == "b"
    assert sel.n_scorable == 3
    assert sel.degenerate is False
    assert sel.all_tied is False


def test_tie_break_prefers_earlier_index() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["first", "second"],
        behaviours=[(0.51, 0.0, 0.0), (0.5, 0.0, 0.0)],  # U = 1.0404, 1.0
        tie_tolerance=0.05,  # difference 0.0404 is within tolerance
    )
    assert sel.winner_index == 0  # earliest index wins the tie
    assert sel.all_tied is True


def test_below_tolerance_uses_true_argmin() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["first", "second"],
        behaviours=[(0.51, 0.0, 0.0), (0.5, 0.0, 0.0)],
        tie_tolerance=0.03,  # difference 0.0404 is above tolerance
    )
    assert sel.winner_index == 1
    assert sel.all_tied is False


def test_k1_degenerate_selection() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["only"],
        behaviours=[(0.7, 0.0, 0.0)],
    )
    assert sel.degenerate is True
    assert sel.winner_index == 0
    assert sel.winner.deflection == pytest.approx(1.96)


def test_total_failure_fallback() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["a", "b"],
        behaviours=[None, None],
    )
    assert sel.winner_index is None
    assert sel.winner is None
    assert sel.n_scorable == 0
    assert "all 2 candidates" in sel.fallback_reason


def test_partial_failure_scores_remaining() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["a", "b", "c"],
        behaviours=[None, (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    assert sel.n_scorable == 2
    assert sel.winner_index == 1
    assert "no behaviour EPA" in sel.scores[0].error
    assert sel.scores[0].deflection is None


def test_malformed_behaviour_marked_unscorable() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["bad"],
        behaviours=[(1.0,)],
    )
    assert sel.winner_index is None
    assert "shape (3,)" in sel.scores[0].error


def test_nonfinite_deflection_marked_unscorable() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["inf"],
        behaviours=[(np.inf, 0.0, 0.0)],
    )
    assert sel.winner_index is None
    assert "not finite" in sel.scores[0].error


def test_validation_errors() -> None:
    coeffs = _synthetic_coeffs()
    with pytest.raises(ValueError, match="non-empty"):
        select_best_of_k(coeffs, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [], [])
    with pytest.raises(ValueError, match="same length"):
        select_best_of_k(
            coeffs, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ["a", "b"], [(0.0, 0.0, 0.0)]
        )
    with pytest.raises(ValueError, match="tie_tolerance"):
        select_best_of_k(
            coeffs, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ["a"], [(0.0, 0.0, 0.0)],
            tie_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="actor"):
        select_best_of_k(coeffs, (0.0,), (0.0, 0.0, 0.0), ["a"], [(0.0, 0.0, 0.0)])


def test_selection_to_dict_is_json_serialisable() -> None:
    sel = select_best_of_k(
        _synthetic_coeffs(),
        actor=(0.0, 0.0, 0.0),
        object_=(0.0, 0.0, 0.0),
        candidates=["a", "bad"],
        behaviours=[(0.5, 0.0, 0.0), None],
    )
    data = json.loads(json.dumps(sel.to_dict()))
    assert data["winner_index"] == 0
    assert data["n_scorable"] == 1
    assert len(data["scores"]) == 2
    assert data["scores"][1]["deflection"] is None
    assert data["fallback_reason"] is None


# --- selector with state, reader and generator --------------------------------


def _reader(table: dict[str, tuple[float, float, float]]):
    def reader(text: str):
        return table[text]

    return reader


TABLE = {"r0": (1.0, 0.0, 0.0), "r1": (-1.0, 0.5, 0.2), "r2": (0.0, -0.3, 0.9)}


def _state_with_pending_message(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> ConversationState:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    message = dictionary.behaviors[3]
    state.record_user_message(coeffs, message.as_vector(), label=message.term)
    return state


def test_selector_with_state_and_real_coeffs(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(coeffs, reader=_reader(TABLE))
    sel = selector.select(state, ["r0", "r1", "r2"])
    assert sel.winner_index is not None
    deflections = [s.deflection for s in sel.scores]
    assert all(d >= 0.0 for d in deflections)
    assert deflections[sel.winner_index] == min(deflections)
    assert sel.n_scorable == 3


def test_selector_reader_failure_marks_unscorable(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(coeffs, reader=_reader(TABLE))
    sel = selector.select(state, ["r0", "unknown", "r1"])
    assert "KeyError" in sel.scores[1].error
    assert sel.n_scorable == 2
    assert sel.winner_index in (0, 2)


def test_selector_reader_returning_epa_objects(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    """Readers may return EPA objects, not just tuples/arrays (regression)."""
    state = _state_with_pending_message(coeffs, dictionary)

    def epa_reader(text: str) -> EPA:
        return EPA.from_array(np.array(TABLE[text]))

    selector = BestOfKSelector(coeffs, reader=epa_reader)
    sel = selector.select(state, ["r0", "r1", "r2"])
    assert sel.winner_index is not None
    assert sel.n_scorable == 3


def test_selector_all_reader_failures_fallback(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(coeffs, reader=_reader(TABLE))
    sel = selector.select(state, ["x", "y"])
    assert sel.winner_index is None
    assert sel.n_scorable == 0
    assert "all 2 candidates" in sel.fallback_reason


def test_generate_and_select(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(
        coeffs, reader=_reader(TABLE), generator=lambda _s: ["r0", "r1", "r2"], k=3
    )
    candidates, sel = selector.generate_and_select(state)
    assert candidates == ("r0", "r1", "r2")
    assert sel.winner_index is not None
    assert sel.k == 3


def test_generate_and_select_k_mismatch(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(
        coeffs, reader=_reader(TABLE), generator=lambda _s: ["r0", "r1"], k=3
    )
    with pytest.raises(ValueError, match="expected k=3"):
        selector.generate_and_select(state)


def test_generate_and_select_requires_generator(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(coeffs, reader=_reader(TABLE), generator=None)
    with pytest.raises(ValueError, match="no candidate generator"):
        selector.generate_and_select(state)


def test_score_candidates_empty_raises(
    coeffs: ACTCoefficients, dictionary: ACTDictionary
) -> None:
    state = _state_with_pending_message(coeffs, dictionary)
    selector = BestOfKSelector(coeffs, reader=_reader(TABLE))
    with pytest.raises(ValueError, match="non-empty"):
        selector.score_candidates(state, [])
