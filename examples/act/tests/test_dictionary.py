"""Tests for the MTurk-Interact dictionary loader and lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from act_steering.dictionary import ACTDictionary, normalize_term


@pytest.fixture(scope="module")
def dictionary(data_dir: Path) -> ACTDictionary:
    """Load the real MTurk-Interact dictionaries once per module."""
    return ACTDictionary.load(data_dir)


def test_load_counts(dictionary: ACTDictionary) -> None:
    """The current release ships 968 identities, 853 behaviours, 660 modifiers."""
    assert dictionary.counts == {"identities": 968, "behaviors": 853, "modifiers": 660}


def test_known_identity_values(dictionary: ACTDictionary) -> None:
    entry = dictionary.require("27_year_old")
    assert entry.as_vector() == (0.77, 0.18, 1.31)


def test_known_behavior_values(dictionary: ACTDictionary) -> None:
    entry = dictionary.require("abandon")
    assert entry.as_vector() == (-3.19, -0.43, -0.60)


def test_known_modifier_values(dictionary: ACTDictionary) -> None:
    entry = dictionary.require("able_bodied")
    assert entry.as_vector() == (3.0, 2.22, 1.51)


def test_normalize_term() -> None:
    assert normalize_term(" 27 Year Old ") == "27_year_old"
    assert normalize_term("air-force-officer") == "air_force_officer"
    assert normalize_term("a__b") == "a_b"


def test_lookup_is_normalization_insensitive(dictionary: ACTDictionary) -> None:
    assert dictionary.lookup("27 Year Old") is not None
    assert dictionary.lookup("AIR_FORCE_OFFICER") is not None
    assert dictionary.lookup("not_a_real_term_xyz") is None


def test_require_missing_raises(dictionary: ACTDictionary) -> None:
    with pytest.raises(KeyError):
        dictionary.require("not_a_real_term_xyz")


def test_sample_reproducible(dictionary: ACTDictionary) -> None:
    first = dictionary.sample(10, seed=7)
    second = dictionary.sample(10, seed=7)
    assert [t.term for t in first] == [t.term for t in second]


def test_sample_seed_sensitive(dictionary: ACTDictionary) -> None:
    first = [t.term for t in dictionary.sample(25, seed=1)]
    second = [t.term for t in dictionary.sample(25, seed=2)]
    assert first != second


def test_sample_without_replacement(dictionary: ACTDictionary) -> None:
    sample = dictionary.sample(50, seed=3)
    assert len({t.term for t in sample}) == 50


def test_sample_excludes_terms(dictionary: ACTDictionary) -> None:
    sample = dictionary.sample(5, seed=3, exclude=["abandon"])
    assert all(t.term != "abandon" for t in sample)


def test_sample_from_identities(dictionary: ACTDictionary) -> None:
    sample = dictionary.sample(5, seed=3, collection="identities")
    assert len(sample) == 5
    assert all(t.term in {t.term for t in dictionary.identities} for t in sample)


def test_sample_too_many_raises(dictionary: ACTDictionary) -> None:
    with pytest.raises(ValueError):
        dictionary.sample(10_000, seed=3)


def test_sample_unknown_collection_raises(dictionary: ACTDictionary) -> None:
    with pytest.raises(ValueError):
        dictionary.sample(5, seed=3, collection="nope")


def test_all_values_in_human_rated_range(dictionary: ACTDictionary) -> None:
    """Every EPA value must sit inside the observed human-rating range."""
    terms = dictionary.identities + dictionary.behaviors + dictionary.modifiers
    assert len(terms) == 2481
    for term in terms:
        for value in term.as_vector():
            assert -4.35 <= value <= 4.35
