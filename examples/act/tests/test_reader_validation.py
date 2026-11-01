"""CPU tests for EPA reader validation (S11.2)."""

from __future__ import annotations

import json

import pytest

import evaluation.reader_validation as rv
from act_steering.dictionary import ACTDictionary
from evaluation import stats
from evaluation.reader_validation import (
    DIMENSIONS,
    ValidationItem,
    build_mock_reader,
    main,
    sample_dictionary_items,
    validate_reader,
)

DATA_DIR = r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act"


@pytest.fixture(scope="module")
def dictionary() -> ACTDictionary:
    return ACTDictionary.load(DATA_DIR)


def test_sample_dictionary_items(dictionary: ACTDictionary) -> None:
    items = sample_dictionary_items(dictionary, n_behaviours=5, n_identities=3, seed=7)
    assert len(items) == 8
    assert items[:5] and items[5:]
    # reproducible
    again = sample_dictionary_items(dictionary, n_behaviours=5, n_identities=3, seed=7)
    assert [i.text for i in items] == [i.text for i in again]
    # reference EPA matches the dictionary entry (the human rating)
    for item in items:
        term = dictionary.require(item.text)
        assert item.reference_epa == pytest.approx(term.as_vector())


def test_validate_reader_identity_reader() -> None:
    items = [
        ValidationItem("a", (1.0, -2.0, 0.5)),
        ValidationItem("b", (-0.5, 0.5, 1.5)),
        ValidationItem("c", (0.0, 1.0, -1.0)),
        ValidationItem("d", (2.0, 0.0, 0.0)),
        ValidationItem("e", (-1.5, 2.0, 0.8)),
        ValidationItem("f", (0.7, -0.5, -0.3)),
        ValidationItem("g", (-0.2, 0.2, 1.2)),
        ValidationItem("h", (1.2, -1.0, -0.9)),
    ]
    table = {i.text: i.reference_epa for i in items}

    def reader(text: str):
        return table[text]

    report = validate_reader(reader, items, n_folds=4)
    assert report["n_scorable"] == 8.0
    assert report["n_failed"] == 0.0
    for dim in DIMENSIONS:
        m = report["per_dimension"][dim]
        assert m["pearson_r"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["ccc"] == pytest.approx(1.0)
    assert report["overall"]["pearson_r"] == pytest.approx(1.0)
    assert report["overall"]["mean_3d_distance"] == pytest.approx(0.0)
    # 8 items in 4 folds of 2 each: every fold must be present
    assert len(report["folds"]) == 4
    assert all(f["n"] == 2.0 for f in report["folds"])


def test_validate_reader_mock_structure() -> None:
    items = [
        ValidationItem(f"t{i}", (float(i), float(-i), 0.1 * i)) for i in range(10)
    ]
    report = validate_reader(build_mock_reader(), items, n_folds=5, seed=42)
    assert set(report["per_dimension"].keys()) == set(DIMENSIONS)
    for m in report["per_dimension"].values():
        assert -1.0 <= m["pearson_r"] <= 1.0
        assert -1.0 <= m["spearman_r"] <= 1.0
        assert m["mae"] >= 0.0
        assert -1.0 <= m["ccc"] <= 1.0
    assert report["overall"]["n"] == 30.0  # pooled across 3 dimensions
    fold_n = sum(f["n"] for f in report["folds"])
    assert fold_n == report["n_scorable"]
    json.dumps(report)


def test_validate_reader_counts_failures() -> None:
    items = [ValidationItem(f"t{i}", (float(i), 0.0, 0.0)) for i in range(6)]

    def flaky_reader(text: str):
        if text in ("t1", "t4"):
            raise RuntimeError("reader down")
        return items[int(text[1])].reference_epa

    report = validate_reader(flaky_reader, items)
    assert report["n_failed"] == 2.0
    assert report["n_scorable"] == 4.0
    assert report["n_items"] == 6.0


def test_validate_reader_all_failures() -> None:
    items = [ValidationItem("a", (1.0, 0.0, 0.0)), ValidationItem("b", (0.0, 1.0, 0.0))]

    def dead_reader(text: str):
        raise RuntimeError("no model")

    report = validate_reader(dead_reader, items)
    assert report["n_scorable"] == 0.0
    assert report["per_dimension"]["E"] == {"n": 0.0}
    assert report["overall"]["n"] == 0.0
    assert report["folds"] == []


def test_validate_reader_empty_items_raise() -> None:
    with pytest.raises(ValueError, match="no validation items"):
        validate_reader(build_mock_reader(), [])


def test_resolve_stimulus_template_fit_bare_literal() -> None:
    fit = "You respond by {behaviour}."
    assert rv.resolve_stimulus_template("fit", fit) == fit
    assert rv.resolve_stimulus_template("bare", "anything") is None
    assert rv.resolve_stimulus_template("My reply: {behaviour}.", "x") == "My reply: {behaviour}."
    with pytest.raises(ValueError, match="behaviour"):
        rv.resolve_stimulus_template("no slot here", "x")


def test_wrap_items_wraps_and_keeps_references() -> None:
    items = [ValidationItem("praise", (2.0, 0.5, 0.5)), ValidationItem("help", (1.0, 0.0, 0.0))]
    wrapped = rv.wrap_items(items, "You respond by {behaviour}.")
    assert [i.text for i in wrapped] == [
        "You respond by praise.",
        "You respond by help.",
    ]
    # References and source are preserved; input not mutated.
    assert wrapped[0].reference_epa == items[0].reference_epa
    assert items[0].text == "praise"


def test_wrap_items_bare_is_identity() -> None:
    items = [ValidationItem("praise", (2.0, 0.5, 0.5))]
    wrapped = rv.wrap_items(items, None)
    assert wrapped == list(items)
    assert wrapped is not items


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "reader_validation.json"
    rc = main(["--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["config"]["reader"] == "mock"
    assert data["meta"]["model"] is None
    for dim in DIMENSIONS:
        assert dim in data["per_dimension"]
    assert "pearson_r" in data["overall"]
    assert data["n_items"] <= 40


def test_cli_model_mode_requires_directions(tmp_path) -> None:
    """Model mode exits fast on a missing directions artifact.

    The check runs before the model is loaded, so this test never touches
    the GPU even on a machine with the runbook artifacts cached.
    """
    missing = tmp_path / "missing" / "directions.npz"
    with pytest.raises(SystemExit):
        main(
            [
                "--reader", "model",
                "--directions", str(missing),
                "--out", str(tmp_path / "r.json"),
            ]
        )


def test_cli_seed_reproducible(tmp_path) -> None:
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    main(["--reader", "mock", "--n-items", "12", "--seed", "99", "--out", str(out_a)])
    main(["--reader", "mock", "--n-items", "12", "--seed", "99", "--out", str(out_b)])
    a = json.loads(out_a.read_text(encoding="utf-8"))
    b = json.loads(out_b.read_text(encoding="utf-8"))
    # same items, same mock reader, same fold split -> identical metrics
    assert a["per_dimension"] == b["per_dimension"]
    assert a["overall"] == b["overall"]
    assert a["folds"] == b["folds"]


def test_stats_writer_integration(tmp_path) -> None:
    # write_results must embed the meta block (regression for S11.1 contract)
    out = stats.write_results(tmp_path / "x.json", {"per_dimension": {}}, stats.build_meta())
    meta = json.loads(out.read_text(encoding="utf-8"))["meta"]
    assert isinstance(meta["git_hash"], str)
    assert "timestamp" in meta


def test_build_model_reader_projects_fake_model() -> None:
    import numpy as np

    from act_steering import directions as d
    from evaluation.reader_validation import build_model_reader

    class _Tok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            n = 1 if not text else 3
            return {
                "input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n),
            }

    class _Model:
        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            import torch

            n = int(input_ids.shape[1]) if input_ids is not None else 1
            hidden = torch.zeros(1, n, 4)
            hidden[:, :, 0] = 1.0

            class _Out:
                hidden_states = (hidden,) * 6
                logits = torch.zeros(1, n, 4)

            return _Out()

    directions = {
        "epa.evaluation": {2: np.array([1.0, 0.0, 0.0, 0.0]), 3: np.array([0.0, 1.0, 0.0, 0.0])},
        "epa.potency": {2: np.array([0.0, 0.0, 1.0, 0.0]), 3: np.array([0.0, 0.0, 0.0, 1.0])},
        "epa.activity": {2: np.array([0.6, 0.0, 0.0, 0.0]), 3: np.array([0.0, 0.8, 0.0, 0.0])},
    }
    bundle = d.DirectionsBundle(
        model_name="test/rv-fake", hidden_size=4, hidden_layers=[2, 3],
        directions=directions, seed=42,
    )
    reader = build_model_reader(_Model(), _Tok(), bundle)
    pred = reader("hello world")
    assert pred.e == pytest.approx(0.5)
    assert pred.p == pytest.approx(0.0)
    assert pred.a == pytest.approx(0.3)
