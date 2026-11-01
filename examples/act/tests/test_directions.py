"""CPU tests for direction extraction (S6.1).

Covers the torch-free half of ``act_steering.directions``: contrast text
formatting, EPA extreme-behaviour contrast sets, the linear algebra
(normalisation, Løwdin orthogonalisation, cosine statistics), reader-array
conversion, the artifact bundle round-trip, and the CLI help path. The
GPU smoke run (``python -m act_steering.directions --quick``) is S6.2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from act_steering import directions as d
from act_steering.act_core import ACTCoefficients
from act_steering.config import DataPaths
from act_steering.dictionary import ACTDictionary
from act_steering.stimuli import (
    RESPONSE_TEMPLATE,
    ContrastItem,
    ContrastPair,
    build_contrast_set,
    dicts_to_pairs,
    pairs_to_dicts,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dictionary(data_dir: Path) -> ACTDictionary:
    return ACTDictionary.load(data_dir)


@pytest.fixture(scope="module")
def contrast_pairs(data_dir: Path) -> list[ContrastPair]:
    paths = DataPaths(data_dir=data_dir)
    coeffs = ACTCoefficients.load(paths.coefficients_file)
    dictionary = ACTDictionary.load(data_dir)
    return build_contrast_set(
        coeffs, dictionary, n_pairs=6, seed=7, candidates_per_trial=40
    )


def _tiny_pair() -> ContrastPair:
    """A synthetic pair that does not touch the real data files."""
    context = "You are a teacher. A person who is a student says to you: praise."
    low = ContrastItem(
        context=context,
        user_term="student",
        agent_term="teacher",
        message_term="praise",
        behaviour_term="help",
        behaviour_epa=(2.0, 0.5, 0.5),
        deflection=0.1,
    )
    high = ContrastItem(
        context=context,
        user_term="student",
        agent_term="teacher",
        message_term="praise",
        behaviour_term="insult",
        behaviour_epa=(-2.0, 0.5, 0.5),
        deflection=9.0,
    )
    return ContrastPair(low=low, high=high, delta_u=8.9)


# --- contrast text formatting ---------------------------------------------


def test_contrast_pair_to_texts_shares_context() -> None:
    pair = _tiny_pair()
    low, high = d.contrast_pair_to_texts(pair)
    assert low.startswith(pair.low.context)
    assert high.startswith(pair.high.context)
    assert RESPONSE_TEMPLATE.format(behaviour="help") in low
    assert RESPONSE_TEMPLATE.format(behaviour="insult") in high
    assert low != high


def test_contrast_pair_to_texts_real_pairs(contrast_pairs: list[ContrastPair]) -> None:
    assert len(contrast_pairs) == 6
    for pair in contrast_pairs:
        low, high = d.contrast_pair_to_texts(pair)
        assert pair.low.context == pair.high.context
        assert low.split("You respond by ")[0] == high.split("You respond by ")[0]
        assert pair.delta_u == pytest.approx(pair.high.deflection - pair.low.deflection)


def test_pairs_to_training_inputs_interleaves_low_first() -> None:
    pairs = [_tiny_pair(), _tiny_pair()]
    inputs, labels = d.pairs_to_training_inputs(pairs)
    assert len(inputs) == 4
    assert labels == [[1, 0], [1, 0]]
    low, high = d.contrast_pair_to_texts(pairs[0])
    assert inputs[0] == low
    assert inputs[1] == high


# --- EPA extreme-behaviour contrast sets -----------------------------------


def test_build_epa_contrast_items_shapes(dictionary: ACTDictionary) -> None:
    n = 15
    items = d.build_epa_contrast_items(dictionary, n_per_side=n)
    assert set(items) == set(d.EPA_DIMENSIONS)
    for dim in d.EPA_DIMENSIONS:
        inputs, labels = items[dim]
        assert len(inputs) == 2 * n
        assert labels == [[1, 0]] * n
        assert all(RESPONSE_TEMPLATE.split("{")[0] in text for text in inputs)


def test_build_epa_contrast_items_extreme_ranking(dictionary: ACTDictionary) -> None:
    n = 20
    items = d.build_epa_contrast_items(dictionary, n_per_side=n)
    for dim in d.EPA_DIMENSIONS:
        attr = d.EPA_DIM_ATTR[dim]
        ranked = sorted(dictionary.behaviors, key=lambda t: (getattr(t, attr), t.term))
        expected_hi = ranked[-1]
        expected_lo = ranked[0]
        inputs, _ = items[dim]
        # First pair: i-th highest with i-th lowest, high side first.
        assert expected_hi.term in inputs[0]
        assert expected_lo.term in inputs[1]


def test_build_epa_contrast_items_deterministic(dictionary: ACTDictionary) -> None:
    a = d.build_epa_contrast_items(dictionary, n_per_side=10)
    b = d.build_epa_contrast_items(dictionary, n_per_side=10)
    assert a == b


def test_build_epa_contrast_items_rejects_bad_n(dictionary: ACTDictionary) -> None:
    with pytest.raises(ValueError):
        d.build_epa_contrast_items(dictionary, n_per_side=0)
    with pytest.raises(ValueError):
        d.build_epa_contrast_items(dictionary, n_per_side=len(dictionary.behaviors))


def test_mix_pair_order_deterministic() -> None:
    pairs = [(f"h{i}", f"l{i}") for i in range(50)]
    a = d.mix_pair_order(pairs, seed=1)
    b = d.mix_pair_order(pairs, seed=1)
    c = d.mix_pair_order(pairs, seed=2)
    assert a == b
    assert a != c


def test_mix_pair_order_flips_some_but_keeps_sides() -> None:
    pairs = [(f"h{i}", f"l{i}") for i in range(50)]
    mixed = d.mix_pair_order(pairs, seed=1)
    assert len(mixed) == len(pairs)
    # At least one pair flipped and at least one kept (p(all same) = 2^-50).
    assert any(a != p[0] for (a, _), p in zip(mixed, pairs))
    assert any(a == p[0] for (a, _), p in zip(mixed, pairs))
    for (first, second), (hi, lo) in zip(mixed, pairs):
        assert {first, second} == {hi, lo}


def test_build_epa_contrast_items_flat_labels_unmixed(dictionary: ACTDictionary) -> None:
    n = 15
    items = d.build_epa_contrast_items(
        dictionary, n_per_side=n, label_format="flat"
    )
    for dim in d.EPA_DIMENSIONS:
        inputs, labels = items[dim]
        assert len(inputs) == 2 * n
        assert labels == [1] * n  # un-mixed: high side always emitted first


def test_build_epa_contrast_items_mixed_flat_labels_correct(dictionary: ACTDictionary) -> None:
    n = 20
    items = d.build_epa_contrast_items(
        dictionary, n_per_side=n, label_format="flat", mix_seed=7
    )
    for dim in d.EPA_DIMENSIONS:
        attr = d.EPA_DIM_ATTR[dim]
        values = {t.term: getattr(t, attr) for t in dictionary.behaviors}
        inputs, labels = items[dim]
        assert len(labels) == n
        assert 0 in labels and 1 in labels  # both orderings occurred
        for i, label in enumerate(labels):
            first, second = inputs[2 * i], inputs[2 * i + 1]
            term_first = first.split("by ")[1].rstrip(".")
            term_second = second.split("by ")[1].rstrip(".")
            assert (label == 1) == (values[term_first] > values[term_second])


def test_build_epa_contrast_items_mixed_nested_labels_correct(dictionary: ACTDictionary) -> None:
    n = 20
    items = d.build_epa_contrast_items(
        dictionary, n_per_side=n, label_format="nested", mix_seed=7
    )
    for dim in d.EPA_DIMENSIONS:
        attr = d.EPA_DIM_ATTR[dim]
        values = {t.term: getattr(t, attr) for t in dictionary.behaviors}
        inputs, labels = items[dim]
        assert len(labels) == n
        assert [1, 0] in labels and [0, 1] in labels
        for i, label in enumerate(labels):
            first, second = inputs[2 * i], inputs[2 * i + 1]
            term_first = first.split("by ")[1].rstrip(".")
            term_second = second.split("by ")[1].rstrip(".")
            high_first = values[term_first] > values[term_second]
            assert label == ([1, 0] if high_first else [0, 1])


def test_build_epa_contrast_items_multi_template(dictionary: ACTDictionary) -> None:
    n = 10
    t1 = "You respond by {behaviour}."
    t2 = "Your reply is: {behaviour}."
    a = d.build_epa_contrast_items(
        dictionary, n_per_side=n, templates=[t1, t2], label_format="flat", mix_seed=3
    )
    b = d.build_epa_contrast_items(
        dictionary, n_per_side=n, templates=[t1, t2], label_format="flat", mix_seed=3
    )
    assert a == b
    for dim in d.EPA_DIMENSIONS:
        inputs, labels = a[dim]
        assert len(inputs) == 4 * n
        assert len(labels) == 2 * n
        # Each template contributes n pairs, in template order.
        assert all(i.startswith("You respond by ") for i in inputs[0 : 2 * n])
        assert all(i.startswith("Your reply is: ") for i in inputs[2 * n :])


def test_build_epa_contrast_items_rejects_bad_label_format(dictionary: ACTDictionary) -> None:
    with pytest.raises(ValueError, match="label_format"):
        d.build_epa_contrast_items(dictionary, n_per_side=5, label_format="bogus")


def test_build_epa_contrast_items_rejects_template_without_placeholder(
    dictionary: ACTDictionary,
) -> None:
    with pytest.raises(ValueError, match="behaviour"):
        d.build_epa_contrast_items(dictionary, n_per_side=5, templates=["no slot"])


# --- multi-template direction averaging -----------------------------------


def test_average_signed_directions_basic() -> None:
    v = np.array([0.6, 0.8, 0.0, 0.0])
    per_template = {"t1": {2: v}, "t2": {2: v.copy()}}
    out = d.average_signed_directions(per_template)
    assert out[2] == pytest.approx(v)


def test_average_signed_directions_partial_flip() -> None:
    v = np.array([1.0, 0.0, 0.0, 0.0])
    other = np.array([0.0, 1.0, 0.0, 0.0])
    out = d.average_signed_directions({"t1": {2: v}, "t2": {2: other}})
    expected = np.array([0.5, 0.5, 0.0, 0.0])
    expected = expected / np.linalg.norm(expected)
    assert out[2] == pytest.approx(expected)
    # The shared high-side component is preserved: positive x, positive y.
    assert out[2][0] > 0 and out[2][1] > 0


def test_average_signed_directions_sign_flip_still_points_right() -> None:
    # One template fitted backwards but weaker than the consensus: the
    # average still points the consensus way.
    v = np.array([1.0, 0.0, 0.0, 0.0])
    flipped = np.array([-0.8, 0.0, 0.6, 0.0])
    out = d.average_signed_directions({"t1": {2: v}, "t2": {2: flipped}})
    assert out[2][0] > 0  # consensus direction preserved


def test_average_signed_directions_degenerate_raises() -> None:
    v = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        d.average_signed_directions({"t1": {2: v}, "t2": {2: -v}})


def test_average_signed_directions_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        d.average_signed_directions({})


def test_extract_epa_directions_routes_method() -> None:
    """direction_method is forwarded with the matching finder kwargs."""

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_directions(self, *args, **kwargs) -> dict:
            self.calls.append(kwargs)
            return {"directions": {}, "direction_signs": {}}

    for method, labels, finder_kwargs in (
        ("pca", [[1, 0], [0, 1]], {"n_components": 1}),
        ("cluster_mean", [1, 0], {}),
    ):
        pipeline = _FakePipeline()
        items = {dim: (["a", "b"], labels) for dim in d.EPA_DIMENSIONS}
        d.extract_epa_directions(pipeline, items, [3, 17], direction_method=method)
        assert len(pipeline.calls) == len(d.EPA_DIMENSIONS)
        for call in pipeline.calls:
            assert call["direction_method"] == method
            assert call["direction_finder_kwargs"] == finder_kwargs


def test_extract_epa_directions_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="direction_method"):
        d.extract_epa_directions(object(), {"evaluation": ([], [])}, [3], direction_method="bogus")


# --- linear algebra ---------------------------------------------------------


def test_l2_normalize_unit_norm() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=32)
    u = d.l2_normalize(v)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert np.allclose(u, v / np.linalg.norm(v))


def test_l2_normalize_zero_and_nan() -> None:
    assert np.all(d.l2_normalize(np.zeros(4)) == 0.0)
    out = d.l2_normalize(np.array([1.0, np.nan]))
    assert np.all(out == 0.0)


def test_lowdin_orthonormalize_orthonormal() -> None:
    rng = np.random.default_rng(1)
    D = rng.normal(size=(64, 3))
    Q = d.lowdin_orthonormalize(D)
    assert Q.shape == (64, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)


def test_lowdin_orthonormalize_order_independent() -> None:
    rng = np.random.default_rng(2)
    D = rng.normal(size=(32, 3))
    q1 = d.lowdin_orthonormalize(D)
    q2 = d.lowdin_orthonormalize(D[:, ::-1])
    # Reordering the input columns only reorders the output columns.
    assert np.allclose(q2[:, ::-1], q1, atol=1e-10)


def test_lowdin_rank_deficient_duplicated_columns_collapse() -> None:
    rng = np.random.default_rng(3)
    v = rng.normal(size=16)
    v = v / np.linalg.norm(v)
    w = rng.normal(size=16)
    w = w / np.linalg.norm(w)
    D = np.stack([v, v, w], axis=1)  # first two columns identical
    Q = d.lowdin_orthonormalize(D)
    assert np.all(np.isfinite(Q))
    # Duplicated input columns collapse to identical output columns.
    assert np.allclose(Q[:, 0], Q[:, 1], atol=1e-10)
    # Rank-deficient invariant: Q.T @ Q is the projector onto the row
    # space of D (idempotent, symmetric, rank == rank(D)).
    gram = Q.T @ Q
    assert np.allclose(gram @ gram, gram, atol=1e-8)
    assert np.allclose(gram, gram.T, atol=1e-12)
    assert int(np.linalg.matrix_rank(gram)) == 2
    # Span is preserved: Q lies in span(D).
    coef, *_ = np.linalg.lstsq(D, Q, rcond=None)
    assert np.allclose(D @ coef, Q, atol=1e-8)


def test_lowdin_rejects_1d() -> None:
    with pytest.raises(ValueError):
        d.lowdin_orthonormalize(np.ones(5))


def test_cosine_matrix_known_vectors() -> None:
    D = np.eye(3)[:, :2]  # e1, e2
    cos = d.cosine_matrix(D)
    assert np.allclose(cos, np.eye(2), atol=1e-12)


def test_cosine_stats_shape() -> None:
    rng = np.random.default_rng(4)
    directions = {}
    for dim in d.EPA_DIMENSIONS:
        directions[f"epa.{dim}"] = {
            layer: rng.normal(size=8) for layer in (1, 2)
        }
    stats = d.cosine_stats(directions)
    assert set(stats) == {"1", "2"}
    for layer in ("1", "2"):
        assert len(stats[layer]) == 3
        assert all(len(row) == 3 for row in stats[layer])


# --- reader arrays -> signed unit vectors ----------------------------------


def test_reader_directions_to_vectors_applies_sign() -> None:
    directions = {17: np.array([[3.0, 4.0]])}
    signs = {17: np.array([-1.0])}
    out = d.reader_directions_to_vectors(directions, signs)
    assert out[17].shape == (2,)
    assert np.allclose(out[17], [-0.6, -0.8])


def test_reader_directions_to_vectors_no_signs() -> None:
    directions = {5: np.array([[0.0, -5.0]])}
    out = d.reader_directions_to_vectors(directions, None)
    assert np.allclose(out[5], [0.0, -1.0])


# --- layer index mapping and CLI spec parsing ------------------------------


def test_decoder_layers_to_hf_indices() -> None:
    assert d.decoder_layers_to_hf_indices([0, 31], 32) == [1, 32]
    with pytest.raises(ValueError):
        d.decoder_layers_to_hf_indices([32], 32)


def test_parse_layers() -> None:
    assert d.parse_layers("all") is None
    assert d.parse_layers("ALL") is None
    assert d.parse_layers("1, 0, 17") == [0, 1, 17]
    with pytest.raises(ValueError):
        d.parse_layers("  , ")


# --- artifact bundle round-trip --------------------------------------------


def _synthetic_bundle(rng: np.random.Generator) -> d.DirectionsBundle:
    hidden_size = 8
    layers = [1, 5, 7]
    names = [d.DEFLECTION_DIRECTION, *(f"epa.{dim}" for dim in d.EPA_DIMENSIONS)]
    directions: dict[str, dict[int, np.ndarray]] = {}
    for name in names:
        directions[name] = {
            layer: d.l2_normalize(rng.normal(size=hidden_size)) for layer in layers
        }
    return d.DirectionsBundle(
        model_name="test/fake-model",
        hidden_size=hidden_size,
        hidden_layers=layers,
        directions=directions,
        created="2026-10-30T00:00:00+00:00",
        seed=42,
        lowdin=True,
        contrast_set_path="artifacts/contrast_set.json",
        n_contrast_pairs=1300,
        epa_items_per_side=60,
        stats={"epa_cosines_before": {"1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}},
    )


def test_bundle_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    bundle = _synthetic_bundle(rng)
    npz_path = tmp_path / "bundle.npz"
    json_path = tmp_path / "bundle.json"
    bundle.save(npz_path, json_path)
    assert npz_path.exists() and json_path.exists()

    loaded = d.DirectionsBundle.load(npz_path, json_path)
    loaded.validate()
    assert loaded.model_name == bundle.model_name
    assert loaded.hidden_layers == bundle.hidden_layers
    assert loaded.lowdin is True
    assert loaded.n_contrast_pairs == 1300
    for name in bundle.direction_names:
        for layer in bundle.hidden_layers:
            assert np.allclose(
                loaded.direction(name, layer),
                bundle.direction(name, layer),
                atol=1e-6,  # float32 storage
            )


def test_bundle_raw_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    bundle = _synthetic_bundle(rng)
    for dim in d.EPA_DIMENSIONS:
        bundle.directions[f"epa.{dim}.raw"] = {
            layer: bundle.direction(f"epa.{dim}", layer).copy()
            for layer in bundle.hidden_layers
        }
    bundle.epa_raw_saved = True
    bundle.epa_direction_method = "cluster_mean"
    bundle.epa_templates = ["You respond by {behaviour}."]
    bundle.epa_mix_seed = 42
    npz_path = tmp_path / "raw.npz"
    json_path = tmp_path / "raw.json"
    bundle.save(npz_path, json_path)

    loaded = d.DirectionsBundle.load(npz_path, json_path)
    loaded.validate()  # must also validate the .raw entries
    assert loaded.epa_raw_saved is True
    assert loaded.epa_direction_method == "cluster_mean"
    assert loaded.epa_templates == ["You respond by {behaviour}."]
    assert loaded.epa_mix_seed == 42
    for dim in d.EPA_DIMENSIONS:
        for layer in bundle.hidden_layers:
            assert np.allclose(
                loaded.direction(f"epa.{dim}.raw", layer),
                bundle.direction(f"epa.{dim}.raw", layer),
                atol=1e-6,
            )


def test_bundle_old_format_loads_without_raw(tmp_path: Path) -> None:
    """Bundles saved before the .raw extras exist still load and validate."""
    rng = np.random.default_rng(8)
    bundle = _synthetic_bundle(rng)  # no .raw keys, legacy meta fields absent
    npz_path = tmp_path / "old.npz"
    json_path = tmp_path / "old.json"
    bundle.save(npz_path, json_path)
    loaded = d.DirectionsBundle.load(npz_path, json_path)
    loaded.validate()
    assert loaded.epa_raw_saved is False
    assert loaded.epa_direction_method == "pca"
    assert loaded.epa_templates == []
    assert loaded.epa_mix_seed is None


def test_bundle_validate_rejects_bad_raw(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    bundle = _synthetic_bundle(rng)
    for dim in d.EPA_DIMENSIONS:
        bundle.directions[f"epa.{dim}.raw"] = {
            layer: bundle.direction(f"epa.{dim}", layer).copy()
            for layer in bundle.hidden_layers
        }
    bundle.epa_raw_saved = True
    # Corrupt one raw vector: zero it out.
    bundle.directions["epa.potency.raw"][5] = np.zeros(8)
    with pytest.raises(ValueError, match="epa.potency.raw"):
        bundle.validate()


def test_bundle_epa_matrix_column_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    bundle = _synthetic_bundle(rng)
    npz_path = tmp_path / "b.npz"
    json_path = tmp_path / "b.json"
    bundle.save(npz_path, json_path)
    loaded = d.DirectionsBundle.load(npz_path, json_path)
    mat = loaded.epa_matrix(5)
    assert mat.shape == (8, 3)
    assert np.allclose(mat[:, 0], loaded.direction("epa.evaluation", 5))
    assert np.allclose(mat[:, 1], loaded.direction("epa.potency", 5))
    assert np.allclose(mat[:, 2], loaded.direction("epa.activity", 5))


def test_bundle_load_rejects_unknown_format(tmp_path: Path) -> None:
    json_path = tmp_path / "bad.json"
    json_path.write_text(json.dumps({"format": "bogus-v9"}), encoding="utf-8")
    npz_path = tmp_path / "bad.npz"
    np.savez_compressed(npz_path, **{"x|1": np.zeros(4)})
    with pytest.raises(ValueError, match="unsupported directions format"):
        d.DirectionsBundle.load(npz_path, json_path)


def test_bundle_validate_rejects_zero_vector(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    bundle = _synthetic_bundle(rng)
    bundle.directions["deflection_contrast"][1] = np.zeros(8)
    with pytest.raises(ValueError, match="zero vector"):
        bundle.validate()


# --- CLI --------------------------------------------------------------------


def test_dicts_to_pairs_roundtrip(contrast_pairs: list[ContrastPair]) -> None:
    rebuilt = dicts_to_pairs(pairs_to_dicts(contrast_pairs))
    assert rebuilt == contrast_pairs


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "act_steering.directions", "--help"],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--lowdin" in result.stdout
    assert "--contrast-set" in result.stdout


def test_ensure_pad_token_falls_back_to_eos() -> None:
    from act_steering.directions import ensure_pad_token

    class _Tok:
        pad_token = None
        eos_token = "</s>"
        eos_token_id = 128009

    class _Cfg:
        pad_token_id = None

    class _Model:
        config = _Cfg()

    tok = _Tok()
    model = _Model()
    out = ensure_pad_token(tok, model)
    assert out is tok
    assert tok.pad_token == "</s>"
    assert model.config.pad_token_id == 128009


def test_ensure_pad_token_leaves_existing_untouched() -> None:
    from act_steering.directions import ensure_pad_token

    class _Tok:
        pad_token = "<pad>"
        eos_token = "</s>"
        eos_token_id = 2

    class _Cfg:
        pad_token_id = 0

    class _Model:
        config = _Cfg()

    tok = _Tok()
    model = _Model()
    ensure_pad_token(tok, model)
    assert tok.pad_token == "<pad>"
    assert model.config.pad_token_id == 0
