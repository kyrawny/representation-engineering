"""CPU tests for deflection-reduction analysis (S11.3)."""

from __future__ import annotations

import json

import pytest

from evaluation import stats
from evaluation.deflection_reduction import (
    compare_conditions,
    extract_turns,
    load_trace,
    main,
    run_mock_condition,
    summarize_condition,
)


def _make_trace(pairs: list[tuple[str, float | None]]) -> dict:
    """Build a minimal trace log from (scenario, u_post) pairs."""
    return {
        "turns": [
            {"turn": i, "user_message": scen, "u_pre": 1.0, "u_post": u}
            for i, (scen, u) in enumerate(pairs)
        ]
    }


def test_extract_turns() -> None:
    trace = _make_trace([("s1", 0.5), ("s2", None)])
    records = extract_turns(trace)
    assert records[0].turn == 0
    assert records[0].scenario == "s1"
    assert records[0].u_post == pytest.approx(0.5)
    assert records[1].u_post is None
    assert records[1].u_pre == pytest.approx(1.0)


def test_load_trace_rejects_non_trace(tmp_path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"nope": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a pipeline trace log"):
        load_trace(p)


def test_summarize_condition() -> None:
    trace = _make_trace([("s1", 0.5), ("s1", 1.5), ("s2", None)])
    s = summarize_condition(trace)
    assert s["n_turns"] == 3.0
    assert s["u_post_mean"] == pytest.approx(1.0)
    assert s["u_post_std"] == pytest.approx(0.7071, rel=1e-3)


def test_compare_conditions_treatment_reduces_deflection() -> None:
    baseline = _make_trace([("s1", 1.0), ("s1", 1.2), ("s2", 0.8), ("s2", 0.9), ("s3", 1.1)])
    treatment = _make_trace([("s1", 0.2), ("s1", 0.3), ("s2", 0.1), ("s2", 0.2), ("s3", 0.3)])
    report = compare_conditions(
        {"unsteered": baseline, "best_of_k": treatment},
        baseline="unsteered",
        treatments=["best_of_k"],
        n_bootstrap=200,
        seed=1,
    )
    comp = report["comparisons"][0]
    assert comp["n_pairs"] == 5.0
    assert comp["paired_delta_mean"] < 0.0
    assert comp["ci_high"] < 0.0
    assert comp["p_value"] == pytest.approx(0.0)
    assert comp["reduces_deflection"] is True


def test_compare_conditions_equal_conditions() -> None:
    a = _make_trace([("s1", 0.5), ("s2", 0.5), ("s3", 0.5)])
    b = _make_trace([("s1", 0.5), ("s2", 0.5), ("s3", 0.5)])
    report = compare_conditions({"a": a, "b": b}, baseline="a", treatments=["b"], seed=2)
    comp = report["comparisons"][0]
    assert comp["paired_delta_mean"] == pytest.approx(0.0)
    assert comp["reduces_deflection"] is False


def test_compare_conditions_insufficient_pairs() -> None:
    baseline = _make_trace([("s1", 1.0)])
    treatment = _make_trace([("s1", 0.5)])
    report = compare_conditions({"b": baseline, "t": treatment}, baseline="b", treatments=["t"])
    comp = report["comparisons"][0]
    assert "error" in comp
    assert comp["n_pairs"] == 1.0


def test_compare_conditions_skips_unpaired_and_none() -> None:
    baseline = _make_trace([("s1", 1.0), ("s1", 1.0), ("s2", 0.5)])
    treatment = _make_trace([("s1", 0.2), ("s2", None)])  # s1 turn 0 paired; s2 u_post None
    report = compare_conditions({"b": baseline, "t": treatment}, baseline="b", treatments=["t"])
    comp = report["comparisons"][0]
    assert comp["n_pairs"] == 1.0
    assert "error" in comp  # <2 pairs


def test_compare_conditions_unknown_names() -> None:
    with pytest.raises(ValueError, match="baseline"):
        compare_conditions({}, baseline="x", treatments=[])
    with pytest.raises(ValueError, match="treatment"):
        compare_conditions({"b": _make_trace([("s1", 1.0)])}, baseline="b", treatments=["t"])


def test_run_mock_condition_shapes() -> None:
    trace = run_mock_condition(k=4)
    assert len(trace["turns"]) == 2
    for entry in trace["turns"]:
        assert "u_pre" in entry
        assert entry["u_post"] is None or isinstance(entry["u_post"], float)


def test_mock_best_of_k_does_not_worsen_deflection() -> None:
    u1 = run_mock_condition(k=1)
    u4 = run_mock_condition(k=4)
    report = compare_conditions(
        {"unsteered": u1, "best_of_k": u4},
        baseline="unsteered",
        treatments=["best_of_k"],
        seed=3,
    )
    comp = report["comparisons"][0]
    # best-of-4 picks the min-U candidate, so it cannot do worse than k=1
    assert comp["paired_delta_mean"] <= 0.0


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "dr.json"
    rc = main(["--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["baseline"] == "unsteered"
    assert "unsteered" in data["conditions"]
    assert "best_of_k" in data["conditions"]
    assert data["comparisons"]
    assert data["meta"]["config"]["quick"] is True


def test_cli_with_log_files(tmp_path) -> None:
    pb = tmp_path / "b.json"
    pt = tmp_path / "t.json"
    pb.write_text(json.dumps(_make_trace([("s1", 1.0), ("s2", 0.8)])), encoding="utf-8")
    pt.write_text(json.dumps(_make_trace([("s1", 0.4), ("s2", 0.3)])), encoding="utf-8")
    out = tmp_path / "r.json"
    rc = main(
        [
            "--log", f"base={pb}",
            "--log", f"treat={pt}",
            "--baseline", "base",
            "--treatments", "treat",
            "--n-bootstrap", "200",
            "--out", str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    comp = data["comparisons"][0]
    assert comp["paired_delta_mean"] < 0.0
    assert data["meta"]["git_hash"] is not None


def test_stats_module_consistency() -> None:
    # regression: the analysis must use the shared cluster bootstrap
    values = [0.1, 0.2, -0.1, 0.3]
    clusters = ["a", "a", "b", "b"]
    result = stats.cluster_bootstrap(values, clusters, n_bootstrap=100, seed=0)
    assert "ci_low" in result and "p_value" in result
