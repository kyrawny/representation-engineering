"""CPU tests for the shared evaluation statistics (S11.1)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation import stats


def test_summary_stats_known_values() -> None:
    s = stats.summary_stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4.0
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["std"] == pytest.approx(np.array([1.0, 2.0, 3.0, 4.0]).std(ddof=1))


def test_summary_stats_single_value() -> None:
    s = stats.summary_stats([7.0])
    assert s["std"] == 0.0
    assert s["mean"] == 7.0


def test_summary_stats_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        stats.summary_stats([])


def test_paired_differences() -> None:
    diff = stats.paired_differences([3.0, 5.0], [1.0, 2.0])
    assert diff == pytest.approx([2.0, 3.0])


def test_paired_differences_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        stats.paired_differences([1.0, 2.0], [1.0])


def test_paired_differences_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        stats.paired_differences([1.0], [2.0])


def test_percentile_ci_known() -> None:
    values = list(np.linspace(0.0, 1.0, 101))
    low, high = stats.percentile_ci(values, alpha=0.10)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.95)


def test_percentile_ci_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        stats.percentile_ci([1.0, 2.0], alpha=0.0)


def test_percentile_ci_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        stats.percentile_ci([])


def test_cluster_bootstrap_reproducible_and_positive() -> None:
    # Two scenarios with different means, all differences positive: the
    # bootstrap distribution must have real spread and the CI must exclude 0.
    values = [1.0, 1.5, 2.0, 0.1, 0.2]
    clusters = ["s1", "s1", "s1", "s2", "s2"]
    a = stats.cluster_bootstrap(values, clusters, n_bootstrap=200, seed=42)
    b = stats.cluster_bootstrap(values, clusters, n_bootstrap=200, seed=42)
    assert a["ci_low"] == b["ci_low"]  # same seed -> same resamples
    assert a["ci_high"] == b["ci_high"]
    assert a["ci_low"] < a["ci_high"]  # non-degenerate distribution
    assert a["ci_low"] > 0.0
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["p_value"] == pytest.approx(0.0)
    assert a["n_clusters"] == 2.0


def test_cluster_bootstrap_negative_mean_p_value() -> None:
    values = [-1.0, -1.5, -2.0, -0.2]
    clusters = ["s1", "s1", "s2", "s2"]
    a = stats.cluster_bootstrap(values, clusters, n_bootstrap=200, seed=42)
    assert a["mean"] < 0.0
    assert a["ci_high"] < 0.0  # whole CI below 0
    assert a["ci_low"] < a["ci_high"]
    assert a["p_value"] == pytest.approx(0.0)


def test_cluster_bootstrap_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        stats.cluster_bootstrap([1.0, 2.0], ["s1", "s2", "s3"])


def test_cluster_bootstrap_single_cluster_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        stats.cluster_bootstrap([1.0, 2.0, 3.0], ["s1", "s1", "s1"])


def test_lin_ccc_perfect_concordance() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert stats.lin_ccc(x, x) == pytest.approx(1.0)


def test_lin_ccc_affine_shift_penalised() -> None:
    # Same shape but shifted: CCC < 1 (that is the point of CCC vs Pearson).
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 4.0, 5.0, 6.0]
    ccc = stats.lin_ccc(x, y)
    # 2*cov / (vx+vy+(mx-my)^2) = 3.333 / 7.333 for this case
    assert ccc == pytest.approx(0.4545, abs=1e-3)
    assert ccc < 1.0


def test_lin_ccc_constant_inputs() -> None:
    assert stats.lin_ccc([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_lin_ccc_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        stats.lin_ccc([1.0, 2.0], [1.0])


def test_agreement_stats_identity() -> None:
    x = [0.5, -1.0, 2.0, 0.0, 3.0]
    s = stats.agreement_stats(x, x)
    assert s["pearson_r"] == pytest.approx(1.0)
    assert s["spearman_r"] == pytest.approx(1.0)
    assert s["mae"] == pytest.approx(0.0)
    assert s["ccc"] == pytest.approx(1.0)
    assert s["n"] == 5.0


def test_agreement_stats_noisy() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.2, 2.5, 2.8, 4.3, 5.1]
    s = stats.agreement_stats(x, y)
    assert 0.9 < s["pearson_r"] < 1.0
    assert s["mae"] > 0.0
    assert 0.0 < s["ccc"] < 1.0


def test_agreement_stats_ties_in_spearman() -> None:
    x = [1.0, 1.0, 2.0, 3.0]
    y = [1.0, 2.0, 2.0, 3.0]
    s = stats.agreement_stats(x, y)
    assert 0.5 < s["spearman_r"] < 1.0


def test_agreement_stats_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        stats.agreement_stats([1.0], [1.0, 2.0])


def test_git_hash_returns_string() -> None:
    assert isinstance(stats.git_hash(), str)


def test_build_meta_contents() -> None:
    meta = stats.build_meta(model="mock", seed=7, config={"k": 4})
    assert meta["model"] == "mock"
    assert meta["seed"] == 7
    assert meta["config"] == {"k": 4}
    assert "timestamp" in meta
    assert "git_hash" in meta


def test_write_results(tmp_path) -> None:
    meta = stats.build_meta(model="mock")
    out = stats.write_results(tmp_path / "sub" / "res.json", {"a": 1}, meta)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["a"] == 1
    assert data["meta"]["model"] == "mock"


def test_write_results_rejects_meta_key() -> None:
    with pytest.raises(ValueError, match="'meta'"):
        stats.write_results("x.json", {"meta": {"x": 1}}, {})
