"""Shared statistics for the evaluation suite (S11).

Pure numpy (torch-free): cluster-bootstrap confidence intervals, Lin's
concordance correlation, agreement metrics (Pearson / Spearman / MAE /
CCC), paired differences, summary statistics, and a JSON results writer
with run metadata (model, seed, git hash, config).

All resampling takes an explicit seed so evaluation runs are
reproducible.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_SEED

__all__ = [
    "agreement_stats",
    "build_meta",
    "cluster_bootstrap",
    "git_hash",
    "lin_ccc",
    "paired_differences",
    "percentile_ci",
    "summary_stats",
    "write_results",
]


def _validate_paired(
    name: str, a: Sequence[float], b: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equally sized, non-empty numeric sequences."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{name}: inputs must be 1-D")
    if x.size != y.size:
        raise ValueError(f"{name}: length mismatch ({x.size} vs {y.size})")
    if x.size < 2:
        raise ValueError(f"{name}: need at least 2 values, got {x.size}")
    return x, y


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties assigned the mean of their positions."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # average ranks over ties
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def summary_stats(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics for a numeric sequence.

    Args:
        values: The values to summarise.

    Returns:
        A dict with ``n``, ``mean``, ``std`` (sample), ``median``, ``min``,
        ``max``.

    Raises:
        ValueError: If the sequence is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Elementwise differences ``a - b`` for paired observations.

    Args:
        a: The first sequence (e.g. unsteered values).
        b: The second sequence (e.g. steered values).

    Returns:
        The difference array ``a - b``.

    Raises:
        ValueError: On length mismatch, non-1-D input, or fewer than 2 values.
    """
    x, y = _validate_paired("paired_differences", a, b)
    return x - y


def percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval for the mean via resampling.

    Args:
        values: The values (one draw per resample, or the observation set).
        alpha: Significance level (0.05 gives a 95% interval).

    Returns:
        A tuple ``(low, high)``.

    Raises:
        ValueError: If the input is empty or ``alpha`` is out of range.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot compute a confidence interval for an empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Cluster bootstrap for the mean of paired differences.

    Resamples *clusters* (not rows) with replacement, then pools all rows
    of the resampled clusters. This is the correct unit for repeated
    measures within a scenario (review W4).

    Args:
        values: The paired differences (one per observation).
        clusters: The cluster label per observation (e.g. scenario id).
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level for the percentile interval.
        seed: RNG seed for reproducibility.

    Returns:
        A dict with ``mean``, ``ci_low``, ``ci_high``, ``p_value``
        (two-sided, bootstrap sign test against 0), ``n``, ``n_clusters``,
        ``n_bootstrap``.

    Raises:
        ValueError: On length mismatch, fewer than 2 values, or fewer than
            2 clusters.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    cluster_list = list(clusters)
    if values_arr.size != len(cluster_list):
        raise ValueError(
            f"cluster_bootstrap: length mismatch ({values_arr.size} vs {len(cluster_list)})"
        )
    if values_arr.size < 2:
        raise ValueError("cluster_bootstrap: need at least 2 values")
    unique_clusters = sorted(set(cluster_list))
    if len(unique_clusters) < 2:
        raise ValueError("cluster_bootstrap: need at least 2 clusters")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    cluster_index = {c: i for i, c in enumerate(unique_clusters)}
    groups = [
        values_arr[np.array([cluster_index[c] for c in cluster_list]) == i]
        for i in range(len(unique_clusters))
    ]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        picked = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        boot_means[b] = np.concatenate([groups[i] for i in picked]).mean()

    mean = float(values_arr.mean())
    low, high = percentile_ci(boot_means, alpha)
    if mean > 0:
        p_value = float(2.0 * min(1.0, (boot_means <= 0.0).mean()))
    elif mean < 0:
        p_value = float(2.0 * min(1.0, (boot_means >= 0.0).mean()))
    else:
        p_value = 1.0
    p_value = min(1.0, p_value)
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "n": float(values_arr.size),
        "n_clusters": float(len(unique_clusters)),
        "n_bootstrap": float(n_bootstrap),
    }


def lin_ccc(a: Sequence[float], b: Sequence[float]) -> float:
    """Lin's concordance correlation coefficient (reproducibility).

    Args:
        a: The reference values (e.g. human EPA ratings).
        b: The compared values (e.g. reader predictions).

    Returns:
        The CCC in ``[-1, 1]``; 1 for perfect concordance. Returns 0.0 when
        the denominator vanishes (constant inputs).

    Raises:
        ValueError: On length mismatch, non-1-D input, or fewer than 2 values.
    """
    x, y = _validate_paired("lin_ccc", a, b)
    mx, my = x.mean(), y.mean()
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    denom = vx + vy + (mx - my) ** 2
    return 0.0 if denom == 0 else float(2.0 * cov / denom)


def agreement_stats(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Agreement metrics between a reference and a prediction.

    Args:
        a: The reference values.
        b: The predicted values.

    Returns:
        A dict with ``n``, ``pearson_r``, ``spearman_r``, ``mae``,
        ``ccc``.

    Raises:
        ValueError: See :func:`lin_ccc`.
    """
    x, y = _validate_paired("agreement_stats", a, b)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = float(np.sqrt((dx**2).sum() * (dy**2).sum()))
    if denom == 0:  # noqa: SIM108 - the equivalent ternary exceeds the line limit
        pearson = 1.0 if np.allclose(x, y) else 0.0
    else:
        pearson = float((dx * dy).sum() / denom)
    rx = _ranks(x)
    ry = _ranks(y)
    drx = rx - rx.mean()
    dry = ry - ry.mean()
    rdenom = float(np.sqrt((drx**2).sum() * (dry**2).sum()))
    spearman = 0.0 if rdenom == 0 else float((drx * dry).sum() / rdenom)
    return {
        "n": float(x.size),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mae": float(np.abs(x - y).mean()),
        "ccc": lin_ccc(x, y),
    }


def git_hash() -> str:
    """Return the short git HEAD hash of the working tree ("" when unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def build_meta(
    model: str | None = None, seed: int | None = None, config: dict | None = None
) -> dict[str, Any]:
    """Build the standard run-metadata block for results JSON.

    Args:
        model: The model name (None when model-free).
        seed: The seed used (None when fully deterministic).
        config: The run configuration to embed verbatim.

    Returns:
        The metadata dict.
    """
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "model": model,
        "seed": seed if seed is not None else DEFAULT_SEED,
    }
    if config is not None:
        meta["config"] = config
    return meta


def write_results(path: Path | str, payload: dict[str, Any], meta: dict[str, Any]) -> Path:
    """Write a results JSON file with run metadata (creating parent dirs).

    Args:
        path: The output path.
        payload: The result payload (merged at the top level).
        meta: The run metadata (see :func:`build_meta`), stored under
            ``"meta"``.

    Returns:
        The written path.

    Raises:
        ValueError: If the payload already contains a ``"meta"`` key.
    """
    if "meta" in payload:
        raise ValueError("payload must not contain a 'meta' key")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {"meta": meta, **payload}
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return out
