"""
Statistical testing utilities for the experiment suite.

Provides bootstrap confidence intervals, permutation tests, effect sizes,
and formatting helpers for producing publication-quality statistics.

All functions operate on numpy arrays for speed.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as sp_stats


# =========================================================================
# Bootstrap confidence intervals
# =========================================================================

def bootstrap_ci(
    data: Union[np.ndarray, List[float]],
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for a statistic.

    Args:
        data: 1-D array of observations.
        statistic_fn: Function that computes the statistic from a sample.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(data))

    boot_stats = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic_fn(sample)

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper


def bootstrap_ci_paired(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap CI (resamples indices to preserve pairing).

    Args:
        x: First array of observations.
        y: Second array (same length as x).
        statistic_fn: Function(x_sample, y_sample) → statistic.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y), "x and y must have the same length"

    rng = np.random.default_rng(seed)
    point = float(statistic_fn(x, y))
    n = len(x)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic_fn(x[idx], y[idx])

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper


# =========================================================================
# Permutation tests
# =========================================================================

def paired_permutation_test(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    n_perm: int = 10_000,
    alternative: str = "two-sided",
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Paired permutation test for the mean difference.

    Under H₀, the sign of each difference d_i = x_i − y_i is random.

    Args:
        x: First array.
        y: Second array (same length).
        n_perm: Number of permutations.
        alternative: 'two-sided', 'greater', or 'less'.
        seed: Random seed.

    Returns:
        (observed_mean_diff, p_value)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y)

    diffs = x - y
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    n = len(diffs)

    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = np.mean(diffs * signs)
        if alternative == "two-sided":
            if abs(perm_mean) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if perm_mean >= observed:
                count += 1
        elif alternative == "less":
            if perm_mean <= observed:
                count += 1

    p_value = (count + 1) / (n_perm + 1)  # +1 for the observed statistic
    return observed, float(p_value)


# =========================================================================
# Effect sizes
# =========================================================================

def cohens_d(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
) -> float:
    """
    Compute Cohen's d for paired or independent samples.

    Uses the pooled standard deviation for independent samples.
    For paired data, pass the two paired arrays.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's d (positive means x > y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx, ny = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)

    # Pooled standard deviation
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float(mean_diff / pooled_std)


def cohens_d_paired(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
) -> float:
    """
    Cohen's d for paired samples (uses SD of differences).

    Args:
        x: First paired sample.
        y: Second paired sample (same length).

    Returns:
        Cohen's d_z (paired).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    sd = np.std(diffs, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diffs) / sd)


# =========================================================================
# Standard tests (wrappers with consistent API)
# =========================================================================

def wilcoxon_signed_rank(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Args:
        x: First sample.
        y: Second sample (same length).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        (statistic, p_value)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = sp_stats.wilcoxon(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def mann_whitney_u(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Mann–Whitney U test for independent samples.

    Args:
        x: First sample.
        y: Second sample.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        (statistic, p_value)
    """
    result = sp_stats.mannwhitneyu(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


# =========================================================================
# Formatting helpers
# =========================================================================

def format_ci(
    point: float,
    ci_lo: float,
    ci_hi: float,
    decimal: int = 3,
) -> str:
    """
    Format a point estimate with CI as a LaTeX string.

    Example: ``$0.574\\ [0.521, 0.628]$``
    """
    fmt = f".{decimal}f"
    return f"${point:{fmt}}\\ [{ci_lo:{fmt}},\\ {ci_hi:{fmt}}]$"


def format_p(p: float) -> str:
    """Format a p-value for publication."""
    if p < 0.001:
        return "$p < .001$"
    elif p < 0.01:
        return f"$p = {p:.3f}$"
    elif p < 0.05:
        return f"$p = {p:.3f}$"
    else:
        return f"$p = {p:.2f}$"


def significance_stars(p: float) -> str:
    """Return significance stars: *** / ** / * / ns."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"
