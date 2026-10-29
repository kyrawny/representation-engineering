"""
Measurement metrics for the revision (WS2.3).

All functions are numpy-only so they can be used from any analysis script.

- ``lin_ccc``           Lin's concordance correlation coefficient
- ``tracking_fit``      slope/intercept of observed EPA on target EPA
- ``cluster_bootstrap`` percentile CIs resampling whole clusters (scenarios)
- ``mde_paired``        minimum detectable effect for a paired mean difference
- ``holm``              Holm–Bonferroni adjusted p-values
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def lin_ccc(e: np.ndarray, t: np.ndarray) -> float:
    """Lin (1989) concordance correlation between observed ``e`` and target ``t``.

    rho_c = 2 s_et / (s_e^2 + s_t^2 + (mu_e - mu_t)^2), using population moments.
    """
    e = np.asarray(e, float)
    t = np.asarray(t, float)
    me, mt = e.mean(), t.mean()
    se2, st2 = e.var(), t.var()
    set_ = np.mean((e - me) * (t - mt))
    denom = se2 + st2 + (me - mt) ** 2
    return float(2 * set_ / denom) if denom > 0 else float("nan")


def tracking_fit(e: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """OLS fit e = b0 + b1 t.  Ideal tracking is b1 = 1, b0 = 0."""
    e = np.asarray(e, float)
    t = np.asarray(t, float)
    X = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    r = np.corrcoef(e, t)[0, 1] if e.std() > 0 and t.std() > 0 else float("nan")
    return {"b0": float(beta[0]), "b1": float(beta[1]), "r": float(r)}


def cluster_bootstrap(
    clusters: Sequence,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap that resamples clusters with replacement.

    Args:
        clusters: Cluster label per observation (e.g. scenario id).
        stat_fn: Function of an index array (into the observations) that
            returns the statistic.  Called once on ``arange(n)`` for the
            point estimate, then on each resampled index set.
    """
    labels = np.asarray(clusters)
    uniq, inv = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(len(labels))))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        boots[b] = stat_fn(idx)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"est": point, "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))))}


def design_effect(values: np.ndarray, clusters: Sequence) -> float:
    """Kish design effect 1 + (m - 1) ICC for equal-ish cluster sizes (one-way ANOVA ICC)."""
    v = np.asarray(values, float)
    labels = np.asarray(clusters)
    uniq = np.unique(labels)
    k = len(uniq)
    m = len(v) / k
    grand = v.mean()
    means = np.array([v[labels == u].mean() for u in uniq])
    ssb = m * np.sum((means - grand) ** 2)
    ssw = sum(np.sum((v[labels == u] - v[labels == u].mean()) ** 2) for u in uniq)
    msb = ssb / (k - 1)
    msw = ssw / (len(v) - k)
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    icc = max(0.0, icc)
    return float(1 + (m - 1) * icc)


def mde_paired(sd_diff: float, n_eff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean paired difference (two-sided normal approximation)."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd_diff / np.sqrt(n_eff))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
