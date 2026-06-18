"""
Calibrated EPA Reader.

Reads EPA values from text using representation reading with optimised
per-layer weights and linear calibration.  Supports five layer-selection
methods (Simple, Greedy, SFFS, Ridge, ElasticNet) and stores the full
configuration for reproducibility.

Typical usage::

    reader = EPAReader.from_tuning_results("epa_reading_tuning_v2_results.json",
                                            rep_readers, method="ElasticNet")
    epa = reader.read_epa(rep_reading_pipeline, "Hello, how are you?")
    # -> {"evaluation": 1.23, "potency": -0.45, "activity": 0.67}
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from .prompt_formatting import DIMENSION_NAMES, DIM_KEY, format_for_reading


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class DimensionConfig:
    """Per-dimension reading configuration."""

    # {layer_index: weight}  — only non-zero layers
    selected_layers: Dict[int, float] = field(default_factory=dict)

    # Sign correction per layer (+1 or -1)
    layer_signs: Dict[int, float] = field(default_factory=dict)

    # Linear calibration: calibrated = slope * raw_reading + intercept
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0


@dataclass
class EPAReaderConfig:
    """Full reader configuration for all three dimensions."""

    dimensions: Dict[str, DimensionConfig] = field(default_factory=dict)
    method: str = "Simple"
    model_name: str = ""

    def to_dict(self) -> Dict:
        """Serialise to a JSON-friendly dict."""
        out: Dict[str, Any] = {
            "method": self.method,
            "model_name": self.model_name,
            "dimensions": {},
        }
        for dim, cfg in self.dimensions.items():
            out["dimensions"][dim] = {
                "selected_layers": {str(k): v for k, v in cfg.selected_layers.items()},
                "layer_signs": {str(k): v for k, v in cfg.layer_signs.items()},
                "calibration_slope": cfg.calibration_slope,
                "calibration_intercept": cfg.calibration_intercept,
            }
        return out

    @classmethod
    def from_dict(cls, d: Dict) -> "EPAReaderConfig":
        """Deserialise from a dict."""
        config = cls(method=d.get("method", "Simple"), model_name=d.get("model_name", ""))
        for dim, dim_d in d.get("dimensions", {}).items():
            dc = DimensionConfig(
                selected_layers={int(k): v for k, v in dim_d["selected_layers"].items()},
                layer_signs={int(k): v for k, v in dim_d["layer_signs"].items()},
                calibration_slope=dim_d.get("calibration_slope", 1.0),
                calibration_intercept=dim_d.get("calibration_intercept", 0.0),
            )
            config.dimensions[dim] = dc
        return config


# =========================================================================
# Core reader class
# =========================================================================

class EPAReader:
    """
    Read calibrated EPA values from text.

    Holds per-dimension layer weights, sign corrections, and linear
    calibration coefficients.  Call ``read_epa()`` to get a dict of
    calibrated EPA values from a single text string.
    """

    def __init__(
        self,
        rep_readers: Dict[str, Any],
        config: EPAReaderConfig,
    ):
        """
        Args:
            rep_readers: Dict mapping dimension name to ``RepReader``.
            config: ``EPAReaderConfig`` with layer weights and calibration.
        """
        self.rep_readers = rep_readers
        self.config = config

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    def read_epa(
        self,
        rep_reading_pipeline,
        text: str,
    ) -> Dict[str, float]:
        """
        Read calibrated EPA from a single text.

        Args:
            rep_reading_pipeline: A HuggingFace ``rep-reading`` pipeline.
            text: The raw text to analyse.

        Returns:
            Dict with ``'evaluation'``, ``'potency'``, ``'activity'`` keys
            mapped to calibrated float values.
        """
        formatted = format_for_reading(text)
        all_layers = sorted(self.rep_readers[DIMENSION_NAMES[0]].directions.keys())

        epa: Dict[str, float] = {}
        for dim in DIMENSION_NAMES:
            cfg = self.config.dimensions[dim]
            reader = self.rep_readers[dim]

            scores = rep_reading_pipeline(
                [formatted],
                hidden_layers=all_layers,
                rep_reader=reader,
                batch_size=1,
                padding=True,
                truncation=True,
            )

            # Weighted average of sign-corrected layer readings
            total = 0.0
            total_weight = 0.0
            for layer, weight in cfg.selected_layers.items():
                sign = cfg.layer_signs.get(layer, 1.0)
                raw = float(scores[0][layer])
                total += weight * sign * raw
                total_weight += weight

            raw_reading = total / total_weight if total_weight > 0 else 0.0
            calibrated = cfg.calibration_slope * raw_reading + cfg.calibration_intercept
            epa[dim] = calibrated

        return epa

    def read_epa_batch(
        self,
        rep_reading_pipeline,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[Dict[str, float]]:
        """
        Read calibrated EPA from a batch of texts.

        Args:
            rep_reading_pipeline: A HuggingFace ``rep-reading`` pipeline.
            texts: List of raw texts.
            batch_size: Batch size for inference.

        Returns:
            List of dicts, one per text, each with ``'evaluation'``,
            ``'potency'``, ``'activity'`` keys.
        """
        formatted = [format_for_reading(t) for t in texts]
        all_layers = sorted(self.rep_readers[DIMENSION_NAMES[0]].directions.keys())

        # Pre-compute all scores per dimension
        dim_all_scores: Dict[str, list] = {}
        for dim in DIMENSION_NAMES:
            reader = self.rep_readers[dim]
            raw_scores = rep_reading_pipeline(
                formatted,
                hidden_layers=all_layers,
                rep_reader=reader,
                batch_size=batch_size,
                padding=True,
                truncation=True,
            )
            dim_all_scores[dim] = raw_scores

        results: List[Dict[str, float]] = []
        for i in range(len(texts)):
            epa: Dict[str, float] = {}
            for dim in DIMENSION_NAMES:
                cfg = self.config.dimensions[dim]
                total = 0.0
                total_weight = 0.0
                for layer, weight in cfg.selected_layers.items():
                    sign = cfg.layer_signs.get(layer, 1.0)
                    raw = float(dim_all_scores[dim][i][layer])
                    total += weight * sign * raw
                    total_weight += weight
                raw_reading = total / total_weight if total_weight > 0 else 0.0
                epa[dim] = cfg.calibration_slope * raw_reading + cfg.calibration_intercept
            results.append(epa)

        return results

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save_config(self, path: str) -> None:
        """Save reader configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_config_file(
        cls,
        path: str,
        rep_readers: Dict[str, Any],
    ) -> "EPAReader":
        """Load an ``EPAReader`` from a saved config JSON."""
        with open(path, "r") as f:
            d = json.load(f)
        config = EPAReaderConfig.from_dict(d)
        return cls(rep_readers=rep_readers, config=config)

    @classmethod
    def from_tuning_results(
        cls,
        results_path: str,
        rep_readers: Dict[str, Any],
        method: str = "ElasticNet",
    ) -> "EPAReader":
        """
        Build an ``EPAReader`` from a ``epa_reading_tuning_v2_results.json``.

        Args:
            results_path: Path to the tuning results JSON.
            rep_readers: Dict of ``RepReader`` objects.
            method: One of ``'Simple'``, ``'Greedy'``, ``'SFFS'``,
                ``'Ridge'``, ``'ElasticNet'``.

        Returns:
            Configured ``EPAReader``.
        """
        with open(results_path, "r") as f:
            results = json.load(f)

        model_name = results.get("metadata", {}).get("model_name", "")
        phase1 = results.get("phase1_correlations", {})
        method_data = results["methods"][method]

        config = EPAReaderConfig(method=method, model_name=model_name)

        for dim in DIMENSION_NAMES:
            md = method_data[dim]
            selected = {int(k): v for k, v in md["selected_layers"].items()}

            signs: Dict[int, float] = {}
            for layer_str, sign_str in md.get("layer_signs", {}).items():
                signs[int(layer_str)] = 1.0 if sign_str == "+" else -1.0

            # If layer_signs not stored, derive from phase1 correlations
            if not signs and dim in phase1:
                for layer_str in selected:
                    rho = phase1[dim].get(str(layer_str), {}).get("rho", 0)
                    signs[layer_str] = 1.0 if rho > 0 else -1.0

            cal = md.get("calibration", {})

            config.dimensions[dim] = DimensionConfig(
                selected_layers=selected,
                layer_signs=signs,
                calibration_slope=cal.get("slope", 1.0),
                calibration_intercept=cal.get("intercept", 0.0),
            )

        return cls(rep_readers=rep_readers, config=config)


# =========================================================================
# Layer selection methods (for building configs from raw readings)
# =========================================================================

def compute_phase1_correlations(
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    all_layers: List[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute per-layer Spearman rho between raw readings and ground truth.

    Args:
        train_scores: ``{dim: {layer: np.array of scores}}``.
        train_gt: ``{dim: np.array of ground-truth EPA values}``.
        all_layers: List of all layer indices.

    Returns:
        ``{dim: {layer: {'rho': float, 'pval': float}}}``.
    """
    phase1: Dict[str, Dict[int, Dict[str, float]]] = {dim: {} for dim in DIMENSION_NAMES}
    for dim in DIMENSION_NAMES:
        for layer in all_layers:
            rho, pval = spearmanr(train_scores[dim][layer], train_gt[dim])
            phase1[dim][layer] = {"rho": float(rho), "pval": float(pval)}
    return phase1


def _eval_rho(
    selected_lw: Dict[int, float],
    dim: str,
    scores_dict: Dict[str, Dict[int, np.ndarray]],
    gt_dict: Dict[str, np.ndarray],
    phase1: Dict[str, Dict[int, Dict[str, float]]],
) -> float:
    """Compute Spearman rho for a weighted layer combination."""
    n = len(list(scores_dict[dim].values())[0])
    total = np.zeros(n)
    total_weight = 0.0
    for layer, weight in selected_lw.items():
        sign = 1.0 if phase1[dim][layer]["rho"] > 0 else -1.0
        total += weight * sign * scores_dict[dim][layer]
        total_weight += weight
    avg = total / total_weight if total_weight > 0 else total
    rho, _ = spearmanr(avg, gt_dict[dim])
    return float(rho)


def select_layers_simple(
    phase1: Dict[str, Dict[int, Dict[str, float]]],
    k: int = 5,
) -> Dict[str, Dict[int, float]]:
    """Uniform average of top-K layers by absolute Spearman rho."""
    result: Dict[str, Dict[int, float]] = {}
    for dim in DIMENSION_NAMES:
        ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:k]
        result[dim] = {l: 1.0 for l, _ in ranked}
    return result


def select_layers_greedy(
    phase1: Dict[str, Dict[int, Dict[str, float]]],
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    top_k: int = 15,
    max_layers: int = 10,
    min_improvement: float = 0.002,
    weight_candidates: Optional[List[float]] = None,
) -> Dict[str, Dict[int, float]]:
    """Greedy forward selection: add the best layer+weight at each step."""
    if weight_candidates is None:
        weight_candidates = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    result: Dict[str, Dict[int, float]] = {}
    for dim in DIMENSION_NAMES:
        ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:top_k]
        candidates = [l for l, _ in ranked]

        selected = {candidates[0]: 1.0}
        cur_rho = _eval_rho(selected, dim, train_scores, train_gt, phase1)

        for _step in range(1, max_layers):
            remaining = [l for l in candidates if l not in selected]
            if not remaining:
                break
            best_score = cur_rho
            best_add: Optional[Tuple[int, float]] = None
            for cand in remaining:
                for w in weight_candidates:
                    trial = {**selected, cand: w}
                    rho = _eval_rho(trial, dim, train_scores, train_gt, phase1)
                    if rho > best_score:
                        best_score = rho
                        best_add = (cand, w)
            if best_add is None or (best_score - cur_rho) < min_improvement:
                break
            selected[best_add[0]] = best_add[1]
            cur_rho = best_score

        result[dim] = dict(selected)
    return result


def select_layers_sffs(
    phase1: Dict[str, Dict[int, Dict[str, float]]],
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    top_k: int = 15,
    max_layers: int = 10,
    min_improvement: float = 0.002,
    weight_candidates: Optional[List[float]] = None,
) -> Dict[str, Dict[int, float]]:
    """Sequential floating forward selection (greedy + backtracking)."""
    if weight_candidates is None:
        weight_candidates = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    result: Dict[str, Dict[int, float]] = {}
    for dim in DIMENSION_NAMES:
        ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:top_k]
        candidates = [l for l, _ in ranked]

        selected = {candidates[0]: 1.0}
        cur_rho = _eval_rho(selected, dim, train_scores, train_gt, phase1)

        for _step in range(1, max_layers):
            # Forward: try adding
            remaining = [l for l in candidates if l not in selected]
            if not remaining:
                break
            best_score = cur_rho
            best_add: Optional[Tuple[int, float]] = None
            for cand in remaining:
                for w in weight_candidates:
                    trial = {**selected, cand: w}
                    rho = _eval_rho(trial, dim, train_scores, train_gt, phase1)
                    if rho > best_score:
                        best_score = rho
                        best_add = (cand, w)
            if best_add is None or (best_score - cur_rho) < min_improvement:
                break
            selected[best_add[0]] = best_add[1]
            cur_rho = best_score

            # Backward: try removing each layer
            improved = True
            while improved and len(selected) > 1:
                improved = False
                best_remove: Optional[int] = None
                best_remove_rho = cur_rho
                for layer_to_remove in list(selected.keys()):
                    trial = {k: v for k, v in selected.items() if k != layer_to_remove}
                    rho = _eval_rho(trial, dim, train_scores, train_gt, phase1)
                    if rho > best_remove_rho:
                        best_remove_rho = rho
                        best_remove = layer_to_remove
                if best_remove is not None:
                    del selected[best_remove]
                    cur_rho = best_remove_rho
                    improved = True

        result[dim] = dict(selected)
    return result


def select_layers_ridge(
    phase1: Dict[str, Dict[int, Dict[str, float]]],
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    all_layers: List[int],
) -> Dict[str, Dict[int, float]]:
    """Non-negative Ridge regression (ElasticNet with l1_ratio≈0)."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    result: Dict[str, Dict[int, float]] = {}
    for dim in DIMENSION_NAMES:
        X = _build_feature_matrix(dim, train_scores, all_layers, phase1)
        y = train_gt[dim]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = ElasticNetCV(
            l1_ratio=[0.01],
            alphas=np.logspace(-4, 4, 50),
            cv=5,
            positive=True,
            max_iter=10000,
        )
        model.fit(X_scaled, y)
        weights = model.coef_ / scaler.scale_

        selected: Dict[int, float] = {}
        for j, layer in enumerate(all_layers):
            if weights[j] > 1e-6:
                selected[layer] = float(round(weights[j], 6))
        result[dim] = selected
    return result


def select_layers_elasticnet(
    phase1: Dict[str, Dict[int, Dict[str, float]]],
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    all_layers: List[int],
) -> Dict[str, Dict[int, float]]:
    """Non-negative ElasticNet regression."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    result: Dict[str, Dict[int, float]] = {}
    for dim in DIMENSION_NAMES:
        X = _build_feature_matrix(dim, train_scores, all_layers, phase1)
        y = train_gt[dim]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 2, 50),
            cv=5,
            positive=True,
            max_iter=10000,
        )
        model.fit(X_scaled, y)
        weights = model.coef_ / scaler.scale_

        selected: Dict[int, float] = {}
        for j, layer in enumerate(all_layers):
            if weights[j] > 1e-6:
                selected[layer] = float(round(weights[j], 6))
        result[dim] = selected
    return result


def _build_feature_matrix(
    dim: str,
    scores_dict: Dict[str, Dict[int, np.ndarray]],
    all_layers: List[int],
    phase1: Dict[str, Dict[int, Dict[str, float]]],
) -> np.ndarray:
    """Build sign-corrected feature matrix (n_samples × n_layers)."""
    n = len(list(scores_dict[dim].values())[0])
    X = np.zeros((n, len(all_layers)))
    for j, layer in enumerate(all_layers):
        sign = 1.0 if phase1[dim][layer]["rho"] > 0 else -1.0
        X[:, j] = sign * scores_dict[dim][layer]
    return X


def fit_calibration(
    selected: Dict[int, float],
    dim: str,
    train_scores: Dict[str, Dict[int, np.ndarray]],
    train_gt: Dict[str, np.ndarray],
    phase1: Dict[str, Dict[int, Dict[str, float]]],
) -> Tuple[float, float]:
    """
    Fit a linear regression calibration mapping raw weighted reading → EPA.

    Returns:
        Tuple of (slope, intercept).
    """
    from sklearn.linear_model import LinearRegression

    n = len(list(train_scores[dim].values())[0])
    total = np.zeros(n)
    total_weight = 0.0
    for layer, weight in selected.items():
        sign = 1.0 if phase1[dim][layer]["rho"] > 0 else -1.0
        total += weight * sign * train_scores[dim][layer]
        total_weight += weight
    raw = total / total_weight if total_weight > 0 else total

    lr = LinearRegression()
    lr.fit(raw.reshape(-1, 1), train_gt[dim])
    return float(lr.coef_[0]), float(lr.intercept_)
