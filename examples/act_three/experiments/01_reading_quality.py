"""
Experiment 01 — Reading Quality Evaluation.

Loads the calibration test set, reads EPA with the calibrated reader,
and produces predicted-vs-ground-truth data for scatter plots and
per-dimension quality metrics.

Usage::

    python -m examples.act_three.experiments.01_reading_quality
    python -m examples.act_three.experiments.01_reading_quality --quick
"""

import argparse
import time
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from .config import DIMENSION_NAMES, TUNING_TEST_PATH, TUNING_TRAIN_PATH
from .setup import (
    load_experiment_components,
    save_results,
)
from .scenarios import get_scenarios


def _load_tuning_dataset(path: str) -> list:
    """Load a tuning dataset JSON file.

    Handles both wrapped format ``{"metadata": ..., "utterances": [...]}``
    and flat list-of-dicts format.
    """
    import json
    with open(path, "r") as f:
        raw = json.load(f)
    # Unwrap if in {"metadata": ..., "utterances": [...]} format
    if isinstance(raw, dict) and "utterances" in raw:
        return raw["utterances"]
    return raw


def _compute_metrics(y_true, y_pred):
    """Compute all reading quality metrics."""
    rho, rho_p = spearmanr(y_true, y_pred)
    r, r_p = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(rho_p),
        "pearson_r": float(r),
        "pearson_pval": float(r_p),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "n": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 01: Reading Quality")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="01_reading_quality.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    # ---- Load components (no steerer needed) ----
    comp = load_experiment_components(load_steerer=False)

    # ---- Load test set ----
    test_data = _load_tuning_dataset(TUNING_TEST_PATH)
    if args.quick:
        test_data = test_data[:20]
    print(f"Evaluating on {len(test_data)} test utterances")

    # ---- Read EPA for each utterance ----
    ground_truth = {dim: [] for dim in DIMENSION_NAMES}
    predictions = {dim: [] for dim in DIMENSION_NAMES}

    texts = [item["text"] for item in test_data]
    gt_epas = [item["target_epa"] for item in test_data]


    print("Reading EPA values...")
    start_time = time.time()
    pred_epas = comp.reader.read_epa_batch(
        comp.rep_reading_pipeline, texts, batch_size=8,
    )
    read_time = time.time() - start_time
    print(f"Read {len(texts)} utterances in {read_time:.1f}s "
          f"({read_time/len(texts):.2f}s each)")

    for gt, pred in zip(gt_epas, pred_epas):
        for dim, key in [("evaluation", "e"), ("potency", "p"), ("activity", "a")]:
            ground_truth[dim].append(gt[key])
            predictions[dim].append(pred[dim])

    # ---- Compute metrics ----
    metrics = {}
    scatter_data = {}
    for dim in DIMENSION_NAMES:
        y_true = np.array(ground_truth[dim])
        y_pred = np.array(predictions[dim])
        metrics[dim] = _compute_metrics(y_true, y_pred)
        scatter_data[dim] = {
            "ground_truth": y_true.tolist(),
            "predicted": y_pred.tolist(),
        }
        print(f"  {dim:>12s}: ρ={metrics[dim]['spearman_rho']:.4f}, "
              f"r={metrics[dim]['pearson_r']:.4f}, "
              f"MAE={metrics[dim]['mae']:.4f}, "
              f"R²={metrics[dim]['r2']:.4f}")

    # ---- Also read the experiment scenarios (no GT, but useful) ----
    scenarios = get_scenarios(quick=args.quick)
    scenario_texts = [s["text"] for s in scenarios]
    print(f"\nReading {len(scenario_texts)} experiment scenarios...")
    scenario_epas = comp.reader.read_epa_batch(
        comp.rep_reading_pipeline, scenario_texts, batch_size=8,
    )
    scenario_readings = []
    for s, epa in zip(scenarios, scenario_epas):
        scenario_readings.append({
            "id": s["id"],
            "category": s["category"],
            "text": s["text"],
            "read_epa": epa,
        })

    # ---- Save ----
    results = {
        "metadata": {
            "experiment": "01_reading_quality",
            "timestamp": datetime.now().isoformat(),
            "n_test": len(test_data),
            "n_scenarios": len(scenarios),
            "read_time_seconds": read_time,
            "quick_mode": args.quick,
        },
        "test_metrics": metrics,
        "test_scatter": scatter_data,
        "scenario_readings": scenario_readings,
    }
    save_results(results, args.output)


if __name__ == "__main__":
    main()
