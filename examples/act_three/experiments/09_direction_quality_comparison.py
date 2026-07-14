"""
Experiment 09 — Direction Quality Comparison.

Before/after comparison of direction extraction improvements:
1. Original directions vs improved extraction templates
2. Non-orthogonalised vs orthogonalised directions
3. Per-dimension reader quality metrics

This script loads both direction sets (if available), reads EPA from the
test dataset with each, and produces a comparison table.

Usage::

    python -m examples.act_three.experiments.09_direction_quality_comparison
    python -m examples.act_three.experiments.09_direction_quality_comparison --quick
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    DIMENSION_NAMES,
    DIRECTIONS_PATH,
    DIRECTIONS_PATH_ORTHO,
    READING_RESULTS_PATH,
)
from .setup import save_results


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 09: Direction Quality Comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Use a subset of test data")
    parser.add_argument("--output", default="09_direction_quality_comparison.json",
                        help="Output filename in results/")
    parser.add_argument("--backup-dir", default=None,
                        help="Path to backup v1 directions (default: backups/v1/)")
    args = parser.parse_args()

    act_three_dir = Path(__file__).resolve().parent.parent
    backup_dir = args.backup_dir or str(act_three_dir / "backups" / "v1")

    import pickle

    results = {
        "metadata": {
            "experiment": "09_direction_quality_comparison",
            "timestamp": datetime.now().isoformat(),
            "quick_mode": args.quick,
        },
        "direction_sets": {},
        "cosine_similarities": {},
    }

    # ---- Discover available direction sets ----
    direction_sets = {}

    # Original (v1 backup)
    v1_path = os.path.join(backup_dir, "epa_directions.pkl")
    if os.path.exists(v1_path):
        with open(v1_path, "rb") as f:
            direction_sets["original_v1"] = pickle.load(f)
        print(f"Loaded original v1 directions from {v1_path}")
    else:
        print(f"No v1 backup found at {v1_path}")

    # Current (may be re-extracted with new templates)
    if os.path.exists(DIRECTIONS_PATH):
        with open(DIRECTIONS_PATH, "rb") as f:
            direction_sets["current"] = pickle.load(f)
        print(f"Loaded current directions from {DIRECTIONS_PATH}")

    # Orthogonalised
    if os.path.exists(DIRECTIONS_PATH_ORTHO):
        with open(DIRECTIONS_PATH_ORTHO, "rb") as f:
            direction_sets["orthogonalised"] = pickle.load(f)
        print(f"Loaded orthogonalised directions from {DIRECTIONS_PATH_ORTHO}")

    if len(direction_sets) < 2:
        print("WARNING: Need at least 2 direction sets for comparison. "
              "Run extract_directions.py and/or orthogonalise_directions.py first.")

    # ---- Compute per-layer cosine similarities between sets ----
    set_names = list(direction_sets.keys())
    for i, name_a in enumerate(set_names):
        for name_b in set_names[i + 1:]:
            readers_a = direction_sets[name_a]["rep_readers"]
            readers_b = direction_sets[name_b]["rep_readers"]

            pair_key = f"{name_a}_vs_{name_b}"
            pair_result = {}

            for dim in DIMENSION_NAMES:
                layers = sorted(readers_a[dim].directions.keys())
                cosines = []
                per_layer = {}
                for layer in layers:
                    dir_a = np.array(readers_a[dim].directions[layer]).flatten()
                    dir_b = np.array(readers_b[dim].directions[layer]).flatten()
                    cos = _cosine_similarity(dir_a, dir_b)
                    cosines.append(cos)
                    per_layer[str(layer)] = cos

                pair_result[dim] = {
                    "mean_cosine": float(np.mean(cosines)),
                    "std_cosine": float(np.std(cosines)),
                    "min_cosine": float(np.min(cosines)),
                    "max_cosine": float(np.max(cosines)),
                    "per_layer": per_layer,
                }

            results["cosine_similarities"][pair_key] = pair_result

    # ---- Cross-dimension cosines (entanglement measure) ----
    for set_name, saved in direction_sets.items():
        readers = saved["rep_readers"]
        layers = sorted(readers["evaluation"].directions.keys())

        cross_dim = {}
        for dim_a, dim_b in [("evaluation", "potency"),
                              ("evaluation", "activity"),
                              ("potency", "activity")]:
            cosines = []
            for layer in layers:
                dir_a = np.array(readers[dim_a].directions[layer]).flatten()
                dir_b = np.array(readers[dim_b].directions[layer]).flatten()
                cosines.append(abs(_cosine_similarity(dir_a, dir_b)))

            cross_dim[f"{dim_a}_vs_{dim_b}"] = {
                "mean_abs_cosine": float(np.mean(cosines)),
                "max_abs_cosine": float(np.max(cosines)),
            }

        results["direction_sets"][set_name] = {
            "n_layers": len(layers),
            "orthogonalised": saved.get("orthogonalised", False),
            "cross_dimension_cosines": cross_dim,
        }

    # ---- If reader tuning results exist, compare reader quality ----
    if os.path.exists(READING_RESULTS_PATH):
        with open(READING_RESULTS_PATH, "r") as f:
            reading_results = json.load(f)

        reader_quality = {}
        for method_name, method_data in reading_results.get("methods", {}).items():
            method_quality = {}
            for dim in DIMENSION_NAMES:
                if dim in method_data:
                    metrics = method_data[dim].get("metrics_test", {})
                    method_quality[dim] = {
                        "spearman": metrics.get("spearman"),
                        "pearson": metrics.get("pearson"),
                        "mae": metrics.get("mae"),
                        "r2": metrics.get("r2"),
                    }
            reader_quality[method_name] = method_quality

        results["reader_quality"] = reader_quality

    # ---- Save ----
    save_results(results, args.output)

    # ---- Print summary ----
    print("\n=== Cross-Dimension Entanglement ===")
    print(f"{'Set':<20} {'E·P':>8} {'E·A':>8} {'P·A':>8}")
    print("-" * 46)
    for set_name, info in results["direction_sets"].items():
        cd = info["cross_dimension_cosines"]
        ep = cd["evaluation_vs_potency"]["mean_abs_cosine"]
        ea = cd["evaluation_vs_activity"]["mean_abs_cosine"]
        pa = cd["potency_vs_activity"]["mean_abs_cosine"]
        print(f"{set_name:<20} {ep:>8.4f} {ea:>8.4f} {pa:>8.4f}")

    if results["cosine_similarities"]:
        print("\n=== Direction Similarity Between Sets ===")
        for pair, dims in results["cosine_similarities"].items():
            print(f"\n  {pair}:")
            for dim in DIMENSION_NAMES:
                mc = dims[dim]["mean_cosine"]
                print(f"    {dim}: mean cosine = {mc:.4f}")


if __name__ == "__main__":
    main()
