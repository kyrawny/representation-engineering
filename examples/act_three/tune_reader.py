"""
Standalone EPA Reader Tuning CLI.

Runs the full reader tuning pipeline (layer selection + calibration) from
the command line.  Replaces the notebook-based workflow.

Usage::

    python -m examples.act_three.tune_reader
    python -m examples.act_three.tune_reader --directions epa_directions_ortho.pkl \\
        --method ElasticNet --output epa_reading_tuning_v2_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(
        description="Tune the EPA reader (layer selection + calibration)")
    parser.add_argument(
        "--directions", default=None,
        help="Path to directions pickle (default: epa_directions.pkl)")
    parser.add_argument(
        "--train-data", default=None,
        help="Training dataset JSON (default: epa_tuning_dataset_train.json)")
    parser.add_argument(
        "--test-data", default=None,
        help="Test dataset JSON (default: epa_tuning_dataset_test.json)")
    parser.add_argument(
        "--method", default="ElasticNet",
        choices=["Simple", "Greedy", "SFFS", "Ridge", "ElasticNet"],
        help="Layer selection method (default: ElasticNet)")
    parser.add_argument(
        "--output", default=None,
        help="Output results JSON (default: epa_reading_tuning_v2_results.json)")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for metadata")
    parser.add_argument(
        "--simple-k", type=int, default=5,
        help="Top-K for Simple method (default: 5)")
    args = parser.parse_args()

    # ---- Resolve paths ----
    act_three_dir = Path(__file__).resolve().parent
    repo_root = act_three_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from examples.act_three.model_registry import get_short_name

    short_name = get_short_name(args.model)
    model_dir = act_three_dir / "models" / short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir}")

    # Resolve directions path — model-aware default
    directions_path = args.directions or str(model_dir / "epa_directions.pkl")
    if not Path(directions_path).is_absolute() and not Path(directions_path).exists():
        candidate = model_dir / directions_path
        if candidate.exists():
            directions_path = str(candidate)
    train_path = args.train_data or str(model_dir / "epa_tuning_dataset_train.json")
    test_path = args.test_data or str(model_dir / "epa_tuning_dataset_test.json")
    output_path = args.output or str(model_dir / "epa_reading_tuning_v2_results.json")

    from examples.act_three.direction_extraction import load_directions
    from examples.act_three.prompt_formatting import DIMENSION_NAMES, format_for_reading
    from examples.act_three.epa_reader import (
        compute_phase1_correlations,
        select_layers_simple,
        select_layers_greedy,
        select_layers_sffs,
        select_layers_ridge,
        select_layers_elasticnet,
        fit_calibration,
    )

    # ---- Load directions ----
    print(f"Loading directions from: {directions_path}")
    saved = load_directions(directions_path)
    rep_readers = saved["rep_readers"]
    hidden_layers = saved["hidden_layers"]
    all_layers = sorted(hidden_layers)

    # ---- Load datasets ----
    print(f"Loading training data from: {train_path}")
    with open(train_path, "r") as f:
        train_raw = json.load(f)
    print(f"Loading test data from: {test_path}")
    with open(test_path, "r") as f:
        test_raw = json.load(f)

    def _parse_dataset(raw) -> list:
        """Parse dataset JSON, handling multiple known formats."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # Format: {"utterances": [...]} (from 00_setup_datasets.py)
            if "utterances" in raw:
                return raw["utterances"]
            # Format: {"data": [...]} or {"items": [...]}
            if "data" in raw:
                return raw["data"]
            if "items" in raw:
                return raw["items"]
        return []

    train_items = _parse_dataset(train_raw)
    test_items = _parse_dataset(test_raw)
    print(f"  Train: {len(train_items)} items, Test: {len(test_items)} items")

    if len(train_items) == 0:
        print("ERROR: No training items found. Check dataset format.")
        print(f"  Dataset keys: {list(train_raw.keys()) if isinstance(train_raw, dict) else 'list'}")
        sys.exit(1)

    # ---- Compute raw scores via rep-reading pipeline ----
    print("Loading model for rep-reading pipeline...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    from repe import repe_pipeline_registry

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    repe_pipeline_registry()
    rep_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    def compute_scores(items: list) -> tuple:
        """Compute raw per-layer scores and ground truth arrays."""
        texts = [item["text"] for item in items]
        formatted = [format_for_reading(t) for t in texts]

        gt = {}
        for dim in DIMENSION_NAMES:
            gt[dim] = np.array([item["target_epa"][dim[0]] for item in items])

        scores: Dict[str, Dict[int, np.ndarray]] = {dim: {} for dim in DIMENSION_NAMES}
        for dim in DIMENSION_NAMES:
            print(f"  Computing {dim} scores...")
            reader = rep_readers[dim]
            raw = rep_pipeline(
                formatted,
                hidden_layers=all_layers,
                rep_reader=reader,
                batch_size=8,
                padding=True,
                truncation=True,
            )
            for layer in all_layers:
                scores[dim][layer] = np.array([float(raw[i][layer]) for i in range(len(texts))])

        return scores, gt

    print("\nComputing training scores...")
    t0 = time.time()
    train_scores, train_gt = compute_scores(train_items)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Computing test scores...")
    t0 = time.time()
    test_scores, test_gt = compute_scores(test_items)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Phase 1: Per-layer correlations ----
    print("\nPhase 1: Computing per-layer correlations...")
    phase1 = compute_phase1_correlations(train_scores, train_gt, all_layers)

    # ---- Phase 2: Layer selection ----
    print(f"\nPhase 2: Layer selection (method={args.method})...")

    method_map = {
        "Simple": lambda: select_layers_simple(phase1, k=args.simple_k),
        "Greedy": lambda: select_layers_greedy(phase1, train_scores, train_gt),
        "SFFS": lambda: select_layers_sffs(phase1, train_scores, train_gt),
        "Ridge": lambda: select_layers_ridge(phase1, train_scores, train_gt, all_layers),
        "ElasticNet": lambda: select_layers_elasticnet(phase1, train_scores, train_gt, all_layers),
    }

    # Run all methods for comparison
    results = {
        "metadata": {
            "model_name": args.model,
            "timestamp": datetime.now().isoformat(),
            "n_train": len(train_items),
            "n_test": len(test_items),
            "directions_path": directions_path,
            "orthogonalised": saved.get("orthogonalised", False),
        },
        "phase1_correlations": {},
        "methods": {},
    }

    # Store phase1
    for dim in DIMENSION_NAMES:
        results["phase1_correlations"][dim] = {
            str(k): v for k, v in phase1[dim].items()
        }

    for method_name, method_fn in method_map.items():
        print(f"\n  --- {method_name} ---")
        selected = method_fn()
        method_results: Dict[str, Any] = {}

        for dim in DIMENSION_NAMES:
            # Fit calibration
            slope, intercept = fit_calibration(
                selected[dim], dim, train_scores, train_gt, phase1)

            # Build signs
            signs = {}
            for layer in selected[dim]:
                rho = phase1[dim][layer]["rho"]
                signs[layer] = "+" if rho > 0 else "-"

            # Evaluate on test
            n = len(test_items)
            total = np.zeros(n)
            total_weight = 0.0
            for layer, weight in selected[dim].items():
                sign = 1.0 if phase1[dim][layer]["rho"] > 0 else -1.0
                total += weight * sign * test_scores[dim][layer]
                total_weight += weight
            raw_test = total / total_weight if total_weight > 0 else total
            pred_test = slope * raw_test + intercept
            gt_test = test_gt[dim]

            rho_test, _ = spearmanr(pred_test, gt_test)
            r_test, _ = pearsonr(pred_test, gt_test)
            mae_test = mean_absolute_error(gt_test, pred_test)
            rmse_test = float(np.sqrt(mean_squared_error(gt_test, pred_test)))
            r2_test = r2_score(gt_test, pred_test)

            # Train metrics
            total_tr = np.zeros(len(train_items))
            total_weight_tr = 0.0
            for layer, weight in selected[dim].items():
                sign = 1.0 if phase1[dim][layer]["rho"] > 0 else -1.0
                total_tr += weight * sign * train_scores[dim][layer]
                total_weight_tr += weight
            raw_tr = total_tr / total_weight_tr if total_weight_tr > 0 else total_tr
            pred_tr = slope * raw_tr + intercept
            rho_tr, _ = spearmanr(pred_tr, train_gt[dim])

            method_results[dim] = {
                "selected_layers": {str(k): v for k, v in selected[dim].items()},
                "layer_signs": {str(k): v for k, v in signs.items()},
                "n_layers": len(selected[dim]),
                "rho_train": float(rho_tr),
                "calibration": {
                    "slope": float(slope),
                    "intercept": float(intercept),
                },
                "metrics_train": {
                    "spearman": float(rho_tr),
                },
                "metrics_test": {
                    "spearman": float(rho_test),
                    "pearson": float(r_test),
                    "mae": float(mae_test),
                    "rmse": float(rmse_test),
                    "r2": float(r2_test),
                },
                "test_scatter": {
                    "ground_truth": gt_test.tolist(),
                    "predicted": pred_test.tolist(),
                },
            }

            print(f"    {dim}: ρ_train={rho_tr:.3f}, ρ_test={rho_test:.3f}, "
                  f"MAE={mae_test:.3f}, layers={len(selected[dim])}")

        results["methods"][method_name] = method_results

    # ---- Save ----
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ---- Summary ----
    best = results["methods"][args.method]
    print(f"\n=== {args.method} Summary ===")
    for dim in DIMENSION_NAMES:
        m = best[dim]["metrics_test"]
        print(f"  {dim}: ρ={m['spearman']:.3f}, r={m['pearson']:.3f}, "
              f"MAE={m['mae']:.3f}, R²={m['r2']:.3f}")


if __name__ == "__main__":
    main()
