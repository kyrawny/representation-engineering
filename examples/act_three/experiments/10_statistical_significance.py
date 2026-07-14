"""
Experiment 10 — Post-Hoc Statistical Significance Analysis.

Loads results from experiments 02, 03, and 08, runs comprehensive
pairwise significance tests, and produces a summary table suitable
for direct inclusion in the AAAI paper.

Usage::

    python -m examples.act_three.experiments.10_statistical_significance
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import DIMENSION_NAMES
from .setup import save_results, ensure_results_dir
from .stats_utils import (
    bootstrap_ci,
    bootstrap_ci_paired,
    paired_permutation_test,
    cohens_d_paired,
    wilcoxon_signed_rank,
    format_ci,
    format_p,
    significance_stars,
    effect_size_label,
)


def _load_if_exists(filename: str) -> dict:
    """Load results JSON, return empty dict if missing."""
    results_dir = ensure_results_dir()
    path = results_dir / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _extract_distances_02(data: dict) -> dict:
    """Extract per-trial distances from experiment 02."""
    result = {"steered": {dim: [] for dim in DIMENSION_NAMES},
              "unsteered": {dim: [] for dim in DIMENSION_NAMES}}
    for trial in data.get("trials", []):
        for dim in DIMENSION_NAMES:
            target = trial["target_epa"][dim]
            result["steered"][dim].append(
                abs(trial["steered_epa"][dim] - target))
            result["unsteered"][dim].append(
                abs(trial["unsteered_epa"][dim] - target))
    return result


def _extract_distances_03(data: dict) -> dict:
    """Extract per-trial distances from experiment 03."""
    result = {}
    for trial in data.get("trials", []):
        for cond_name, cond_data in trial.get("conditions", {}).items():
            if cond_name not in result:
                result[cond_name] = {dim: [] for dim in DIMENSION_NAMES}
            for dim in DIMENSION_NAMES:
                result[cond_name][dim].append(cond_data["distances"][dim])
    return result


def _extract_distances_08(data: dict) -> dict:
    """Extract per-trial distances from experiment 08."""
    result = {}
    for trial in data.get("trials", []):
        for cond_name, cond_data in trial.get("conditions", {}).items():
            if cond_name not in result:
                result[cond_name] = {dim: [] for dim in DIMENSION_NAMES}
            for dim in DIMENSION_NAMES:
                result[cond_name][dim].append(cond_data["distances"][dim])
    return result


def _run_pairwise(
    name_a: str,
    name_b: str,
    dists_a: dict,
    dists_b: dict,
) -> dict:
    """Run all pairwise tests between two conditions."""
    comparison = {}
    for dim in DIMENSION_NAMES:
        a = np.array(dists_a[dim])
        b = np.array(dists_b[dim])

        if len(a) == 0 or len(b) == 0:
            comparison[dim] = {"error": "no data"}
            continue

        # Ensure same length (for paired tests)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        # Mean improvement (positive = a has larger distance = b is better)
        mean_diff = float(np.mean(a - b))

        # Bootstrap CI on the mean difference
        _, ci_lo, ci_hi = bootstrap_ci_paired(
            a, b,
            statistic_fn=lambda x, y: float(np.mean(x - y)),
        )

        # Permutation test
        _, perm_p = paired_permutation_test(a, b)

        # Wilcoxon
        try:
            _, wilcox_p = wilcoxon_signed_rank(a, b)
        except ValueError:
            wilcox_p = 1.0  # all differences are zero

        # Effect size
        d = cohens_d_paired(a, b)

        comparison[dim] = {
            "n": int(n),
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "permutation_p": float(perm_p),
            "wilcoxon_p": float(wilcox_p),
            "cohens_d": float(d),
            "effect_label": effect_size_label(d),
            "significance": significance_stars(min(perm_p, wilcox_p)),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Post-Hoc Statistical Significance")
    parser.add_argument("--output", default="10_statistical_significance.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    print("Loading experiment results...")
    exp02 = _load_if_exists("02_closed_loop_steering.json")
    exp03 = _load_if_exists("03_prompt_engineering_baseline.json")
    exp08 = _load_if_exists("08_hybrid_steering.json")

    comparisons = {}

    # ---- Exp 02: Steered vs Unsteered ----
    if exp02:
        print("Analysing Exp 02: Steered vs Unsteered...")
        dists = _extract_distances_02(exp02)
        comparisons["steered_vs_unsteered"] = _run_pairwise(
            "unsteered", "steered",
            dists["unsteered"], dists["steered"],
        )

    # ---- Exp 03: All pairwise ----
    if exp03:
        print("Analysing Exp 03: Baseline comparisons...")
        dists = _extract_distances_03(exp03)
        cond_names = list(dists.keys())
        for i, a in enumerate(cond_names):
            for b in cond_names[i + 1:]:
                key = f"{a}_vs_{b}"
                comparisons[key] = _run_pairwise(a, b, dists[a], dists[b])

    # ---- Exp 08: Hybrid comparisons ----
    if exp08:
        print("Analysing Exp 08: Hybrid comparisons...")
        dists = _extract_distances_08(exp08)
        if "hybrid" in dists:
            for baseline in ["unsteered", "repe_only", "pe_only"]:
                if baseline in dists:
                    key = f"hybrid_vs_{baseline}"
                    comparisons[key] = _run_pairwise(
                        baseline, "hybrid",
                        dists[baseline], dists["hybrid"],
                    )

    # ---- Per-dimension summary (best method per dim) ----
    summary = {}
    for dim in DIMENSION_NAMES:
        best_method = None
        best_distance = float("inf")

        # Collect mean distances from exp 08 (most comprehensive)
        if exp08:
            for trial in exp08.get("trials", []):
                for cond_name, cond_data in trial.get("conditions", {}).items():
                    # Use aggregate
                    pass

            agg = exp08.get("aggregate", {})
            for cond_name, cond_agg in agg.items():
                dist = cond_agg.get("per_dimension", {}).get(dim, {}).get(
                    "mean_distance", float("inf"))
                if dist < best_distance:
                    best_distance = dist
                    best_method = cond_name

        summary[dim] = {
            "best_method": best_method,
            "best_distance": best_distance,
        }

    # ---- Generate LaTeX table ----
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pairwise significance tests for steering methods. "
        r"$\Delta$ = mean distance improvement (positive = second method closer to target). "
        r"CI = 95\% bootstrap.}",
        r"\label{tab:significance}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Dim} & $\boldsymbol{\Delta}$ & "
        r"\textbf{95\% CI} & $\boldsymbol{p}$ & \textbf{d} & \textbf{Sig.} \\",
        r"\midrule",
    ]

    for comp_name, comp_data in comparisons.items():
        clean_name = comp_name.replace("_", " ").title()
        first_dim = True
        for dim in DIMENSION_NAMES:
            d = comp_data.get(dim, {})
            if "error" in d:
                continue
            name_col = clean_name if first_dim else ""
            dim_short = dim[0].upper()
            delta = d.get("mean_diff", 0)
            ci_lo = d.get("ci_lower", 0)
            ci_hi = d.get("ci_upper", 0)
            p_val = min(d.get("permutation_p", 1), d.get("wilcoxon_p", 1))
            cd = d.get("cohens_d", 0)
            sig = d.get("significance", "ns")

            latex_lines.append(
                f"{name_col} & {dim_short} & "
                f"{delta:+.3f} & [{ci_lo:.3f}, {ci_hi:.3f}] & "
                f"{p_val:.3f} & {cd:.2f} & {sig} \\\\"
            )
            first_dim = False
        latex_lines.append(r"\midrule")

    # Remove last midrule, replace with bottomrule
    if latex_lines[-1] == r"\midrule":
        latex_lines[-1] = r"\bottomrule"

    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex_table = "\n".join(latex_lines)

    results = {
        "metadata": {
            "experiment": "10_statistical_significance",
            "timestamp": datetime.now().isoformat(),
            "experiments_analysed": {
                "exp02": bool(exp02),
                "exp03": bool(exp03),
                "exp08": bool(exp08),
            },
        },
        "comparisons": comparisons,
        "per_dimension_best": summary,
        "latex_table": latex_table,
    }

    save_results(results, args.output)

    # ---- Print summary ----
    print("\n=== Statistical Significance Summary ===")
    for comp_name, comp_data in comparisons.items():
        print(f"\n  {comp_name}:")
        for dim in DIMENSION_NAMES:
            d = comp_data.get(dim, {})
            if "error" in d:
                continue
            sig = d.get("significance", "ns")
            delta = d.get("mean_diff", 0)
            p = min(d.get("permutation_p", 1), d.get("wilcoxon_p", 1))
            cd = d.get("cohens_d", 0)
            print(f"    {dim}: Δ={delta:+.3f}, p={p:.4f}, d={cd:.2f} {sig}")

    print(f"\nLaTeX table saved in results JSON.")


if __name__ == "__main__":
    main()
