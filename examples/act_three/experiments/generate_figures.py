"""
Generate all paper figures from saved experiment results.

Produces publication-ready figures as both PDF and PNG.

Usage::

    python -m examples.act_three.experiments.generate_figures
    python -m examples.act_three.experiments.generate_figures --format png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DIMENSION_NAMES, RESULTS_DIR, FIGURES_DIR, READING_RESULTS_PATH
from .setup import load_results


# =========================================================================
# Style setup
# =========================================================================

def _setup_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


DIM_COLORS = {
    "evaluation": "#2ecc71",
    "potency": "#3498db",
    "activity": "#e74c3c",
}
DIM_LABELS = {
    "evaluation": "Evaluation (E)",
    "potency": "Potency (P)",
    "activity": "Activity (A)",
}


def _save_fig(fig, name: str, fmt: str = "pdf"):
    """Save a figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    print(f"  Saved {path}")
    # Also save PNG if primary format is PDF
    if fmt == "pdf":
        png_path = FIGURES_DIR / f"{name}.png"
        fig.savefig(png_path, dpi=300)


# =========================================================================
# Figure 1: Reading calibration scatter plots
# =========================================================================

def fig_reading_scatter(fmt: str = "pdf"):
    """3-panel scatter: predicted vs ground-truth EPA (test set)."""
    plt = _setup_style()
    data = load_results("01_reading_quality.json")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for i, dim in enumerate(DIMENSION_NAMES):
        ax = axes[i]
        gt = np.array(data["test_scatter"][dim]["ground_truth"])
        pred = np.array(data["test_scatter"][dim]["predicted"])
        metrics = data["test_metrics"][dim]

        ax.scatter(gt, pred, alpha=0.4, s=15, color=DIM_COLORS[dim],
                   edgecolors="none")

        # Perfect prediction line
        lims = [min(gt.min(), pred.min()) - 0.5,
                max(gt.max(), pred.max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        # Linear fit
        z = np.polyfit(gt, pred, 1)
        p = np.poly1d(z)
        ax.plot(sorted(gt), p(sorted(gt)), color=DIM_COLORS[dim],
                linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Ground Truth (ACT Dictionary)")
        ax.set_ylabel("Predicted (RepE Reader)")
        ax.set_title(DIM_LABELS[dim])
        ax.text(
            0.05, 0.95,
            f"ρ = {metrics['spearman_rho']:.3f}\n"
            f"r = {metrics['pearson_r']:.3f}\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"MAE = {metrics['mae']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8),
        )

    fig.suptitle("Calibrated EPA Reading Quality (Test Set)", fontsize=13,
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig01_reading_scatter", fmt)
    plt.close(fig)


# =========================================================================
# Figure 2: Per-layer correlation heatmap
# =========================================================================

def fig_layer_correlations(fmt: str = "pdf"):
    """Heatmap of per-layer Spearman ρ for each EPA dimension."""
    plt = _setup_style()
    import seaborn as sns

    with open(READING_RESULTS_PATH, "r") as f:
        tuning = json.load(f)

    phase1 = tuning["phase1_correlations"]

    layers = sorted([int(k) for k in phase1["evaluation"].keys()])
    matrix = np.zeros((3, len(layers)))

    for i, dim in enumerate(DIMENSION_NAMES):
        for j, layer in enumerate(layers):
            matrix[i, j] = phase1[dim][str(layer)]["rho"]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    sns.heatmap(
        matrix, cmap="RdBu_r", center=0, vmin=-0.7, vmax=0.7,
        xticklabels=[str(l) for l in layers],
        yticklabels=[DIM_LABELS[d] for d in DIMENSION_NAMES],
        ax=ax, cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    )
    ax.set_xlabel("Layer Index")
    ax.set_title("Per-Layer Reading Correlation with Ground-Truth EPA")
    plt.tight_layout()
    _save_fig(fig, "fig02_layer_correlations", fmt)
    plt.close(fig)


# =========================================================================
# Figure 3: Closed-loop steering scatter
# =========================================================================

def fig_steering_scatter(fmt: str = "pdf"):
    """3-panel scatter: target EPA vs achieved steered EPA."""
    plt = _setup_style()
    data = load_results("02_closed_loop_steering.json")
    trials = data["trials"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for i, dim in enumerate(DIMENSION_NAMES):
        ax = axes[i]
        targets = [t["target_epa"][dim] for t in trials]
        achieved = [t["steered_epa"][dim] for t in trials]
        unsteered = [t["unsteered_epa"][dim] for t in trials]

        ax.scatter(targets, unsteered, alpha=0.2, s=10, color="gray",
                   label="Unsteered", edgecolors="none")
        ax.scatter(targets, achieved, alpha=0.4, s=15,
                   color=DIM_COLORS[dim], label="Steered",
                   edgecolors="none")

        lims = [min(min(targets), min(achieved)) - 0.5,
                max(max(targets), max(achieved)) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Target EPA")
        ax.set_ylabel("Achieved EPA")
        ax.set_title(DIM_LABELS[dim])
        ax.legend(fontsize=7, loc="lower right")

        # Add hit rate annotation
        metrics = data["aggregate_metrics"]["per_dimension"][dim]
        ax.text(
            0.05, 0.95,
            f"Hit rate: {metrics['hit_rate']:.1%}\n"
            f"Δ dist: {metrics['mean_distance_improvement']:+.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8),
        )

    fig.suptitle("Closed-Loop Steering: Target vs Achieved EPA", fontsize=13,
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig03_steering_scatter", fmt)
    plt.close(fig)


# =========================================================================
# Figure 4: Steering hit rates bar chart
# =========================================================================

def fig_steering_bars(fmt: str = "pdf"):
    """Bar chart of per-dimension hit rates and distance improvement."""
    plt = _setup_style()
    data = load_results("02_closed_loop_steering.json")
    metrics = data["aggregate_metrics"]["per_dimension"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    x = np.arange(3)
    dims = DIMENSION_NAMES

    # Hit rates
    hit_rates = [metrics[d]["hit_rate"] for d in dims]
    bars1 = ax1.bar(x, hit_rates, color=[DIM_COLORS[d] for d in dims],
                    alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5,
                label="Chance (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([DIM_LABELS[d] for d in dims])
    ax1.set_ylabel("Hit Rate")
    ax1.set_title("Fraction Closer to Target")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    for bar, val in zip(bars1, hit_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontsize=9)

    # Distance improvement
    improvements = [metrics[d]["mean_distance_improvement"] for d in dims]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in improvements]
    bars2 = ax2.bar(x, improvements, color=colors, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DIM_LABELS[d] for d in dims])
    ax2.set_ylabel("Mean Distance Improvement")
    ax2.set_title("Distance to Target: Unsteered − Steered")
    for bar, val in zip(bars2, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01 * np.sign(val),
                 f"{val:+.3f}", ha="center", fontsize=9, va="bottom")

    plt.tight_layout()
    _save_fig(fig, "fig04_steering_bars", fmt)
    plt.close(fig)


# =========================================================================
# Figure 5: Coefficient sweep curves
# =========================================================================

def fig_coefficient_sweep(fmt: str = "pdf"):
    """3-panel line plots: EPA reading vs steering coefficient."""
    plt = _setup_style()
    data = load_results("04_coefficient_sweep.json")
    curves = data["sweep_curves"]
    coeffs = [float(c) for c in sorted(curves["evaluation"].keys(),
                                        key=float)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for i, dim in enumerate(DIMENSION_NAMES):
        ax = axes[i]
        means = [curves[dim][str(c)]["mean_on_target"] for c in coeffs]
        stds = [curves[dim][str(c)]["std_on_target"] for c in coeffs]

        ax.plot(coeffs, means, "-o", color=DIM_COLORS[dim],
                markersize=4, linewidth=1.5, label=DIM_LABELS[dim])
        ax.fill_between(coeffs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=DIM_COLORS[dim])

        # Cross-dimensional interference
        for other_dim in DIMENSION_NAMES:
            if other_dim == dim:
                continue
            cross_means = [
                curves[dim][str(c)]["cross_dimension_means"][other_dim]
                for c in coeffs
            ]
            ax.plot(coeffs, cross_means, "--", color=DIM_COLORS[other_dim],
                    alpha=0.5, linewidth=1, label=f"{DIM_LABELS[other_dim]} (cross)")

        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel("Achieved EPA Reading")
        ax.set_title(f"Steering {DIM_LABELS[dim]}")
        ax.legend(fontsize=6, loc="best")
        ax.axhline(0, color="gray", alpha=0.2, linewidth=0.5)

    fig.suptitle("Steering Coefficient vs Achieved EPA", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig05_coefficient_sweep", fmt)
    plt.close(fig)


# =========================================================================
# Figure 6: Identity generalisation heatmap
# =========================================================================

def fig_identity_heatmap(fmt: str = "pdf"):
    """Heatmap of target EPAs across identity pairs for each scenario."""
    plt = _setup_style()
    import seaborn as sns

    data = load_results("06_identity_generalization.json")
    trials = data["trials"]

    # Build matrix: scenarios × pairs × 3 dims
    pairs = [r["identity_pair"] for r in trials[0]["identity_pair_results"]]
    n_scenarios = len(trials)
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, n_scenarios * 0.5)))

    for d, dim in enumerate(DIMENSION_NAMES):
        ax = axes[d]
        matrix = np.zeros((n_scenarios, n_pairs))
        scenario_labels = []

        for i, trial in enumerate(trials):
            scenario_labels.append(trial["scenario_id"])
            for j, pair_result in enumerate(trial["identity_pair_results"]):
                matrix[i, j] = pair_result["target_epa"][dim]

        sns.heatmap(
            matrix, cmap="RdBu_r", center=0, ax=ax,
            xticklabels=[p.replace("_", "\n") for p in pairs],
            yticklabels=scenario_labels,
            cbar_kws={"shrink": 0.6},
            annot=True, fmt=".1f", annot_kws={"size": 7},
        )
        ax.set_title(DIM_LABELS[dim])

    fig.suptitle("ACT-Computed Target EPA by Identity Pair", fontsize=13,
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig06_identity_heatmap", fmt)
    plt.close(fig)


# =========================================================================
# Figure 7: Coherence comparison
# =========================================================================

def fig_coherence(fmt: str = "pdf"):
    """Violin plots comparing coherence metrics: steered vs unsteered."""
    plt = _setup_style()

    data = load_results("07_coherence_evaluation.json")
    trials = data["trials"]

    # Determine available perplexity models
    ppl_models = []
    for key in trials[0]:
        if key.startswith("unsteered_ppl_"):
            ppl_models.append(key.replace("unsteered_ppl_", ""))

    n_plots = 2 + len(ppl_models)  # unique token ratio, bigram rep, + PPLs
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Unique token ratio
    ax = axes[0]
    un_utr = [t["unsteered_text_metrics"]["unique_token_ratio"] for t in trials]
    st_utr = [t["steered_text_metrics"]["unique_token_ratio"] for t in trials]
    parts = ax.violinplot([un_utr, st_utr], positions=[0, 1],
                          showmeans=True, showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unsteered", "Steered"])
    ax.set_ylabel("Unique Token Ratio")
    ax.set_title("Lexical Diversity")

    # Bigram repetition
    ax = axes[1]
    un_br = [t["unsteered_text_metrics"]["bigram_repetition_ratio"] for t in trials]
    st_br = [t["steered_text_metrics"]["bigram_repetition_ratio"] for t in trials]
    parts = ax.violinplot([un_br, st_br], positions=[0, 1],
                          showmeans=True, showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unsteered", "Steered"])
    ax.set_ylabel("Bigram Repetition Ratio")
    ax.set_title("Repetitiveness")

    # Perplexity plots
    for k, model_name in enumerate(ppl_models):
        ax = axes[2 + k]
        un_ppl = [t[f"unsteered_ppl_{model_name}"] for t in trials
                  if t.get(f"unsteered_ppl_{model_name}") is not None]
        st_ppl = [t[f"steered_ppl_{model_name}"] for t in trials
                  if t.get(f"steered_ppl_{model_name}") is not None]

        # Filter infinities for display
        un_ppl = [p for p in un_ppl if p < 1e6]
        st_ppl = [p for p in st_ppl if p < 1e6]

        if un_ppl and st_ppl:
            parts = ax.violinplot([un_ppl, st_ppl], positions=[0, 1],
                                  showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Unsteered", "Steered"])
            ax.set_ylabel("Perplexity")
            ax.set_title(f"Perplexity ({model_name.upper()})")

    fig.suptitle("Coherence: Steered vs Unsteered", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig07_coherence", fmt)
    plt.close(fig)


# =========================================================================
# Figure 8: Baseline comparison (prompt engineering)
# =========================================================================

def fig_baseline_comparison(fmt: str = "pdf"):
    """Grouped bar chart: distance to target for 3 methods."""
    plt = _setup_style()
    data = load_results("03_prompt_baseline.json")
    metrics = data["comparison_metrics"]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(3)
    width = 0.25
    methods = [("unsteered", "Unsteered", "#95a5a6"),
               ("prompt_engineered", "Prompt Eng.", "#f39c12"),
               ("steered", "RepE Steered", "#2ecc71")]

    for j, (key, label, color) in enumerate(methods):
        dists = [metrics[key][dim]["mean_distance"] for dim in DIMENSION_NAMES]
        bars = ax.bar(x + j * width, dists, width, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSION_NAMES])
    ax.set_ylabel("Mean Distance to Target")
    ax.set_title("Baseline Comparison: Distance to ACT-Optimal EPA")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "fig08_baseline_comparison", fmt)
    plt.close(fig)


# =========================================================================
# Figure 9: Hybrid comparison (experiment 08)
# =========================================================================

def fig_hybrid_comparison(fmt: str = "pdf"):
    """Grouped bar chart: distance to target for 4 methods (+ error bars)."""
    plt = _setup_style()
    data = load_results("08_hybrid_steering.json")
    aggregate = data["aggregate"]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(3)
    width = 0.2
    methods = [
        ("unsteered", "Unsteered", "#95a5a6"),
        ("repe_only", "RepE Only", "#2ecc71"),
        ("pe_only", "PE Only", "#f39c12"),
        ("hybrid", "Hybrid", "#9b59b6"),
    ]

    for j, (key, label, color) in enumerate(methods):
        dists = [aggregate[key]["per_dimension"][dim]["mean_distance"]
                 for dim in DIMENSION_NAMES]
        # Error bars from CIs if available
        ci_lo = [aggregate[key]["per_dimension"][dim].get("ci_lower", dists[i])
                 for i, dim in enumerate(DIMENSION_NAMES)]
        ci_hi = [aggregate[key]["per_dimension"][dim].get("ci_upper", dists[i])
                 for i, dim in enumerate(DIMENSION_NAMES)]
        yerr_lo = [d - lo for d, lo in zip(dists, ci_lo)]
        yerr_hi = [hi - d for d, hi in zip(dists, ci_hi)]

        bars = ax.bar(x + j * width, dists, width, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      yerr=[yerr_lo, yerr_hi], capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMENSION_NAMES])
    ax.set_ylabel("Mean Distance to Target")
    ax.set_title("Hybrid Steering: Distance to ACT-Optimal EPA")
    ax.legend(loc="upper left")
    plt.tight_layout()
    _save_fig(fig, "fig09_hybrid_comparison", fmt)
    plt.close(fig)


# =========================================================================
# Figure 10: Direction quality comparison (experiment 09)
# =========================================================================

def fig_direction_comparison(fmt: str = "pdf"):
    """Bar chart comparing cross-dimension entanglement across direction sets."""
    plt = _setup_style()
    data = load_results("09_direction_quality_comparison.json")
    dir_sets = data.get("direction_sets", {})

    if not dir_sets:
        print("  No direction sets to compare")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    set_names = list(dir_sets.keys())
    n_sets = len(set_names)
    pairs = ["evaluation_vs_potency", "evaluation_vs_activity", "potency_vs_activity"]
    pair_labels = ["E·P", "E·A", "P·A"]

    x = np.arange(len(pairs))
    width = 0.8 / n_sets
    colors = ["#95a5a6", "#2ecc71", "#9b59b6", "#e74c3c"]

    for j, set_name in enumerate(set_names):
        vals = [dir_sets[set_name]["cross_dimension_cosines"][p]["mean_abs_cosine"]
                for p in pairs]
        ax.bar(x + j * width, vals, width, label=set_name.replace("_", " ").title(),
               color=colors[j % len(colors)], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (n_sets - 1) / 2)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel("Mean |Cosine Similarity|")
    ax.set_title("Cross-Dimension Entanglement by Direction Set")
    ax.legend()
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="Target threshold")
    plt.tight_layout()
    _save_fig(fig, "fig10_direction_comparison", fmt)
    plt.close(fig)


# =========================================================================
# Figure 11: Statistical significance forest plot (experiment 10)
# =========================================================================

def fig_significance_forest(fmt: str = "pdf"):
    """Forest plot showing effect sizes with CIs for all comparisons."""
    plt = _setup_style()
    data = load_results("10_statistical_significance.json")
    comparisons = data.get("comparisons", {})

    if not comparisons:
        print("  No comparisons to plot")
        return

    # Collect all data points
    labels = []
    effects = []
    ci_los = []
    ci_his = []

    for comp_name, comp_data in comparisons.items():
        for dim in DIMENSION_NAMES:
            d = comp_data.get(dim, {})
            if "cohens_d" not in d:
                continue
            clean_name = comp_name.replace("_", " ").title()
            labels.append(f"{clean_name}\n{dim[0].upper()}")
            effects.append(d["cohens_d"])
            # Approximate CI from the mean_diff CI
            ci_los.append(d.get("ci_lower", d["cohens_d"] - 0.1))
            ci_his.append(d.get("ci_upper", d["cohens_d"] + 0.1))

    if not labels:
        print("  No effect sizes to plot")
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.4)))
    y = np.arange(n)

    # Color by significance
    colors = []
    for comp_name, comp_data in comparisons.items():
        for dim in DIMENSION_NAMES:
            d = comp_data.get(dim, {})
            if "cohens_d" not in d:
                continue
            p = min(d.get("permutation_p", 1), d.get("wilcoxon_p", 1))
            if p < 0.001:
                colors.append("#2ecc71")
            elif p < 0.05:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

    ax.barh(y, effects, color=colors, alpha=0.7, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.2, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(-0.2, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (paired)")
    ax.set_title("Effect Sizes Across Comparisons")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.7, label="p < .001"),
        Patch(facecolor="#f39c12", alpha=0.7, label="p < .05"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="p ≥ .05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    _save_fig(fig, "fig11_significance_forest", fmt)
    plt.close(fig)


# =========================================================================
# Figure 12: Orthogonal direction cosine heatmap (experiment 09)
# =========================================================================

def fig_orthogonal_cosines(fmt: str = "pdf"):
    """Heatmap of per-layer cosine similarities between direction sets."""
    plt = _setup_style()
    import seaborn as sns

    data = load_results("09_direction_quality_comparison.json")
    cosines = data.get("cosine_similarities", {})

    if not cosines:
        print("  No cosine similarity data to plot")
        return

    # Take the first comparison pair
    pair_key = list(cosines.keys())[0]
    pair_data = cosines[pair_key]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    for i, dim in enumerate(DIMENSION_NAMES):
        ax = axes[i]
        per_layer = pair_data[dim]["per_layer"]
        layers = sorted([int(k) for k in per_layer.keys()])
        vals = [per_layer[str(l)] for l in layers]

        ax.bar(range(len(layers)), vals, color=DIM_COLORS[dim], alpha=0.7)
        ax.set_xticks(range(0, len(layers), 5))
        ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 5)],
                           fontsize=7)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{DIM_LABELS[dim]} ({pair_key.replace('_', ' ')})")
        ax.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle("Per-Layer Direction Cosine Similarity", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig12_orthogonal_cosines", fmt)
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

FIGURES = {
    "reading_scatter": fig_reading_scatter,
    "layer_correlations": fig_layer_correlations,
    "steering_scatter": fig_steering_scatter,
    "steering_bars": fig_steering_bars,
    "coefficient_sweep": fig_coefficient_sweep,
    "identity_heatmap": fig_identity_heatmap,
    "coherence": fig_coherence,
    "baseline_comparison": fig_baseline_comparison,
    "hybrid_comparison": fig_hybrid_comparison,
    "direction_comparison": fig_direction_comparison,
    "significance_forest": fig_significance_forest,
    "orthogonal_cosines": fig_orthogonal_cosines,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from experiment results")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"],
                        help="Primary output format")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES.keys()),
                        help="Generate only specific figures")
    args = parser.parse_args()

    targets = args.only if args.only else list(FIGURES.keys())

    for name in targets:
        result_file = {
            "reading_scatter": "01_reading_quality.json",
            "layer_correlations": None,  # uses tuning results directly
            "steering_scatter": "02_closed_loop_steering.json",
            "steering_bars": "02_closed_loop_steering.json",
            "coefficient_sweep": "04_coefficient_sweep.json",
            "identity_heatmap": "06_identity_generalization.json",
            "coherence": "07_coherence_evaluation.json",
            "baseline_comparison": "03_prompt_baseline.json",
            "hybrid_comparison": "08_hybrid_steering.json",
            "direction_comparison": "09_direction_quality_comparison.json",
            "significance_forest": "10_statistical_significance.json",
            "orthogonal_cosines": "09_direction_quality_comparison.json",
        }

        req_file = result_file.get(name)
        if req_file and not (RESULTS_DIR / req_file).exists():
            print(f"SKIP {name}: {req_file} not found (run the experiment first)")
            continue

        print(f"Generating {name}...")
        try:
            FIGURES[name](fmt=args.format)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

