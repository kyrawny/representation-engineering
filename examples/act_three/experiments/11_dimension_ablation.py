"""
Experiment 11 — EPA Dimension Ablation.

For each trial, steers only a subset of {E, P, A} dimensions and measures
the resulting EPA distance to the ACT target.  Compares all 7 non-trivial
dimension combinations (E, P, A, EP, EA, PA, EPA) against the four
conditions from Experiment 08 (unsteered, pe_only, repe_only, hybrid).

This reveals:
  - Which individual dimensions benefit most from RepE steering.
  - Whether steering a subset of dimensions (e.g. P+A only) can match
    or exceed full EPA steering in overall quality.
  - Whether steering some dimensions hurts others (cross-dimension
    interference).

Usage::

    python -m examples.act_three.experiments.11_dimension_ablation
    python -m examples.act_three.experiments.11_dimension_ablation --quick
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import (
    DIMENSION_NAMES,
    IDENTITY_PAIRS,
    GENERATION_DEFAULTS,
    QUICK_N_SCENARIOS,
    QUICK_N_IDENTITY_PAIRS,
)
from .setup import (
    load_experiment_components,
    get_identity_epa,
    make_system_prompt,
    save_results,
    generate_unsteered,
    clean_response,
    ensure_results_dir,
)
from .scenarios import get_scenarios
from .stats_utils import (
    bootstrap_ci,
    paired_permutation_test,
    cohens_d_paired,
    wilcoxon_signed_rank,
)


# =========================================================================
# Dimension subset definitions
# =========================================================================

def _get_dimension_subsets():
    """Return all non-empty subsets of {E, P, A} as sorted tuples.

    Yields 7 subsets: 3 singletons, 3 pairs, 1 triple.
    """
    dims = DIMENSION_NAMES  # ["evaluation", "potency", "activity"]
    subsets = []
    for r in range(1, len(dims) + 1):
        for combo in itertools.combinations(dims, r):
            subsets.append(combo)
    return subsets


def _subset_label(subset):
    """Short label for a dimension subset, e.g. 'E', 'PA', 'EPA'."""
    abbrevs = {"evaluation": "E", "potency": "P", "activity": "A"}
    return "".join(abbrevs[d] for d in subset)


# =========================================================================
# Loading Experiment 08 reference data
# =========================================================================

def _load_exp08_reference(exp08_path=None):
    """Load Experiment 08 results for comparison.

    Args:
        exp08_path: Explicit path to the Exp 08 JSON file.  If None,
            looks in the default results directory.

    Returns a dict mapping (scenario_id, pair) -> {condition: {dim: distance}}.
    Returns None if the file is not found.
    """
    if exp08_path is not None:
        path = Path(exp08_path)
    else:
        results_dir = ensure_results_dir()
        path = results_dir / "08_hybrid_steering.json"

    if not path.exists():
        print(f"WARNING: {path} not found. Comparisons to Exp 08 will be skipped.")
        return None

    print(f"Loading Exp 08 reference from: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    lookup = {}
    for trial in data.get("trials", []):
        key = (trial["scenario_id"], trial["pair"])
        lookup[key] = {}
        for cond_name, cond_data in trial["conditions"].items():
            lookup[key][cond_name] = {
                dim: cond_data["distances"][dim]
                for dim in DIMENSION_NAMES
            }
    return lookup


# =========================================================================
# Main experiment
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 11: EPA Dimension Ablation")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="11_dimension_ablation.json",
                        help="Output filename in results/")
    parser.add_argument("--exp08-path", default=None,
                        help="Path to Exp 08 results JSON for comparison "
                             "(default: results/08_hybrid_steering.json)")
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
        format_llama3_prompt,
    )

    scenarios = get_scenarios(quick=args.quick, n=QUICK_N_SCENARIOS)
    pairs = IDENTITY_PAIRS
    if args.quick:
        pairs = pairs[:QUICK_N_IDENTITY_PAIRS]

    dim_subsets = _get_dimension_subsets()
    subset_labels = [_subset_label(s) for s in dim_subsets]

    total_trials = len(scenarios) * len(pairs)
    total_generations = total_trials * len(dim_subsets)
    print(f"Running {len(scenarios)} scenarios × {len(pairs)} pairs "
          f"= {total_trials} trials")
    print(f"  × {len(dim_subsets)} dimension subsets "
          f"= {total_generations} steered generations")
    print(f"  Subsets: {', '.join(subset_labels)}")

    # ---- Load Exp 08 reference ----
    exp08_ref = _load_exp08_reference(args.exp08_path)

    # ---- Run trials ----
    trials = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for pair_name, agent_term, user_term in pairs:
            agent_epa = get_identity_epa(comp.identities_df, agent_term)
            user_epa = get_identity_epa(comp.identities_df, user_term)

            # Read user message EPA
            user_msg_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, scenario["text"])

            user_behavior = EPA(
                e=user_msg_epa["evaluation"],
                p=user_msg_epa["potency"],
                a=user_msg_epa["activity"],
            )

            # Compute ACT target
            target_epa = get_response_epa_for_deflection_minimization(
                agent_identity=agent_epa,
                user_identity=user_epa,
                user_behavior_epa=user_behavior,
                coefficients=comp.coefficients,
            )
            target_dict = {
                "evaluation": target_epa.e,
                "potency": target_epa.p,
                "activity": target_epa.a,
            }

            # Build the prompt (neutral system prompt, same as Exp 08 unsteered/repe_only)
            sys_prompt = make_system_prompt(agent_term, user_term)
            prompt = format_llama3_prompt(sys_prompt, scenario["text"])

            # ---- Generate for each dimension subset ----
            subset_results = {}

            for subset in dim_subsets:
                label = _subset_label(subset)

                # Build a partial target_epa with only the selected dims
                partial_target = {}
                for dim in DIMENSION_NAMES:
                    if dim in subset:
                        partial_target[dim] = target_dict[dim]
                    else:
                        # Zero = no steering for this dimension
                        partial_target[dim] = 0.0

                # Generate with partial steering
                steered_text = comp.steerer.generate(
                    prompt=prompt,
                    target_epa=partial_target,
                    **GENERATION_DEFAULTS,
                )
                steered_text = clean_response(steered_text)

                # Read back full EPA
                read_epa = comp.reader.read_epa(
                    comp.rep_reading_pipeline, steered_text)

                # Compute distances to the *full* target (all 3 dims)
                distances = {
                    dim: abs(read_epa[dim] - target_dict[dim])
                    for dim in DIMENSION_NAMES
                }
                mean_dist = float(np.mean(list(distances.values())))

                subset_results[label] = {
                    "steered_dims": list(subset),
                    "text": steered_text[:500],
                    "epa": read_epa,
                    "distances": distances,
                    "mean_distance": mean_dist,
                }

            # ---- Pull Exp 08 reference distances if available ----
            exp08_distances = None
            ref_key = (scenario["id"], pair_name)
            if exp08_ref and ref_key in exp08_ref:
                exp08_distances = exp08_ref[ref_key]

            trial = {
                "scenario_id": scenario["id"],
                "pair": pair_name,
                "target_epa": target_dict,
                "ablation_conditions": subset_results,
            }
            if exp08_distances is not None:
                trial["exp08_reference"] = exp08_distances

            trials.append(trial)

    # =================================================================
    # Aggregate statistics
    # =================================================================

    # --- Per-subset aggregates ---
    ablation_aggregate = {}
    for subset in dim_subsets:
        label = _subset_label(subset)
        per_dim_dists = {dim: [] for dim in DIMENSION_NAMES}
        mean_dists = []

        for trial in trials:
            cond = trial["ablation_conditions"][label]
            for dim in DIMENSION_NAMES:
                per_dim_dists[dim].append(cond["distances"][dim])
            mean_dists.append(cond["mean_distance"])

        dim_stats = {}
        for dim in DIMENSION_NAMES:
            arr = np.array(per_dim_dists[dim])
            point, ci_lo, ci_hi = bootstrap_ci(
                arr, lambda x: float(np.mean(x)))
            dim_stats[dim] = {
                "mean_distance": point,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "std": float(np.std(arr)),
            }

        ablation_aggregate[label] = {
            "steered_dims": list(subset),
            "per_dimension": dim_stats,
            "overall_mean_distance": float(np.mean(mean_dists)),
            "overall_std_distance": float(np.std(mean_dists)),
        }

    # --- Comparison to Exp 08 conditions ---
    exp08_conditions = ["unsteered", "repe_only", "pe_only", "hybrid"]
    comparisons = {}

    # Check if we have exp08 data in all trials
    has_exp08 = all("exp08_reference" in t for t in trials)

    if has_exp08:
        for subset in dim_subsets:
            label = _subset_label(subset)
            subset_comparisons = {}

            for baseline_cond in exp08_conditions:
                cond_results = {}

                for dim in DIMENSION_NAMES:
                    ablation_dists = np.array([
                        t["ablation_conditions"][label]["distances"][dim]
                        for t in trials
                    ])
                    baseline_dists = np.array([
                        t["exp08_reference"][baseline_cond][dim]
                        for t in trials
                    ])

                    # Positive delta = ablation is closer (improvement over baseline)
                    diff_mean, perm_p = paired_permutation_test(
                        baseline_dists, ablation_dists,
                        alternative="greater")
                    _, wilcox_p = wilcoxon_signed_rank(
                        baseline_dists, ablation_dists,
                        alternative="greater")
                    d = cohens_d_paired(baseline_dists, ablation_dists)

                    cond_results[dim] = {
                        "mean_improvement": float(diff_mean),
                        "permutation_p": float(perm_p),
                        "wilcoxon_p": float(wilcox_p),
                        "cohens_d": float(d),
                    }

                # Also compute overall mean distance comparison
                ablation_means = np.array([
                    t["ablation_conditions"][label]["mean_distance"]
                    for t in trials
                ])
                baseline_means = np.array([
                    np.mean([
                        t["exp08_reference"][baseline_cond][dim]
                        for dim in DIMENSION_NAMES
                    ])
                    for t in trials
                ])
                overall_diff, overall_perm_p = paired_permutation_test(
                    baseline_means, ablation_means,
                    alternative="greater")
                _, overall_wilcox_p = wilcoxon_signed_rank(
                    baseline_means, ablation_means,
                    alternative="greater")
                overall_d = cohens_d_paired(baseline_means, ablation_means)

                cond_results["overall"] = {
                    "mean_improvement": float(overall_diff),
                    "permutation_p": float(overall_perm_p),
                    "wilcoxon_p": float(overall_wilcox_p),
                    "cohens_d": float(overall_d),
                }

                subset_comparisons[f"vs_{baseline_cond}"] = cond_results

            comparisons[label] = subset_comparisons
    else:
        print("WARNING: Exp 08 reference data not found for all trials. "
              "Skipping cross-experiment comparisons.")

    # --- Cross-dimension interference analysis ---
    # For each single-dimension ablation, check if the steered dimension
    # improved while other dimensions got worse compared to unsteered
    interference = {}
    if has_exp08:
        for dim in DIMENSION_NAMES:
            label = _subset_label((dim,))
            steered_dists = np.array([
                t["ablation_conditions"][label]["distances"][dim]
                for t in trials
            ])
            unsteered_dists = np.array([
                t["exp08_reference"]["unsteered"][dim]
                for t in trials
            ])

            # How did the steered dimension itself change?
            steered_diff = float(np.mean(unsteered_dists - steered_dists))

            # How did the *other* dimensions change?
            other_dims = [d for d in DIMENSION_NAMES if d != dim]
            other_effects = {}
            for other_dim in other_dims:
                other_steered = np.array([
                    t["ablation_conditions"][label]["distances"][other_dim]
                    for t in trials
                ])
                other_unsteered = np.array([
                    t["exp08_reference"]["unsteered"][other_dim]
                    for t in trials
                ])
                other_diff = float(np.mean(other_unsteered - other_steered))
                _, other_p = paired_permutation_test(
                    other_unsteered, other_steered,
                    alternative="greater")
                other_effects[other_dim] = {
                    "mean_change": other_diff,
                    "permutation_p": float(other_p),
                    "interpretation": (
                        "improved" if other_diff > 0.05 else
                        "degraded" if other_diff < -0.05 else
                        "negligible"
                    ),
                }

            interference[dim] = {
                "steered_dim_improvement": steered_diff,
                "other_dim_effects": other_effects,
            }

    # =================================================================
    # Build results
    # =================================================================

    results = {
        "metadata": {
            "experiment": "11_dimension_ablation",
            "timestamp": datetime.now().isoformat(),
            "n_trials": len(trials),
            "n_scenarios": len(scenarios),
            "n_pairs": len(pairs),
            "n_subsets": len(dim_subsets),
            "subsets": subset_labels,
            "quick_mode": args.quick,
            "exp08_reference_loaded": has_exp08,
        },
        "ablation_aggregate": ablation_aggregate,
        "comparisons_vs_exp08": comparisons if comparisons else None,
        "interference_analysis": interference if interference else None,
        "trials": trials,
    }

    save_results(results, args.output)

    # =================================================================
    # Print summary
    # =================================================================

    print(f"\n{'='*70}")
    print(f"  DIMENSION ABLATION SUMMARY ({len(trials)} trials)")
    print(f"{'='*70}")

    print(f"\n{'Subset':<8} {'E dist':>8} {'P dist':>8} {'A dist':>8} {'Overall':>8}")
    print("-" * 46)
    for subset in dim_subsets:
        label = _subset_label(subset)
        agg = ablation_aggregate[label]
        e = agg["per_dimension"]["evaluation"]["mean_distance"]
        p = agg["per_dimension"]["potency"]["mean_distance"]
        a = agg["per_dimension"]["activity"]["mean_distance"]
        o = agg["overall_mean_distance"]
        # Mark steered dims with *
        e_mark = "*" if "evaluation" in subset else " "
        p_mark = "*" if "potency" in subset else " "
        a_mark = "*" if "activity" in subset else " "
        print(f"{label:<8} {e:>7.3f}{e_mark} "
              f"{p:>7.3f}{p_mark} "
              f"{a:>7.3f}{a_mark} {o:>8.3f}")

    if has_exp08:
        print(f"\nExp 08 reference distances:")
        print(f"{'Condition':<12} {'E dist':>8} {'P dist':>8} {'A dist':>8} {'Overall':>8}")
        print("-" * 50)
        # Compute from trial data
        for cond in exp08_conditions:
            e_dists = [t["exp08_reference"][cond]["evaluation"] for t in trials]
            p_dists = [t["exp08_reference"][cond]["potency"] for t in trials]
            a_dists = [t["exp08_reference"][cond]["activity"] for t in trials]
            e = np.mean(e_dists)
            p = np.mean(p_dists)
            a = np.mean(a_dists)
            o = np.mean([np.mean([e_d, p_d, a_d])
                         for e_d, p_d, a_d in zip(e_dists, p_dists, a_dists)])
            print(f"{cond:<12} {e:>8.3f} {p:>8.3f} {a:>8.3f} {o:>8.3f}")

    if interference:
        print(f"\n{'='*70}")
        print(f"  CROSS-DIMENSION INTERFERENCE (single-dim steering)")
        print(f"{'='*70}")
        for dim in DIMENSION_NAMES:
            label = _subset_label((dim,))
            info = interference[dim]
            print(f"\n  Steering {dim.upper()} only:")
            print(f"    {dim}: improvement = {info['steered_dim_improvement']:+.3f}")
            for other_dim, effect in info["other_dim_effects"].items():
                print(f"    {other_dim}: change = {effect['mean_change']:+.3f} "
                      f"(p={effect['permutation_p']:.3f}, "
                      f"{effect['interpretation']})")


if __name__ == "__main__":
    main()
