"""
Experiment 02 — Closed-Loop Steering Evaluation.

The core paper experiment.  For each scenario × identity pair:
    1. Read user message EPA
    2. Compute ACT optimal response EPA
    3. Generate unsteered and steered responses
    4. Read EPA of both responses
    5. Measure whether steering moved the response closer to the target

Usage::

    python -m examples.act_three.experiments.02_closed_loop_steering
    python -m examples.act_three.experiments.02_closed_loop_steering --quick
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import (
    DIMENSION_NAMES,
    IDENTITY_PAIRS,
    GENERATION_DEFAULTS,
    CHECKPOINT_EVERY,
    QUICK_N_SCENARIOS,
    QUICK_N_IDENTITY_PAIRS,
)
from .setup import (
    load_experiment_components,
    get_identity_epa,
    make_system_prompt,
    save_results,
    ensure_results_dir,
    generate_unsteered,
    clean_response,
)
from .scenarios import get_scenarios
from .stats_utils import (
    bootstrap_ci,
    paired_permutation_test,
    cohens_d_paired,
    wilcoxon_signed_rank,
)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 02: Closed-Loop Steering Evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="02_closed_loop_steering.json",
                        help="Output filename in results/")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--per-dim-coeff", action="store_true",
                        help="Use per-dimension optimised steering coefficients")
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
        format_llama3_prompt,
    )

    # ---- Setup scenarios and identity pairs ----
    scenarios = get_scenarios(quick=args.quick, n=QUICK_N_SCENARIOS)
    pairs = IDENTITY_PAIRS
    if args.quick:
        pairs = pairs[:QUICK_N_IDENTITY_PAIRS]

    total_trials = len(scenarios) * len(pairs)
    print(f"Running {len(scenarios)} scenarios × {len(pairs)} identity pairs "
          f"= {total_trials} trials")

    # ---- Resume from checkpoint if available ----
    completed = {}
    if args.resume and Path(args.resume).exists():
        with open(args.resume, "r") as f:
            checkpoint = json.load(f)
        for trial in checkpoint.get("trials", []):
            completed[trial["trial_key"]] = trial
        print(f"Resumed from checkpoint: {len(completed)} trials already done")

    # ---- Run trials ----
    trials = []
    trial_count = 0

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for pair_name, agent_term, user_term in pairs:
            trial_key = f"{scenario['id']}__{pair_name}"
            if trial_key in completed:
                trials.append(completed[trial_key])
                continue

            agent_epa = get_identity_epa(comp.identities_df, agent_term)
            user_epa = get_identity_epa(comp.identities_df, user_term)

            # 1. Read user message EPA
            user_msg_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, scenario["text"])

            # 2. Compute ACT optimal response EPA
            user_behavior = EPA(
                e=user_msg_epa["evaluation"],
                p=user_msg_epa["potency"],
                a=user_msg_epa["activity"],
            )
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

            # 3. Generate unsteered response
            sys_prompt = make_system_prompt(agent_term, user_term)
            prompt = format_llama3_prompt(sys_prompt, scenario["text"])
            unsteered_text = generate_unsteered(
                comp.model, comp.tokenizer, prompt)

            # 4. Generate steered response
            steered_text = comp.steerer.generate(
                prompt=prompt,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            steered_text = clean_response(steered_text)

            # 5. Read EPA of both responses
            unsteered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, unsteered_text)
            steered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, steered_text)

            # Record trial
            trial = {
                "trial_key": trial_key,
                "scenario_id": scenario["id"],
                "scenario_category": scenario["category"],
                "scenario_text": scenario["text"],
                "identity_pair": pair_name,
                "agent_term": agent_term,
                "user_term": user_term,
                "agent_identity_epa": agent_epa.to_dict(),
                "user_identity_epa": user_epa.to_dict(),
                "user_msg_epa": user_msg_epa,
                "target_epa": target_dict,
                "unsteered_text": unsteered_text,
                "steered_text": steered_text,
                "unsteered_epa": unsteered_epa,
                "steered_epa": steered_epa,
            }
            trials.append(trial)
            trial_count += 1

            # Checkpoint
            if trial_count % CHECKPOINT_EVERY == 0:
                _save_checkpoint(trials, args.output)
                print(f"  [checkpoint at {trial_count}/{total_trials}]")

    # ---- Compute aggregate metrics ----
    metrics = _compute_aggregate_metrics(trials)

    # ---- Save final results ----
    results = {
        "metadata": {
            "experiment": "02_closed_loop_steering",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "n_identity_pairs": len(pairs),
            "n_trials": len(trials),
            "quick_mode": args.quick,
        },
        "aggregate_metrics": metrics,
        "trials": trials,
    }
    save_results(results, args.output)


def _compute_aggregate_metrics(trials: list) -> dict:
    """Compute per-dimension and overall steering metrics with CIs."""
    metrics = {"per_dimension": {}, "overall": {}}

    for dim in DIMENSION_NAMES:
        distances_unsteered = []
        distances_steered = []
        improvements = []
        shifts = []

        for t in trials:
            target = t["target_epa"][dim]
            un_val = t["unsteered_epa"][dim]
            st_val = t["steered_epa"][dim]

            d_un = abs(target - un_val)
            d_st = abs(target - st_val)
            distances_unsteered.append(d_un)
            distances_steered.append(d_st)
            improvements.append(1 if d_st < d_un else 0)
            shifts.append(st_val - un_val)

        # Bootstrap CI on hit rate
        hit_point, hit_ci_lo, hit_ci_hi = bootstrap_ci(
            improvements, lambda x: float(np.mean(x)))

        # Bootstrap CI on mean distance improvement
        un_arr = np.array(distances_unsteered)
        st_arr = np.array(distances_steered)
        improv_point, improv_ci_lo, improv_ci_hi = bootstrap_ci(
            un_arr - st_arr, lambda x: float(np.mean(x)))

        # Wilcoxon signed-rank test
        try:
            _, wilcox_p = wilcoxon_signed_rank(un_arr, st_arr)
        except ValueError:
            wilcox_p = 1.0

        # Permutation test
        _, perm_p = paired_permutation_test(un_arr, st_arr)

        # Effect size
        d = cohens_d_paired(un_arr, st_arr)

        metrics["per_dimension"][dim] = {
            "hit_rate": hit_point,
            "hit_rate_ci": [hit_ci_lo, hit_ci_hi],
            "mean_distance_unsteered": float(np.mean(distances_unsteered)),
            "mean_distance_steered": float(np.mean(distances_steered)),
            "mean_distance_improvement": improv_point,
            "improvement_ci": [improv_ci_lo, improv_ci_hi],
            "wilcoxon_p": float(wilcox_p),
            "permutation_p": float(perm_p),
            "cohens_d": float(d),
            "mean_shift": float(np.mean(shifts)),
            "std_shift": float(np.std(shifts)),
        }

    # Overall hit rate (all three dimensions improved)
    all_improved = []
    for t in trials:
        improved_all = all(
            abs(t["target_epa"][dim] - t["steered_epa"][dim])
            < abs(t["target_epa"][dim] - t["unsteered_epa"][dim])
            for dim in DIMENSION_NAMES
        )
        all_improved.append(1 if improved_all else 0)
    metrics["overall"]["all_dims_hit_rate"] = float(np.mean(all_improved))

    # Any dimension improved
    any_improved = []
    for t in trials:
        improved_any = any(
            abs(t["target_epa"][dim] - t["steered_epa"][dim])
            < abs(t["target_epa"][dim] - t["unsteered_epa"][dim])
            for dim in DIMENSION_NAMES
        )
        any_improved.append(1 if improved_any else 0)
    metrics["overall"]["any_dim_hit_rate"] = float(np.mean(any_improved))

    return metrics


def _save_checkpoint(trials: list, output_name: str):
    """Save a checkpoint of completed trials."""
    d = ensure_results_dir()
    path = d / f"_checkpoint_{output_name}"
    with open(path, "w") as f:
        json.dump({"trials": trials}, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else None)


if __name__ == "__main__":
    main()
