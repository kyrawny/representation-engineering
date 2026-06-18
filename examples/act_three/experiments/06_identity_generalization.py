"""
Experiment 06 — Identity Pair Generalisation.

Shows that ACT deflection minimisation produces *different*, identity-
appropriate optimal EPAs for the same user message across different
social identity pairs.

Usage::

    python -m examples.act_three.experiments.06_identity_generalization
    python -m examples.act_three.experiments.06_identity_generalization --quick
"""

import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .config import (
    DIMENSION_NAMES,
    IDENTITY_PAIRS,
    GENERATION_DEFAULTS,
)
from .setup import (
    load_experiment_components,
    get_identity_epa,
    make_system_prompt,
    save_results,
    clean_response,
)
from .scenarios import SCENARIOS


# Hand-picked scenarios that work well across all identity contexts
GENERALISATION_SCENARIOS = [s for s in SCENARIOS if s["id"] in [
    "pos_e_01",   # grateful
    "neg_e_01",   # angry
    "neg_e_05",   # defiant
    "high_p_01",  # demanding
    "low_p_02",   # helpless
    "high_a_06",  # panicking
    "low_a_02",   # calm
    "mixed_07",   # diplomatic disagreement
    "mixed_13",   # skeptical
    "low_p_14",   # desperate
]]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 06: Identity Pair Generalisation")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="06_identity_generalization.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
        format_llama3_prompt,
        impression_formation,
        calculate_deflection,
    )

    scenarios = GENERALISATION_SCENARIOS
    if args.quick:
        scenarios = scenarios[:3]

    pairs = IDENTITY_PAIRS
    total = len(scenarios) * len(pairs)
    print(f"Running {len(scenarios)} scenarios × {len(pairs)} pairs = {total}")

    trials = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        scenario_trials = []

        # Read user EPA once (same message for all pairs)
        user_msg_epa = comp.reader.read_epa(
            comp.rep_reading_pipeline, scenario["text"])

        for pair_name, agent_term, user_term in pairs:
            agent_epa = get_identity_epa(comp.identities_df, agent_term)
            user_epa = get_identity_epa(comp.identities_df, user_term)

            user_behavior = EPA(
                e=user_msg_epa["evaluation"],
                p=user_msg_epa["potency"],
                a=user_msg_epa["activity"],
            )

            # Compute ACT optimal
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

            # Compute deflection before response
            post_user = impression_formation(
                actor=user_epa, behavior=user_behavior,
                obj=agent_epa, coefficients=comp.coefficients,
            )
            pre_deflection = (
                calculate_deflection(user_epa, post_user["actor"])
                + calculate_deflection(agent_epa, post_user["object"])
            )

            # Generate steered response
            sys_prompt = make_system_prompt(agent_term, user_term)
            prompt = format_llama3_prompt(sys_prompt, scenario["text"])
            steered_text = comp.steerer.generate(
                prompt=prompt,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            steered_text = clean_response(steered_text)

            steered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, steered_text)

            scenario_trials.append({
                "identity_pair": pair_name,
                "agent_term": agent_term,
                "user_term": user_term,
                "agent_identity_epa": agent_epa.to_dict(),
                "user_identity_epa": user_epa.to_dict(),
                "target_epa": target_dict,
                "pre_response_deflection": float(pre_deflection),
                "steered_text": steered_text,
                "steered_epa": steered_epa,
            })

        trials.append({
            "scenario_id": scenario["id"],
            "scenario_text": scenario["text"],
            "scenario_category": scenario["category"],
            "user_msg_epa": user_msg_epa,
            "identity_pair_results": scenario_trials,
        })

    # ---- Compute target EPA variance across pairs ----
    target_variance = {}
    for dim in DIMENSION_NAMES:
        all_targets = []
        for trial in trials:
            targets = [r["target_epa"][dim]
                       for r in trial["identity_pair_results"]]
            all_targets.append(targets)
        # Mean variance across scenarios
        variances = [float(np.var(row)) for row in all_targets]
        target_variance[dim] = {
            "mean_variance": float(np.mean(variances)),
            "max_variance": float(np.max(variances)),
            "min_variance": float(np.min(variances)),
        }

    results = {
        "metadata": {
            "experiment": "06_identity_generalization",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "identity_pairs": [p[0] for p in pairs],
            "quick_mode": args.quick,
        },
        "target_epa_variance_across_pairs": target_variance,
        "trials": trials,
    }
    save_results(results, args.output)


if __name__ == "__main__":
    main()
