"""
Quick Coefficient Search.

Generates responses at different steering coefficient strengths for a
handful of scenarios and measures EPA shift. Helps find a good base
coefficient without running the full 500-trial experiment.

Usage::

    python -m examples.act_three.experiments.quick_coeff_search
    python -m examples.act_three.experiments.quick_coeff_search --coeffs 1 3 5 8 10 15
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Quick coefficient search for EPA steering")
    parser.add_argument(
        "--coeffs", nargs="*", type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        help="Coefficient values to test (perturbation = coeff * target_value)")
    parser.add_argument(
        "--n-scenarios", type=int, default=5,
        help="Number of test scenarios")
    args = parser.parse_args()

    # Setup
    exp_dir = Path(__file__).resolve().parent
    act_dir = exp_dir.parent
    repo_root = act_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from .setup import (
        load_experiment_components, make_system_prompt,
        generate_unsteered, clean_response,
    )
    from .config import GENERATION_DEFAULTS, DIMENSION_NAMES
    from .scenarios import get_scenarios
    from examples.act_three.model_registry import format_chat_prompt
    from examples.act_three.epa_steerer import EPASteerer

    # Load components (steerer will be overridden per-coefficient)
    comp = load_experiment_components(load_steerer=True)

    # Get a few scenarios
    scenarios = get_scenarios()[:args.n_scenarios]
    pair_name, agent_term, user_term = "counselor_client", "counselor", "client"

    from .setup import get_identity_epa
    agent_epa = get_identity_epa(comp.identities_df, agent_term)
    user_epa = get_identity_epa(comp.identities_df, user_term)
    from examples.act_three import get_response_epa_for_deflection_minimization

    print(f"\nTesting {len(args.coeffs)} coefficient values on {len(scenarios)} scenarios")
    print(f"Current steerer coefficients from tuning file:")
    for dim in DIMENSION_NAMES:
        cfg = comp.steerer.steering_configs[dim]
        print(f"  {dim}: base_coeff={cfg['base_coeff']}, layers={cfg['layers']}")

    # Results storage
    results = {c: {"texts": [], "epa_shifts": [], "distances": []} for c in args.coeffs}

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}/{len(scenarios)}: {scenario['id']} ---")
        print(f"  Text: {scenario['text'][:80]}...")

        # Read user message EPA
        sys_prompt = make_system_prompt(agent_term, user_term)
        prompt = format_chat_prompt(comp.tokenizer, sys_prompt, scenario["text"])

        user_msg_epa = comp.reader.read_epa(
            comp.rep_reading_pipeline, scenario["text"])

        from examples.act_three import EPA
        user_msg = EPA(
            e=user_msg_epa["evaluation"],
            p=user_msg_epa["potency"],
            a=user_msg_epa["activity"],
        )

        target_epa = get_response_epa_for_deflection_minimization(
            agent_epa, user_epa, user_msg, comp.coefficients)

        target_dict = {
            "evaluation": target_epa.e,
            "potency": target_epa.p,
            "activity": target_epa.a,
        }
        print(f"  Target EPA: E={target_epa.e:.2f}, P={target_epa.p:.2f}, A={target_epa.a:.2f}")

        # Generate unsteered baseline
        unsteered_text = generate_unsteered(comp.model, comp.tokenizer, prompt)
        unsteered_epa = comp.reader.read_epa(comp.rep_reading_pipeline, unsteered_text)
        print(f"  Unsteered: E={unsteered_epa['evaluation']:.2f}, "
              f"P={unsteered_epa['potency']:.2f}, A={unsteered_epa['activity']:.2f}")

        # Test each coefficient
        for coeff in args.coeffs:
            # Override coefficient for all dimensions
            for dim in DIMENSION_NAMES:
                comp.steerer.steering_configs[dim]["base_coeff"] = coeff

            steered_text = comp.steerer.generate(
                prompt=prompt,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            steered_text = clean_response(steered_text)

            steered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, steered_text)

            # Compute distance to target
            dist = sum(
                abs(steered_epa[dim] - target_dict[dim])
                for dim in DIMENSION_NAMES
            )
            unstn_dist = sum(
                abs(unsteered_epa[dim] - target_dict[dim])
                for dim in DIMENSION_NAMES
            )

            improvement = unstn_dist - dist

            results[coeff]["texts"].append(steered_text[:100])
            results[coeff]["epa_shifts"].append({
                dim: steered_epa[dim] - unsteered_epa[dim]
                for dim in DIMENSION_NAMES
            })
            results[coeff]["distances"].append(dist)

            text_preview = steered_text[:60].replace('\n', ' ')
            print(f"  coeff={coeff:5.1f}: E={steered_epa['evaluation']:+.2f}, "
                  f"P={steered_epa['potency']:+.2f}, A={steered_epa['activity']:+.2f}, "
                  f"dist={dist:.2f}, impr={improvement:+.2f} | {text_preview}...")

    # Summary
    print(f"\n{'='*80}")
    print("=== SUMMARY: Mean EPA shift from unsteered (across scenarios) ===\n")
    print(f"{'Coeff':>8} | {'dE':>8} {'dP':>8} {'dA':>8} | {'Mean dist':>9} | {'Coherent?':>9}")
    print("-" * 75)

    for coeff in args.coeffs:
        shifts = results[coeff]["epa_shifts"]
        if not shifts:
            continue
        mean_dE = np.mean([s["evaluation"] for s in shifts])
        mean_dP = np.mean([s["potency"] for s in shifts])
        mean_dA = np.mean([s["activity"] for s in shifts])
        mean_dist = np.mean(results[coeff]["distances"])

        # Check coherence: are texts gibberish?
        texts = results[coeff]["texts"]
        coherent = all(len(t) > 20 and not t.startswith("!!") for t in texts)

        print(f"{coeff:8.2f} | {mean_dE:+8.3f} {mean_dP:+8.3f} {mean_dA:+8.3f} | "
              f"{mean_dist:9.3f} | {'Yes' if coherent else 'NO'}")

    # Save
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "coefficients_tested": args.coeffs,
        "n_scenarios": len(scenarios),
        "results": {
            str(c): {
                "mean_distance": float(np.mean(r["distances"])),
                "mean_shifts": {
                    dim: float(np.mean([s[dim] for s in r["epa_shifts"]]))
                    for dim in DIMENSION_NAMES
                } if r["epa_shifts"] else {},
            }
            for c, r in results.items()
        }
    }
    with open(results_dir / "quick_coeff_search.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/quick_coeff_search.json")


if __name__ == "__main__":
    main()
