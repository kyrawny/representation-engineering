"""
Experiment 04 — Steering Coefficient Sweep.

For each EPA dimension, sweeps the steering coefficient from 0 to 5
and records the achieved EPA shift.  Reveals the monotonic steering
relationship and the coherence cliff at extreme coefficients.

Usage::

    python -m examples.act_three.experiments.04_coefficient_sweep
    python -m examples.act_three.experiments.04_coefficient_sweep --quick
"""

import argparse
import json
import copy
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .config import (
    DIMENSION_NAMES,
    IDENTITY_PAIRS,
    COEFF_SWEEP_VALUES,
    GENERATION_DEFAULTS,
    QUICK_N_SCENARIOS,
)
from .setup import (
    load_experiment_components,
    make_system_prompt,
    save_results,
    clean_response,
)
from .scenarios import get_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 04: Coefficient Sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--n-scenarios", type=int, default=20,
                        help="Number of scenarios to test (default: 20)")
    parser.add_argument("--output", default="04_coefficient_sweep.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import format_llama3_prompt, EPASteerer

    # ---- Setup ----
    n = 5 if args.quick else args.n_scenarios
    scenarios = get_scenarios(quick=args.quick, n=n)[:n]
    coeffs = COEFF_SWEEP_VALUES
    if args.quick:
        coeffs = [0.0, 0.1, 0.5, 1.0, 2.0]

    print(f"Sweeping {len(coeffs)} coefficients × {len(scenarios)} scenarios "
          f"× 3 dimensions")

    # Use first identity pair for conversational context
    _, default_agent, default_user = IDENTITY_PAIRS[0]
    sys_prompt = make_system_prompt(default_agent, default_user)

    # We steer one dimension at a time, zeroing the others
    results_by_dim = {}

    for steer_dim in DIMENSION_NAMES:
        print(f"\n--- Sweeping {steer_dim} ---")
        sweep_data = []

        for scenario in tqdm(scenarios, desc=steer_dim):
            prompt = format_llama3_prompt(sys_prompt, scenario["text"])

            for coeff in coeffs:
                # Build target: +2.0 on the steered dimension, 0 on others
                # The magnitude direction is positive for positive steering
                target = {dim: 0.0 for dim in DIMENSION_NAMES}
                target[steer_dim] = 2.0  # Fixed positive target

                # Override the base_coeff for this dimension
                coeff_overrides = {
                    steer_dim: {
                        layer: coeff
                        for layer in comp.steerer.steering_configs[steer_dim]["layers"]
                    }
                }

                # Generate steered response
                steered_text = comp.steerer.generate(
                    prompt=prompt,
                    target_epa=target,
                    coeff_overrides=coeff_overrides,
                    **GENERATION_DEFAULTS,
                )
                steered_text = clean_response(steered_text)

                # Read EPA
                read_epa = comp.reader.read_epa(
                    comp.rep_reading_pipeline, steered_text)

                sweep_data.append({
                    "scenario_id": scenario["id"],
                    "coefficient": coeff,
                    "target_value": 2.0,
                    "steered_text": steered_text,
                    "read_epa": read_epa,
                    "response_length": len(steered_text),
                })

        results_by_dim[steer_dim] = sweep_data

    # ---- Compute sweep curves ----
    curves = {}
    for dim in DIMENSION_NAMES:
        curve = {}
        for coeff in coeffs:
            matching = [r for r in results_by_dim[dim]
                        if r["coefficient"] == coeff]
            on_target_vals = [r["read_epa"][dim] for r in matching]
            # Cross-dim values for interference measurement
            cross_vals = {}
            for other_dim in DIMENSION_NAMES:
                if other_dim != dim:
                    cross_vals[other_dim] = float(np.mean(
                        [r["read_epa"][other_dim] for r in matching]))
            curve[str(coeff)] = {
                "mean_on_target": float(np.mean(on_target_vals)),
                "std_on_target": float(np.std(on_target_vals)),
                "mean_response_length": float(np.mean(
                    [r["response_length"] for r in matching])),
                "cross_dimension_means": cross_vals,
            }
        curves[dim] = curve

    # ---- Save ----
    results = {
        "metadata": {
            "experiment": "04_coefficient_sweep",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "coefficients": coeffs,
            "target_value": 2.0,
            "quick_mode": args.quick,
        },
        "sweep_curves": curves,
        "raw_trials": results_by_dim,
    }
    save_results(results, args.output)


if __name__ == "__main__":
    main()
