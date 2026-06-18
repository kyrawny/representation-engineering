"""
Experiment 05 — L2 Normalisation Ablation.

Compares steering performance with and without L2 normalisation
of the direction vectors.  Without normalisation, direction vectors
in 4096-dim space have norms of ~5-20, making coefficients unpredictable.

Usage::

    python -m examples.act_three.experiments.05_ablation_normalization
    python -m examples.act_three.experiments.05_ablation_normalization --quick
"""

import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .config import DIMENSION_NAMES, IDENTITY_PAIRS, GENERATION_DEFAULTS
from .setup import (
    load_experiment_components,
    make_system_prompt,
    save_results,
    generate_unsteered,
    clean_response,
)
from .scenarios import get_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 05: L2 Normalisation Ablation")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--n-scenarios", type=int, default=20,
                        help="Number of scenarios (default: 20)")
    parser.add_argument("--output", default="05_ablation_normalization.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import format_llama3_prompt
    from examples.act_three.epa_steerer import make_epa_activations

    # ---- Setup ----
    n = 5 if args.quick else args.n_scenarios
    scenarios = get_scenarios(quick=args.quick, n=n)[:n]

    # Fixed target EPA for controlled comparison
    target_epa = {"evaluation": 1.5, "potency": 0.5, "activity": 0.5}
    fixed_coeff = 0.5

    print(f"Comparing normalised vs unnormalised steering on {len(scenarios)} "
          f"scenarios at coeff={fixed_coeff}")

    # Use first identity pair for conversational context
    _, default_agent, default_user = IDENTITY_PAIRS[0]
    sys_prompt = make_system_prompt(default_agent, default_user)

    trials = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        prompt = format_llama3_prompt(sys_prompt, scenario["text"])

        # ---- Normalised (default) ----
        steered_norm = comp.steerer.generate(
            prompt=prompt,
            target_epa=target_epa,
            **GENERATION_DEFAULTS,
        )
        steered_norm = clean_response(steered_norm)

        # ---- Unnormalised ----
        # Build activations manually without normalisation
        all_layers = comp.steerer.all_layers
        activations_unnorm = make_epa_activations(
            rep_readers=comp.rep_readers,
            layers=all_layers,
            e_coeff=target_epa["evaluation"] * fixed_coeff,
            p_coeff=target_epa["potency"] * fixed_coeff,
            a_coeff=target_epa["activity"] * fixed_coeff,
            device=comp.model.device,
            dtype=comp.model.dtype,
            normalize=False,  # <-- key difference
        )

        outputs = comp.steerer.control_pipeline(
            prompt,
            activations=activations_unnorm,
            batch_size=1,
            **GENERATION_DEFAULTS,
        )
        steered_unnorm = outputs[0]["generated_text"]
        if steered_unnorm.startswith(prompt):
            steered_unnorm = steered_unnorm[len(prompt):]
        steered_unnorm = clean_response(steered_unnorm)

        # ---- Read EPA of both ----
        epa_norm = comp.reader.read_epa(
            comp.rep_reading_pipeline, steered_norm)
        epa_unnorm = comp.reader.read_epa(
            comp.rep_reading_pipeline, steered_unnorm)

        # ---- Basic coherence metrics ----
        def coherence_metrics(text):
            tokens = text.split()
            n_tokens = len(tokens)
            unique_ratio = len(set(tokens)) / max(n_tokens, 1)
            # Bigram repetition
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
            bigram_repeat = 1 - len(set(bigrams)) / max(len(bigrams), 1)
            return {
                "length_chars": len(text),
                "length_tokens": n_tokens,
                "unique_token_ratio": round(unique_ratio, 4),
                "bigram_repetition_ratio": round(bigram_repeat, 4),
            }

        trial = {
            "scenario_id": scenario["id"],
            "target_epa": target_epa,
            "coefficient": fixed_coeff,
            # Normalised
            "normalised_text": steered_norm,
            "normalised_epa": epa_norm,
            "normalised_coherence": coherence_metrics(steered_norm),
            # Unnormalised
            "unnormalised_text": steered_unnorm,
            "unnormalised_epa": epa_unnorm,
            "unnormalised_coherence": coherence_metrics(steered_unnorm),
        }
        trials.append(trial)

    # ---- Aggregate ----
    agg = {"normalised": {}, "unnormalised": {}}
    for condition in ["normalised", "unnormalised"]:
        for dim in DIMENSION_NAMES:
            distances = [
                abs(t["target_epa"][dim] - t[f"{condition}_epa"][dim])
                for t in trials
            ]
            agg[condition][dim] = {
                "mean_distance_to_target": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
            }
        # Coherence
        agg[condition]["mean_unique_token_ratio"] = float(np.mean(
            [t[f"{condition}_coherence"]["unique_token_ratio"] for t in trials]))
        agg[condition]["mean_bigram_repetition"] = float(np.mean(
            [t[f"{condition}_coherence"]["bigram_repetition_ratio"]
             for t in trials]))
        agg[condition]["mean_length"] = float(np.mean(
            [t[f"{condition}_coherence"]["length_chars"] for t in trials]))

    # Direction norms (for the paper)
    direction_norms = {}
    for dim in DIMENSION_NAMES:
        norms = []
        reader = comp.rep_readers[dim]
        for layer, direction in reader.directions.items():
            norms.append(float(np.linalg.norm(direction)))
        direction_norms[dim] = {
            "mean_norm": float(np.mean(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
        }

    results = {
        "metadata": {
            "experiment": "05_ablation_normalization",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "target_epa": target_epa,
            "coefficient": fixed_coeff,
            "quick_mode": args.quick,
        },
        "aggregate": agg,
        "direction_norms": direction_norms,
        "trials": trials,
    }
    save_results(results, args.output)


if __name__ == "__main__":
    main()
