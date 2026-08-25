"""
Experiment 08 — Hybrid Steering (Prompt Engineering + RepE).

Combines the strengths of prompt engineering (better for P and A per Exp 03)
with RepE steering (better for E per Exp 03).  For each trial:

1. Unsteered baseline (system prompt only, no affective instruction)
2. RepE-only: activation steering, neutral system prompt
3. PE-only: affective system prompt, no activation steering
4. Hybrid: affective system prompt + activation steering

Usage::

    python -m examples.act_three.experiments.08_hybrid_steering
    python -m examples.act_three.experiments.08_hybrid_steering --quick
"""

import argparse
import json
from datetime import datetime

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
    add_model_arg,
    get_identity_epa,
    make_system_prompt,
    save_results,
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


def _make_affective_system_prompt(
    agent_term: str,
    user_term: str,
    target_epa: dict,
) -> str:
    """Build a system prompt that includes an affective instruction."""

    def _describe(dim: str, val: float) -> str:
        if dim == "evaluation":
            if val > 1.0:
                return "warm, kind, and supportive"
            elif val > 0.0:
                return "polite and pleasant"
            elif val > -1.0:
                return "neutral and matter-of-fact"
            else:
                return "cold and disapproving"
        elif dim == "potency":
            if val > 1.0:
                return "authoritative and commanding"
            elif val > 0.0:
                return "confident and assertive"
            elif val > -1.0:
                return "gentle and moderate"
            else:
                return "meek and deferential"
        else:  # activity
            if val > 1.0:
                return "energetic and animated"
            elif val > 0.0:
                return "lively and engaged"
            elif val > -1.0:
                return "calm and measured"
            else:
                return "quiet and subdued"

    e_desc = _describe("evaluation", target_epa.get("evaluation", 0))
    p_desc = _describe("potency", target_epa.get("potency", 0))
    a_desc = _describe("activity", target_epa.get("activity", 0))

    return (
        f"Pretend you're a {agent_term} replying to a {user_term} "
        f"in a conversation. Your tone should be {e_desc}, {p_desc}, "
        f"and {a_desc}. Respond concisely in-character."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 08: Hybrid Steering (PE + RepE)")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="08_hybrid_steering.json",
                        help="Output filename in results/")
    add_model_arg(parser)
    args = parser.parse_args()

    comp = load_experiment_components(load_steerer=True, model_name=args.model)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
    )
    from examples.act_three.model_registry import format_chat_prompt

    scenarios = get_scenarios(quick=args.quick, n=QUICK_N_SCENARIOS)
    pairs = IDENTITY_PAIRS
    if args.quick:
        pairs = pairs[:QUICK_N_IDENTITY_PAIRS]

    total = len(scenarios) * len(pairs)
    print(f"Running {len(scenarios)} scenarios × {len(pairs)} pairs = {total}")

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

            # --- Condition 1: Unsteered (neutral system prompt) ---
            neutral_prompt = format_chat_prompt(
                comp.tokenizer,
                make_system_prompt(agent_term, user_term),
                scenario["text"])
            unsteered_text = generate_unsteered(
                comp.model, comp.tokenizer, neutral_prompt)
            unsteered_text = clean_response(unsteered_text)

            # --- Condition 2: RepE only (neutral system prompt + steering) ---
            repe_text = comp.steerer.generate(
                prompt=neutral_prompt,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            repe_text = clean_response(repe_text)

            # --- Condition 3: PE only (affective system prompt, no steering) ---
            affective_prompt = format_chat_prompt(
                comp.tokenizer,
                _make_affective_system_prompt(agent_term, user_term, target_dict),
                scenario["text"])
            pe_text = generate_unsteered(
                comp.model, comp.tokenizer, affective_prompt)
            pe_text = clean_response(pe_text)

            # --- Condition 4: Hybrid (affective prompt + RepE steering) ---
            hybrid_text = comp.steerer.generate(
                prompt=affective_prompt,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            hybrid_text = clean_response(hybrid_text)

            # Read EPA of all four conditions
            all_texts = [unsteered_text, repe_text, pe_text, hybrid_text]
            all_epas = [
                comp.reader.read_epa(comp.rep_reading_pipeline, t)
                for t in all_texts
            ]

            # Compute distances to target
            def _dist(epa_dict, dim):
                return abs(epa_dict[dim] - target_dict[dim])

            trial = {
                "scenario_id": scenario["id"],
                "pair": pair_name,
                "agent_identity": agent_term,
                "user_identity": user_term,
                "agent_identity_epa": {
                    "evaluation": agent_epa.e,
                    "potency": agent_epa.p,
                    "activity": agent_epa.a,
                },
                "user_identity_epa": {
                    "evaluation": user_epa.e,
                    "potency": user_epa.p,
                    "activity": user_epa.a,
                },
                "user_statement": scenario["text"],
                "user_statement_epa": user_msg_epa,
                "target_epa": target_dict,
                "conditions": {},
            }

            # Per-condition prompts
            condition_prompts = {
                "unsteered": neutral_prompt,
                "repe_only": neutral_prompt,
                "pe_only": affective_prompt,
                "hybrid": affective_prompt,
            }

            for cond_name, text, epa_dict in zip(
                ["unsteered", "repe_only", "pe_only", "hybrid"],
                all_texts,
                all_epas,
            ):
                distances = {dim: _dist(epa_dict, dim) for dim in DIMENSION_NAMES}
                trial["conditions"][cond_name] = {
                    "text": text[:500],
                    "prompt": condition_prompts[cond_name],
                    "epa": epa_dict,
                    "distances": distances,
                    "mean_distance": float(np.mean(list(distances.values()))),
                }

            trials.append(trial)

    # --- Aggregate statistics ---
    conditions = ["unsteered", "repe_only", "pe_only", "hybrid"]
    aggregate = {}

    for cond in conditions:
        per_dim_dists = {dim: [] for dim in DIMENSION_NAMES}
        mean_dists = []
        for trial in trials:
            for dim in DIMENSION_NAMES:
                per_dim_dists[dim].append(
                    trial["conditions"][cond]["distances"][dim])
            mean_dists.append(trial["conditions"][cond]["mean_distance"])

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

        aggregate[cond] = {
            "per_dimension": dim_stats,
            "overall_mean_distance": float(np.mean(mean_dists)),
        }

    # --- Pairwise comparisons (hybrid vs each other) ---
    comparisons = {}
    for baseline in ["unsteered", "repe_only", "pe_only"]:
        comp_results = {}
        for dim in DIMENSION_NAMES:
            hybrid_dists = [t["conditions"]["hybrid"]["distances"][dim]
                            for t in trials]
            baseline_dists = [t["conditions"][baseline]["distances"][dim]
                              for t in trials]

            # Positive delta = hybrid is closer (improvement)
            diff_mean, perm_p = paired_permutation_test(
                np.array(baseline_dists), np.array(hybrid_dists),
                alternative="greater")
            _, wilcox_p = wilcoxon_signed_rank(
                np.array(baseline_dists), np.array(hybrid_dists),
                alternative="greater")
            d = cohens_d_paired(
                np.array(baseline_dists), np.array(hybrid_dists))

            comp_results[dim] = {
                "mean_improvement": float(diff_mean),
                "permutation_p": float(perm_p),
                "wilcoxon_p": float(wilcox_p),
                "cohens_d": float(d),
            }

        comparisons[f"hybrid_vs_{baseline}"] = comp_results

    results = {
        "metadata": {
            "experiment": "08_hybrid_steering",
            "timestamp": datetime.now().isoformat(),
            "n_trials": len(trials),
            "n_scenarios": len(scenarios),
            "n_pairs": len(pairs),
            "quick_mode": args.quick,
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "trials": trials,
    }

    save_results(results, args.output)
    print(f"\n=== Results Summary ===")
    print(f"{'Condition':<15} {'E dist':>8} {'P dist':>8} {'A dist':>8} {'Overall':>8}")
    print("-" * 50)
    for cond in conditions:
        agg = aggregate[cond]
        e = agg["per_dimension"]["evaluation"]["mean_distance"]
        p = agg["per_dimension"]["potency"]["mean_distance"]
        a = agg["per_dimension"]["activity"]["mean_distance"]
        o = agg["overall_mean_distance"]
        print(f"{cond:<15} {e:>8.3f} {p:>8.3f} {a:>8.3f} {o:>8.3f}")


if __name__ == "__main__":
    main()
