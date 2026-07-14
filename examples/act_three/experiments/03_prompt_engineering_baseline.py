"""
Experiment 03 — Prompt-Engineering Baseline + VADER Sentiment Baseline.

Compares three approaches for achieving target EPA:
    1. RepE steering (from experiment 02 results)
    2. Prompt engineering (explicit affective instructions in the system prompt)
    3. VADER sentiment analysis as an independent external validator

Usage::

    python -m examples.act_three.experiments.03_prompt_engineering_baseline
    python -m examples.act_three.experiments.03_prompt_engineering_baseline --quick
"""

import argparse
import time
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


# =========================================================================
# EPA → natural language descriptors
# =========================================================================

def epa_to_instruction(target_epa: dict) -> str:
    """
    Map a target EPA to a natural-language affective instruction.

    Returns a sentence like:
        "Respond warmly and supportively, with authority, in an
         energetic and animated manner."
    """
    parts = []

    e = target_epa["evaluation"]
    if e > 1.0:
        parts.append("warmly, supportively, and with kindness")
    elif e > 0.3:
        parts.append("in a positive and friendly manner")
    elif e < -1.0:
        parts.append("coldly, critically, and with hostility")
    elif e < -0.3:
        parts.append("in a negative and disapproving manner")

    p = target_epa["potency"]
    if p > 1.5:
        parts.append("with strong authority and dominance")
    elif p > 0.5:
        parts.append("firmly and assertively")
    elif p < -1.0:
        parts.append("meekly and deferentially")
    elif p < -0.3:
        parts.append("gently and softly")

    a = target_epa["activity"]
    if a > 1.0:
        parts.append("in an energetic and animated way")
    elif a > 0.3:
        parts.append("with moderate energy")
    elif a < -0.5:
        parts.append("in a calm and measured way")
    elif a < 0.0:
        parts.append("in a quiet and subdued way")

    if not parts:
        return "Respond naturally."

    return "Respond " + ", ".join(parts) + "."


# =========================================================================
# VADER sentiment analysis
# =========================================================================

def compute_vader_scores(texts: list) -> list:
    """
    Compute VADER compound sentiment for a list of texts.

    Returns a list of dicts with 'compound', 'pos', 'neg', 'neu' keys.
    Falls back gracefully if vaderSentiment is not installed.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("WARNING: vaderSentiment not installed. "
              "Run: pip install vaderSentiment")
        return [{"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
                for _ in texts]

    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        results.append({
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 03: Prompt-Engineering & VADER Baselines")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--output", default="03_prompt_baseline.json",
                        help="Output filename in results/")
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
        format_llama3_prompt,
    )

    # ---- Setup ----
    scenarios = get_scenarios(quick=args.quick, n=QUICK_N_SCENARIOS)
    pairs = IDENTITY_PAIRS
    if args.quick:
        pairs = pairs[:QUICK_N_IDENTITY_PAIRS]

    total = len(scenarios) * len(pairs)
    print(f"Running {len(scenarios)} scenarios × {len(pairs)} pairs = {total} trials")

    trials = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for pair_name, agent_term, user_term in pairs:
            agent_epa = get_identity_epa(comp.identities_df, agent_term)
            user_epa_identity = get_identity_epa(comp.identities_df, user_term)

            # 1. Read user EPA & compute target
            user_msg_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, scenario["text"])
            user_behavior = EPA(
                e=user_msg_epa["evaluation"],
                p=user_msg_epa["potency"],
                a=user_msg_epa["activity"],
            )
            target_epa = get_response_epa_for_deflection_minimization(
                agent_identity=agent_epa,
                user_identity=user_epa_identity,
                user_behavior_epa=user_behavior,
                coefficients=comp.coefficients,
            )
            target_dict = {
                "evaluation": target_epa.e,
                "potency": target_epa.p,
                "activity": target_epa.a,
            }

            # 2. Unsteered baseline (with identity preamble)
            sys_prompt = make_system_prompt(agent_term, user_term)
            prompt_base = format_llama3_prompt(sys_prompt, scenario["text"])
            unsteered_text = generate_unsteered(
                comp.model, comp.tokenizer, prompt_base)

            # 3. Prompt-engineered baseline
            affective_instruction = epa_to_instruction(target_dict)
            system_prompt_pe = (
                f"You are a {agent_term} speaking with a {user_term}. "
                f"{affective_instruction} Keep your response concise."
            )
            prompt_pe = format_llama3_prompt(system_prompt_pe, scenario["text"])
            pe_text = generate_unsteered(
                comp.model, comp.tokenizer, prompt_pe)

            # 4. Steered (RepE) — same base prompt as unsteered
            steered_text = comp.steerer.generate(
                prompt=prompt_base,
                target_epa=target_dict,
                **GENERATION_DEFAULTS,
            )
            steered_text = clean_response(steered_text)

            # 5. Read EPA of all three responses
            unsteered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, unsteered_text)
            pe_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, pe_text)
            steered_epa = comp.reader.read_epa(
                comp.rep_reading_pipeline, steered_text)

            # 6. VADER on all three
            vader_scores = compute_vader_scores(
                [unsteered_text, pe_text, steered_text])

            trial = {
                "scenario_id": scenario["id"],
                "identity_pair": pair_name,
                "target_epa": target_dict,
                "affective_instruction": affective_instruction,
                # Unsteered
                "unsteered_text": unsteered_text,
                "unsteered_epa": unsteered_epa,
                "unsteered_vader": vader_scores[0],
                # Prompt-engineered
                "prompt_engineered_text": pe_text,
                "prompt_engineered_epa": pe_epa,
                "prompt_engineered_vader": vader_scores[1],
                # RepE steered
                "steered_text": steered_text,
                "steered_epa": steered_epa,
                "steered_vader": vader_scores[2],
            }
            trials.append(trial)

    # ---- Aggregate metrics ----
    metrics = _compute_comparison_metrics(trials)

    results = {
        "metadata": {
            "experiment": "03_prompt_engineering_baseline",
            "timestamp": datetime.now().isoformat(),
            "n_trials": len(trials),
            "quick_mode": args.quick,
        },
        "comparison_metrics": metrics,
        "trials": trials,
    }
    save_results(results, args.output)


def _compute_comparison_metrics(trials: list) -> dict:
    """Compare distance-to-target for all three methods with CIs."""
    methods = ["unsteered", "prompt_engineered", "steered"]
    metrics = {m: {} for m in methods}

    for dim in DIMENSION_NAMES:
        for method in methods:
            distances = np.array([
                abs(t["target_epa"][dim] - t[f"{method}_epa"][dim])
                for t in trials
            ])
            point, ci_lo, ci_hi = bootstrap_ci(
                distances, lambda x: float(np.mean(x)))
            metrics[method][dim] = {
                "mean_distance": point,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "std_distance": float(np.std(distances)),
                "median_distance": float(np.median(distances)),
            }

    # Pairwise comparisons: RepE vs PE, RepE vs Unsteered, PE vs Unsteered
    pairwise = {}
    for name_a, name_b in [("steered", "prompt_engineered"),
                           ("steered", "unsteered"),
                           ("prompt_engineered", "unsteered")]:
        pair_key = f"{name_a}_vs_{name_b}"
        pairwise[pair_key] = {}
        for dim in DIMENSION_NAMES:
            dists_a = np.array([
                abs(t["target_epa"][dim] - t[f"{name_a}_epa"][dim])
                for t in trials
            ])
            dists_b = np.array([
                abs(t["target_epa"][dim] - t[f"{name_b}_epa"][dim])
                for t in trials
            ])
            # Positive = name_b has larger distance = name_a is closer
            mean_diff, perm_p = paired_permutation_test(dists_b, dists_a)
            try:
                _, wilcox_p = wilcoxon_signed_rank(dists_b, dists_a)
            except ValueError:
                wilcox_p = 1.0
            d = cohens_d_paired(dists_b, dists_a)
            hit_rate = float(np.mean(dists_a < dists_b))

            pairwise[pair_key][dim] = {
                "hit_rate": hit_rate,
                "mean_improvement": float(mean_diff),
                "permutation_p": float(perm_p),
                "wilcoxon_p": float(wilcox_p),
                "cohens_d": float(d),
            }
    metrics["pairwise_tests"] = pairwise

    # VADER correlation with Evaluation dimension
    from scipy.stats import spearmanr

    for method in methods:
        vader_compounds = [t[f"{method}_vader"]["compound"] for t in trials]
        eval_readings = [t[f"{method}_epa"]["evaluation"] for t in trials]
        rho, pval = spearmanr(vader_compounds, eval_readings)
        metrics[f"{method}_vader_eval_correlation"] = {
            "spearman_rho": float(rho),
            "pval": float(pval),
        }

    return metrics


if __name__ == "__main__":
    main()
