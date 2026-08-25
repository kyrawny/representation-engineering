"""
Experiment 07 — Coherence Evaluation.

Measures whether steering degrades text quality by computing:
    - Perplexity (self-model and/or GPT-2)
    - Response length, unique token ratio, bigram repetition
    - VADER sentiment as an external coherence proxy

Supports two perplexity modes (toggleable):
    --perplexity-model self     Use the same Llama model (default)
    --perplexity-model gpt2     Use GPT-2 as an independent evaluator
    --perplexity-model both     Run both

Usage::

    python -m examples.act_three.experiments.07_coherence_evaluation
    python -m examples.act_three.experiments.07_coherence_evaluation --quick
    python -m examples.act_three.experiments.07_coherence_evaluation --perplexity-model both
"""

import argparse
import math
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .config import DIMENSION_NAMES, GENERATION_DEFAULTS
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


# =========================================================================
# Perplexity computation
# =========================================================================

def compute_perplexity_self(
    model, tokenizer, texts: list, batch_size: int = 4,
) -> list:
    """
    Compute perplexity using the same model that generated the text.

    Uses teacher-forced log-likelihood: the model sees the full text
    and we compute the average per-token negative log-likelihood.
    """
    perplexities = []
    for text in texts:
        if not text.strip():
            perplexities.append(float("inf"))
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        perplexities.append(math.exp(loss))

    return perplexities


def compute_perplexity_gpt2(texts: list) -> list:
    """
    Compute perplexity using GPT-2 as an independent evaluator.

    Loads GPT-2 on CPU (small enough) to avoid competing for GPU memory.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2 for perplexity evaluation...")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model.eval()

    perplexities = []
    for text in texts:
        if not text.strip():
            perplexities.append(float("inf"))
            continue

        inputs = gpt2_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        perplexities.append(math.exp(loss))

    # Free memory
    del gpt2_model, gpt2_tokenizer
    return perplexities


# =========================================================================
# Text quality metrics
# =========================================================================

def compute_text_metrics(text: str) -> dict:
    """Compute surface-level text quality metrics."""
    tokens = text.split()
    n_tokens = len(tokens)
    n_chars = len(text)

    # Unique token ratio
    unique_ratio = len(set(tokens)) / max(n_tokens, 1)

    # Bigram repetition ratio
    if n_tokens >= 2:
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(n_tokens - 1)]
        bigram_repeat = 1 - len(set(bigrams)) / max(len(bigrams), 1)
    else:
        bigram_repeat = 0.0

    # Trigram repetition
    if n_tokens >= 3:
        trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                    for i in range(n_tokens - 2)]
        trigram_repeat = 1 - len(set(trigrams)) / max(len(trigrams), 1)
    else:
        trigram_repeat = 0.0

    return {
        "length_chars": n_chars,
        "length_tokens": n_tokens,
        "unique_token_ratio": round(unique_ratio, 4),
        "bigram_repetition_ratio": round(bigram_repeat, 4),
        "trigram_repetition_ratio": round(trigram_repeat, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 07: Coherence Evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--n-scenarios", type=int, default=50,
                        help="Number of scenarios (default: 50)")
    parser.add_argument("--perplexity-model", default="self",
                        choices=["self", "gpt2", "both"],
                        help="Which model to use for perplexity")
    parser.add_argument("--output", default="07_coherence_evaluation.json",
                        help="Output filename in results/")
    add_model_arg(parser)
    args = parser.parse_args()

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True, model_name=args.model)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
    )
    from examples.act_three.model_registry import format_chat_prompt
    from .config import IDENTITY_PAIRS

    # ---- Setup ----
    n = 10 if args.quick else args.n_scenarios
    scenarios = get_scenarios(quick=args.quick, n=n)[:n]

    # Use a single identity pair (counselor/client) for controlled comparison
    _, agent_term, user_term = IDENTITY_PAIRS[0]
    agent_epa = get_identity_epa(comp.identities_df, agent_term)
    user_epa_id = get_identity_epa(comp.identities_df, user_term)

    # Build system prompt for the identity pair
    sys_prompt = make_system_prompt(agent_term, user_term)

    print(f"Evaluating coherence on {len(scenarios)} scenarios")
    print(f"Perplexity model: {args.perplexity_model}")

    # ---- Generate responses ----
    unsteered_texts = []
    steered_texts = []
    target_epas = []

    for scenario in tqdm(scenarios, desc="Generating"):
        prompt = format_chat_prompt(comp.tokenizer, sys_prompt, scenario["text"])

        # Read user EPA and compute target
        user_msg_epa = comp.reader.read_epa(
            comp.rep_reading_pipeline, scenario["text"])
        user_behavior = EPA(
            e=user_msg_epa["evaluation"],
            p=user_msg_epa["potency"],
            a=user_msg_epa["activity"],
        )
        target = get_response_epa_for_deflection_minimization(
            agent_identity=agent_epa,
            user_identity=user_epa_id,
            user_behavior_epa=user_behavior,
            coefficients=comp.coefficients,
        )
        target_dict = {
            "evaluation": target.e,
            "potency": target.p,
            "activity": target.a,
        }
        target_epas.append(target_dict)

        # Unsteered
        un_text = generate_unsteered(comp.model, comp.tokenizer, prompt)
        unsteered_texts.append(un_text)

        # Steered
        st_text = comp.steerer.generate(
            prompt=prompt, target_epa=target_dict, **GENERATION_DEFAULTS)
        steered_texts.append(clean_response(st_text))

    # ---- Perplexity ----
    ppl_results = {"unsteered": {}, "steered": {}}

    if args.perplexity_model in ("self", "both"):
        print("Computing self-model perplexity...")
        ppl_results["unsteered"]["self"] = compute_perplexity_self(
            comp.model, comp.tokenizer, unsteered_texts)
        ppl_results["steered"]["self"] = compute_perplexity_self(
            comp.model, comp.tokenizer, steered_texts)

    if args.perplexity_model in ("gpt2", "both"):
        print("Computing GPT-2 perplexity...")
        ppl_results["unsteered"]["gpt2"] = compute_perplexity_gpt2(
            unsteered_texts)
        ppl_results["steered"]["gpt2"] = compute_perplexity_gpt2(
            steered_texts)

    # ---- Text quality metrics ----
    unsteered_metrics = [compute_text_metrics(t) for t in unsteered_texts]
    steered_metrics = [compute_text_metrics(t) for t in steered_texts]

    # ---- VADER ----
    import importlib
    _baseline_mod = importlib.import_module(
        ".03_prompt_engineering_baseline", package=__package__)
    compute_vader_scores = _baseline_mod.compute_vader_scores
    vader_unsteered = compute_vader_scores(unsteered_texts)
    vader_steered = compute_vader_scores(steered_texts)

    # ---- Build per-trial records ----
    trials = []
    for i, scenario in enumerate(scenarios):
        trial = {
            "scenario_id": scenario["id"],
            "target_epa": target_epas[i],
            "unsteered_text": unsteered_texts[i],
            "steered_text": steered_texts[i],
            "unsteered_text_metrics": unsteered_metrics[i],
            "steered_text_metrics": steered_metrics[i],
            "unsteered_vader": vader_unsteered[i],
            "steered_vader": vader_steered[i],
        }
        for model_name in ppl_results["unsteered"]:
            trial[f"unsteered_ppl_{model_name}"] = ppl_results["unsteered"][model_name][i]
            trial[f"steered_ppl_{model_name}"] = ppl_results["steered"][model_name][i]
        trials.append(trial)

    # ---- Aggregate ----
    aggregate = {"unsteered": {}, "steered": {}}
    for condition, texts, metrics_list in [
        ("unsteered", unsteered_texts, unsteered_metrics),
        ("steered", steered_texts, steered_metrics),
    ]:
        aggregate[condition]["mean_length_chars"] = float(np.mean(
            [m["length_chars"] for m in metrics_list]))
        aggregate[condition]["mean_length_tokens"] = float(np.mean(
            [m["length_tokens"] for m in metrics_list]))
        aggregate[condition]["mean_unique_token_ratio"] = float(np.mean(
            [m["unique_token_ratio"] for m in metrics_list]))
        aggregate[condition]["mean_bigram_repetition"] = float(np.mean(
            [m["bigram_repetition_ratio"] for m in metrics_list]))
        aggregate[condition]["mean_trigram_repetition"] = float(np.mean(
            [m["trigram_repetition_ratio"] for m in metrics_list]))

        for model_name in ppl_results[condition]:
            ppls = [p for p in ppl_results[condition][model_name]
                    if not math.isinf(p)]
            if ppls:
                aggregate[condition][f"mean_ppl_{model_name}"] = float(np.mean(ppls))
                aggregate[condition][f"median_ppl_{model_name}"] = float(np.median(ppls))

    results = {
        "metadata": {
            "experiment": "07_coherence_evaluation",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "perplexity_model": args.perplexity_model,
            "identity_pair": IDENTITY_PAIRS[0][0],
            "quick_mode": args.quick,
        },
        "aggregate": aggregate,
        "trials": trials,
    }
    save_results(results, args.output)


if __name__ == "__main__":
    main()
