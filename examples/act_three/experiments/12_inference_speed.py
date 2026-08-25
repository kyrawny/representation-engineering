"""
Experiment 12 — Inference Speed Benchmark.

Measures the wall-clock time for generating responses under different
steering conditions to quantify the latency overhead of RepE steering.

Conditions benchmarked:
  1. Unsteered (raw generation)
  2. RepE-only (activation steering, neutral prompt)
  3. PE-only (affective prompt, no activation steering)
  4. Hybrid (affective prompt + activation steering)

Reports per-token and per-generation latencies, plus the relative
overhead of each steering method compared to unsteered generation.

Usage::

    python -m examples.act_three.experiments.12_inference_speed
    python -m examples.act_three.experiments.12_inference_speed --quick
    python -m examples.act_three.experiments.12_inference_speed --n-trials 50 --warmup 5
"""

import argparse
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .config import (
    DIMENSION_NAMES,
    IDENTITY_PAIRS,
    GENERATION_DEFAULTS,
    QUICK_N_SCENARIOS,
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


def _count_tokens(tokenizer, text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _make_affective_system_prompt(agent_term, user_term, target_epa):
    """Build a system prompt with affective instruction (same as Exp 08)."""

    def _describe(dim, val):
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
        else:
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
        description="Experiment 12: Inference Speed Benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small subset for debugging")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of trials (default: 30, quick: 5)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup generations to discard "
                             "(default: 3)")
    parser.add_argument("--output", default="12_inference_speed.json",
                        help="Output filename in results/")
    add_model_arg(parser)
    args = parser.parse_args()

    n_trials = args.n_trials
    if n_trials is None:
        n_trials = 5 if args.quick else 30
    n_warmup = 1 if args.quick else args.warmup

    # ---- Load components ----
    comp = load_experiment_components(load_steerer=True, model_name=args.model)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
    )
    from examples.act_three.model_registry import format_chat_prompt

    # ---- Select scenarios ----
    scenarios = get_scenarios(quick=args.quick, n=max(n_trials + n_warmup, 20))
    if len(scenarios) < n_trials + n_warmup:
        # Repeat scenarios if we don't have enough
        reps = (n_trials + n_warmup) // len(scenarios) + 1
        scenarios = (scenarios * reps)[: n_trials + n_warmup]
    else:
        scenarios = scenarios[: n_trials + n_warmup]

    # Use a single identity pair for consistency
    pair_name, agent_term, user_term = IDENTITY_PAIRS[0]
    agent_epa = get_identity_epa(comp.identities_df, agent_term)
    user_epa = get_identity_epa(comp.identities_df, user_term)
    sys_prompt = make_system_prompt(agent_term, user_term)

    print(f"Benchmarking with {n_trials} trials + {n_warmup} warmup")
    print(f"Identity pair: {pair_name}")
    print(f"Generation settings: {GENERATION_DEFAULTS}")

    # ---- Precompute targets and prompts ----
    trial_data = []
    for scenario in scenarios:
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

        neutral_prompt = format_chat_prompt(comp.tokenizer, sys_prompt, scenario["text"])
        affective_prompt = format_chat_prompt(
            comp.tokenizer,
            _make_affective_system_prompt(agent_term, user_term, target_dict),
            scenario["text"])

        trial_data.append({
            "scenario_id": scenario["id"],
            "target_epa": target_dict,
            "neutral_prompt": neutral_prompt,
            "affective_prompt": affective_prompt,
        })

    print("Precomputed all targets and prompts.\n")

    # ---- Define benchmark conditions ----
    conditions = {
        "unsteered": {
            "description": "Raw generation, neutral prompt",
            "uses_steering": False,
            "uses_affective_prompt": False,
        },
        "repe_only": {
            "description": "RepE steering, neutral prompt",
            "uses_steering": True,
            "uses_affective_prompt": False,
        },
        "pe_only": {
            "description": "No steering, affective prompt",
            "uses_steering": False,
            "uses_affective_prompt": True,
        },
        "hybrid": {
            "description": "RepE steering + affective prompt",
            "uses_steering": True,
            "uses_affective_prompt": True,
        },
    }

    # ---- Run benchmark ----
    results_per_condition = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"\n--- Benchmarking: {cond_name} ({cond_cfg['description']}) ---")

        timings = []  # wall-clock seconds per generation
        token_counts = []  # output tokens per generation

        for i, td in enumerate(tqdm(trial_data, desc=cond_name)):
            is_warmup = i < n_warmup

            prompt = (td["affective_prompt"]
                      if cond_cfg["uses_affective_prompt"]
                      else td["neutral_prompt"])

            # Sync GPU before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            if cond_cfg["uses_steering"]:
                text = comp.steerer.generate(
                    prompt=prompt,
                    target_epa=td["target_epa"],
                    **GENERATION_DEFAULTS,
                )
                text = clean_response(text)
            else:
                text = generate_unsteered(
                    comp.model, comp.tokenizer, prompt)

            # Sync GPU after generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            elapsed = t_end - t_start

            if not is_warmup:
                n_tokens = _count_tokens(comp.tokenizer, text)
                timings.append(elapsed)
                token_counts.append(n_tokens)

        timings = np.array(timings)
        token_counts = np.array(token_counts)

        # Compute per-token latency (avoid division by zero)
        per_token_times = np.array([
            t / max(n, 1) for t, n in zip(timings, token_counts)
        ])

        stats = {
            "n_trials": len(timings),
            "total_time_seconds": float(np.sum(timings)),
            "mean_time_per_generation": float(np.mean(timings)),
            "std_time_per_generation": float(np.std(timings)),
            "median_time_per_generation": float(np.median(timings)),
            "p5_time": float(np.percentile(timings, 5)),
            "p95_time": float(np.percentile(timings, 95)),
            "mean_output_tokens": float(np.mean(token_counts)),
            "mean_time_per_token": float(np.mean(per_token_times)),
            "std_time_per_token": float(np.std(per_token_times)),
            "mean_tokens_per_second": float(
                np.mean(token_counts / np.maximum(timings, 1e-6))),
        }

        results_per_condition[cond_name] = stats

        print(f"  Mean: {stats['mean_time_per_generation']:.3f}s "
              f"± {stats['std_time_per_generation']:.3f}s per generation")
        print(f"  Mean: {stats['mean_time_per_token']*1000:.1f}ms per token "
              f"({stats['mean_tokens_per_second']:.1f} tok/s)")
        print(f"  Mean output length: {stats['mean_output_tokens']:.0f} tokens")

    # ---- Compute overhead relative to unsteered ----
    overhead = {}
    baseline_mean = results_per_condition["unsteered"]["mean_time_per_generation"]
    baseline_per_tok = results_per_condition["unsteered"]["mean_time_per_token"]

    for cond_name in conditions:
        cond_mean = results_per_condition[cond_name]["mean_time_per_generation"]
        cond_per_tok = results_per_condition[cond_name]["mean_time_per_token"]

        abs_overhead = cond_mean - baseline_mean
        rel_overhead = (abs_overhead / baseline_mean * 100
                        if baseline_mean > 0 else 0.0)

        abs_overhead_per_tok = cond_per_tok - baseline_per_tok
        rel_overhead_per_tok = (abs_overhead_per_tok / baseline_per_tok * 100
                                if baseline_per_tok > 0 else 0.0)

        overhead[cond_name] = {
            "absolute_overhead_seconds": float(abs_overhead),
            "relative_overhead_percent": float(rel_overhead),
            "absolute_overhead_per_token_seconds": float(abs_overhead_per_tok),
            "relative_overhead_per_token_percent": float(rel_overhead_per_tok),
        }

    # ---- GPU info ----
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated_gb": round(
                torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_reserved_gb": round(
                torch.cuda.memory_reserved(0) / 1e9, 2),
        }

    # ---- Save results ----
    results = {
        "metadata": {
            "experiment": "12_inference_speed",
            "timestamp": datetime.now().isoformat(),
            "n_trials": n_trials,
            "n_warmup": n_warmup,
            "identity_pair": pair_name,
            "generation_defaults": GENERATION_DEFAULTS,
            "quick_mode": args.quick,
            "gpu_info": gpu_info,
        },
        "per_condition": results_per_condition,
        "overhead_vs_unsteered": overhead,
    }

    save_results(results, args.output)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print(f"  INFERENCE SPEED SUMMARY ({n_trials} trials, {n_warmup} warmup)")
    print(f"{'='*70}")

    print(f"\n{'Condition':<12} {'Mean (s)':>10} {'Std (s)':>10} "
          f"{'Tok/s':>8} {'Overhead':>10}")
    print("-" * 56)
    for cond_name in conditions:
        s = results_per_condition[cond_name]
        o = overhead[cond_name]
        ovh_str = (f"{o['relative_overhead_percent']:+.1f}%"
                   if cond_name != "unsteered" else "baseline")
        print(f"{cond_name:<12} {s['mean_time_per_generation']:>10.3f} "
              f"{s['std_time_per_generation']:>10.3f} "
              f"{s['mean_tokens_per_second']:>8.1f} "
              f"{ovh_str:>10}")

    if gpu_info:
        print(f"\nGPU: {gpu_info.get('device_name', 'N/A')}")


if __name__ == "__main__":
    main()
