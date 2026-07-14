"""
Potency Direction Template Search.

Extracts Potency directions using multiple candidate template pairs and
evaluates reading quality of each, keeping E and A directions fixed.

Usage::

    python -m examples.act_three.experiments.potency_template_search
    python -m examples.act_three.experiments.potency_template_search --top-k 3
"""

import argparse
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


# ---- Candidate template pairs for Potency ----
# Each pair is (positive_adjective, negative_adjective) inserted into:
#   "Pretend you're {adj} person making a statement in a conversation."

CANDIDATE_TEMPLATES = {
    # --- V1 original ---
    "v1_powerful_weak": ("a powerful", "a weak"),

    # --- Current (v2) ---
    "v2_authoritative_meek": ("an authoritative and commanding", "a meek and submissive"),

    # --- Dominance-focused ---
    "dominant_submissive": ("a dominant", "a submissive"),
    "commanding_yielding": ("a commanding", "a yielding"),
    "controlling_compliant": ("a controlling", "a compliant"),
    "assertive_timid": ("an assertive", "a timid"),
    "forceful_gentle": ("a forceful", "a gentle"),
    "imposing_unassuming": ("an imposing", "an unassuming"),

    # --- Authority-focused ---
    "authoritative_deferential": ("an authoritative", "a deferential"),
    "bossy_meek": ("a bossy", "a meek"),

    # --- Strength-focused (closer to ACT's Potency definition) ---
    "strong_weak": ("a strong", "a weak"),
    "potent_impotent": ("a potent", "an impotent"),
    "mighty_feeble": ("a mighty", "a feeble"),
    "bold_timid": ("a bold", "a timid"),

    # --- Multi-word refined ---
    "confident_insecure": ("a confident and decisive", "an insecure and hesitant"),
    "powerful_powerless": ("a powerful and influential", "a powerless and insignificant"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Search for the best Potency extraction template")
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Show top-K templates by Spearman rho")
    parser.add_argument(
        "--n-train", type=int, default=256,
        help="Number of contrastive training pairs per template")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for rep-reading")
    parser.add_argument(
        "--templates", nargs="*", default=None,
        help="Only test specific template names (default: all)")
    args = parser.parse_args()

    act_three_dir = Path(__file__).resolve().parent.parent
    repo_root = act_three_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    results_dir = act_three_dir / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print("Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    from repe import repe_pipeline_registry

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    repe_pipeline_registry()
    rep_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    # ---- Load test data ----
    test_path = act_three_dir / "epa_tuning_dataset_test.json"
    print(f"Loading test data from: {test_path}")
    with open(test_path, "r") as f:
        test_raw = json.load(f)
    test_items = test_raw.get("utterances", test_raw.get("data", test_raw))
    if isinstance(test_items, dict):
        test_items = list(test_items.values())[0]
    print(f"  Test: {len(test_items)} items")

    test_texts = [item["text"] for item in test_items]
    gt_potency = np.array([item["target_epa"]["p"] for item in test_items])

    # ---- Load base training data for contrastive pairs ----
    data_dir = str(repo_root / "data" / "act")
    from examples.act_three.dataset import load_act_data
    from examples.act_three.prompt_formatting import format_llama3_prompt, format_for_reading

    _user_inputs, truncated_outputs = load_act_data(data_dir)

    # ---- Load existing E and A directions for cross-dim cosine check ----
    v1_path = act_three_dir / "backups" / "v1" / "epa_directions.pkl"
    if v1_path.exists():
        with open(v1_path, "rb") as f:
            v1_data = pickle.load(f)
        v1_readers = v1_data["rep_readers"]
        hidden_layers = v1_data["hidden_layers"]
        print(f"Loaded v1 E/A directions for cosine comparison")
    else:
        print("WARNING: v1 backup not found, skipping cross-dim check")
        v1_readers = None
        hidden_layers = list(range(-1, -32, -1))

    all_layers = sorted(hidden_layers)

    # ---- Format test texts for reading ----
    formatted_test = [format_for_reading(t) for t in test_texts]

    # ---- Test each template ----
    templates_to_test = args.templates or list(CANDIDATE_TEMPLATES.keys())
    results = {}

    for name in templates_to_test:
        if name not in CANDIDATE_TEMPLATES:
            print(f"  SKIP: Unknown template '{name}'")
            continue

        pos_adj, neg_adj = CANDIDATE_TEMPLATES[name]
        template_pos = f"Pretend you're {pos_adj} person making a statement in a conversation."
        template_neg = f"Pretend you're {neg_adj} person making a statement in a conversation."

        print(f"\n{'='*60}")
        print(f"Template: {name}")
        print(f"  Positive: '{template_pos}'")
        print(f"  Negative: '{template_neg}'")

        # Create contrastive dataset
        random.seed(42)
        np.random.seed(42)

        train_data = []
        train_labels = []
        for _ in range(args.n_train):
            truncated = random.choice(truncated_outputs)
            pos_prompt = format_llama3_prompt(None, template_pos, truncated)
            neg_prompt = format_llama3_prompt(None, template_neg, truncated)
            pair = [pos_prompt, neg_prompt]
            random.shuffle(pair)
            train_labels.append([pair[0] == pos_prompt, pair[1] == pos_prompt])
            train_data.extend(pair)

        dataset = {"train": {"data": train_data, "labels": train_labels}}

        # Extract direction
        t0 = time.time()
        from repe.rep_readers import PCARepReader
        rep_reader = PCARepReader(n_components=1)
        rep_reader.fit(
            rep_pipeline,
            dataset,
            hidden_layers=all_layers,
            batch_size=args.batch_size,
        )
        extract_time = time.time() - t0
        print(f"  Extracted in {extract_time:.1f}s")

        # Evaluate reading quality on test set
        raw_scores = rep_pipeline(
            formatted_test,
            hidden_layers=all_layers,
            rep_reader=rep_reader,
            batch_size=args.batch_size,
            padding=True,
            truncation=True,
        )

        # Try each layer individually to find best
        best_rho = -1
        best_layer = None
        layer_rhos = {}

        for layer in all_layers:
            scores = np.array([float(raw_scores[i][layer]) for i in range(len(test_texts))])
            if np.std(scores) < 1e-10:
                continue
            rho, pval = spearmanr(scores, gt_potency)
            layer_rhos[layer] = float(abs(rho))
            if abs(rho) > best_rho:
                best_rho = abs(rho)
                best_layer = layer

        # Multi-layer weighted combination (top-5 by |rho|)
        top5 = sorted(layer_rhos.items(), key=lambda x: x[1], reverse=True)[:5]
        combined = np.zeros(len(test_texts))
        total_weight = 0
        for layer, rho_val in top5:
            scores = np.array([float(raw_scores[i][layer]) for i in range(len(test_texts))])
            # Determine sign from correlation
            actual_rho, _ = spearmanr(scores, gt_potency)
            sign = 1.0 if actual_rho > 0 else -1.0
            combined += rho_val * sign * scores
            total_weight += rho_val
        if total_weight > 0:
            combined /= total_weight

        combined_rho, combined_p = spearmanr(combined, gt_potency)
        combined_r, _ = pearsonr(combined, gt_potency)

        # Cross-dimension cosine with E direction (from v1)
        cross_cos_ep = None
        if v1_readers is not None:
            cosines = []
            for layer in all_layers:
                e_vec = v1_readers["evaluation"].directions[layer].flatten()
                p_vec = rep_reader.directions[layer].flatten()
                cos = abs(float(np.dot(e_vec, p_vec) /
                                (np.linalg.norm(e_vec) * np.linalg.norm(p_vec) + 1e-12)))
                cosines.append(cos)
            cross_cos_ep = float(np.mean(cosines))

        # Cosine with v1 potency direction
        cos_with_v1 = None
        if v1_readers is not None:
            cosines = []
            for layer in all_layers:
                v1_vec = v1_readers["potency"].directions[layer].flatten()
                new_vec = rep_reader.directions[layer].flatten()
                cos = float(np.dot(v1_vec, new_vec) /
                            (np.linalg.norm(v1_vec) * np.linalg.norm(new_vec) + 1e-12))
                cosines.append(cos)
            cos_with_v1 = float(np.mean([abs(c) for c in cosines]))

        result = {
            "template_pos": template_pos,
            "template_neg": template_neg,
            "best_single_layer": best_layer,
            "best_single_rho": float(best_rho),
            "combined_top5_rho": float(abs(combined_rho)),
            "combined_top5_r": float(combined_r),
            "cross_cos_ep": cross_cos_ep,
            "cos_with_v1_p": cos_with_v1,
            "extraction_time": extract_time,
            "top5_layers": {str(l): r for l, r in top5},
        }
        results[name] = result

        print(f"  Best single layer: {best_layer} (rho={best_rho:.4f})")
        print(f"  Combined top-5:    rho={abs(combined_rho):.4f}, r={combined_r:.4f}")
        if cross_cos_ep is not None:
            print(f"  Cross-dim |cos(E,P)|: {cross_cos_ep:.4f}")
        if cos_with_v1 is not None:
            print(f"  cos(v1_P, new_P):    {cos_with_v1:.4f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("=== RANKING (by combined top-5 Spearman rho) ===\n")
    ranked = sorted(results.items(), key=lambda x: x[1]["combined_top5_rho"], reverse=True)

    print(f"{'Rank':>4} | {'Template':>30} | {'rho_top5':>8} | {'rho_best':>8} | {'cos(E,P)':>8} | {'cos(v1)':>8}")
    print("-" * 95)
    for i, (name, r) in enumerate(ranked):
        marker = " ***" if i < args.top_k else ""
        cos_ep = f"{r['cross_cos_ep']:.4f}" if r["cross_cos_ep"] is not None else "N/A"
        cos_v1 = f"{r['cos_with_v1_p']:.4f}" if r["cos_with_v1_p"] is not None else "N/A"
        print(f"{i+1:4d} | {name:>30} | {r['combined_top5_rho']:8.4f} | "
              f"{r['best_single_rho']:8.4f} | {cos_ep:>8} | {cos_v1:>8}{marker}")

    # Save results
    output_path = results_dir / "potency_template_search.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_candidates": len(results),
            "n_test_items": len(test_items),
            "results": results,
            "ranking": [name for name, _ in ranked],
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Recommend
    best_name = ranked[0][0]
    best = ranked[0][1]
    print(f"\n=== RECOMMENDATION ===")
    print(f"Best template: {best_name}")
    print(f"  Combined rho: {best['combined_top5_rho']:.4f}")
    print(f"  Template: {best['template_pos']}")
    print(f"  vs:       {best['template_neg']}")


if __name__ == "__main__":
    main()
