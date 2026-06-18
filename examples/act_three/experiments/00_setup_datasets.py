"""
Experiment 00 — Dataset Setup.

Generates the calibration train/test split datasets required by
the experiment suite.  Supports two modes:

    1. ``--mode llm``  (default)
       Uses the Llama model to generate diverse utterances for each
       ACT behavior.  Requires GPU.  Produces high-quality data
       matching the original tuning pipeline.

    2. ``--mode template``
       Uses simple template sentences (no GPU needed).  Each utterance
       is "A person who {behavior} another person."  Good enough for
       testing the experiment pipeline, but the reading quality metrics
       will differ from the original because the utterances are generic.

Both modes produce files in the format expected by the experiment suite::

    {
      "metadata": { ... },
      "utterances": [
        {
          "text": "...",
          "behavior": "...",
          "variant": 0,
          "target_epa": {"e": 1.23, "p": -0.45, "a": 0.67},
          "likert_targets": {"evaluation": "High", ...}
        },
        ...
      ]
    }

Usage::

    # Full LLM generation (GPU required, ~30 min)
    python -m examples.act_three.experiments.00_setup_datasets --mode llm

    # Quick template mode (no GPU, instant)
    python -m examples.act_three.experiments.00_setup_datasets --mode template

    # Custom split / variant count
    python -m examples.act_three.experiments.00_setup_datasets --mode template \\
        --n-variants 3 --test-fraction 0.1
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Resolve imports relative to repository root
_THIS_DIR = Path(__file__).resolve().parent
_ACT_THREE_DIR = _THIS_DIR.parent
_REPO_ROOT = _ACT_THREE_DIR.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.act_three.act_core import EPA
from examples.act_three.epa_calibration import (
    BehaviorPromptGenerator,
    CONVERSATIONAL_BEHAVIORS,
)
from examples.act_three.tuning_dataset import (
    epa_value_to_likert,
    epa_to_description,
    stratified_behavior_split,
    save_dataset,
)


# =========================================================================
# Output paths (in act_three/, matching the metadata in tuning results)
# =========================================================================

TRAIN_PATH = _ACT_THREE_DIR / "epa_tuning_dataset_train.json"
TEST_PATH = _ACT_THREE_DIR / "epa_tuning_dataset_test.json"


# =========================================================================
# Template-based generation (no GPU)
# =========================================================================

# Multiple template patterns for variety
_TEMPLATES = [
    "A person who {verb} another person.",
    "Someone who {verb} the other person in a conversation.",
    "In a discussion, one person {verb} the other.",
    "During a meeting, one participant {verb} another participant.",
    "A colleague who {verb} a coworker in a professional setting.",
]


def _verb_form(behavior: str) -> str:
    """Convert behavior name to a verb phrase.

    Handles underscore-separated names and common patterns.
    """
    verb = behavior.replace("_", " ")
    # Handle "something" / "ones" placeholders
    verb = verb.replace("something ", "")
    verb = verb.replace("ones ", "one's ")
    return verb


def generate_template_utterances(
    behaviors: List[str],
    generator: BehaviorPromptGenerator,
    n_variants: int = 5,
) -> List[Dict]:
    """Generate utterances using simple sentence templates.

    Each behavior × variant combination gets a different template
    pattern, producing *n_variants* stylistically varied utterances
    per behavior while still carrying the correct ground-truth EPA
    from the ACT dictionary.
    """
    utterances: List[Dict] = []
    for behavior in behaviors:
        epa = generator.get_behavior_epa(behavior)
        if epa is None:
            continue
        verb = _verb_form(behavior)
        for v in range(n_variants):
            template = _TEMPLATES[v % len(_TEMPLATES)]
            text = template.format(verb=verb)
            utterances.append({
                "text": text,
                "behavior": behavior,
                "variant": v,
                "target_epa": {
                    "e": float(epa.e),
                    "p": float(epa.p),
                    "a": float(epa.a),
                },
                "likert_targets": {
                    "evaluation": epa_value_to_likert(epa.e),
                    "potency": epa_value_to_likert(epa.p),
                    "activity": epa_value_to_likert(epa.a),
                },
            })
    return utterances


# =========================================================================
# LLM-based generation (GPU required)
# =========================================================================

def generate_llm_utterances(
    behaviors: List[str],
    generator: BehaviorPromptGenerator,
    n_variants: int = 5,
) -> List[Dict]:
    """Generate utterances using the Llama model.

    Requires the model to be loaded — imports torch and transformers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from examples.act_three.tuning_dataset import make_utterance_generator

    from .config import MODEL_NAME

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    generate_fn = make_utterance_generator(model, tokenizer)
    generator.set_llm_generate_function(generate_fn)

    utterances: List[Dict] = []
    from tqdm import tqdm

    for behavior in tqdm(behaviors, desc="Generating utterances"):
        epa = generator.get_behavior_epa(behavior)
        if epa is None:
            continue
        for v in range(n_variants):
            text = generator.generate_utterance(behavior, variant=v)
            if text is None or text.startswith("["):
                continue
            utterances.append({
                "text": text,
                "behavior": behavior,
                "variant": v,
                "target_epa": {
                    "e": float(epa.e),
                    "p": float(epa.p),
                    "a": float(epa.a),
                },
                "likert_targets": {
                    "evaluation": epa_value_to_likert(epa.e),
                    "potency": epa_value_to_likert(epa.p),
                    "activity": epa_value_to_likert(epa.a),
                },
            })

    # Free model memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return utterances


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 00: Generate calibration train/test datasets")
    parser.add_argument(
        "--mode", default="template", choices=["template", "llm"],
        help="Generation mode: 'template' (no GPU) or 'llm' (GPU required)")
    parser.add_argument(
        "--n-variants", type=int, default=5,
        help="Number of utterance variants per behavior (default: 5)")
    parser.add_argument(
        "--test-fraction", type=float, default=0.10,
        help="Fraction of behaviors held out for testing (default: 0.10)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing dataset files")
    args = parser.parse_args()

    # Check for existing files
    if not args.force and TRAIN_PATH.exists() and TEST_PATH.exists():
        print(f"Dataset files already exist:")
        print(f"  Train: {TRAIN_PATH}")
        print(f"  Test:  {TEST_PATH}")
        print("Use --force to overwrite.")
        return

    print(f"Mode:          {args.mode}")
    print(f"Variants:      {args.n_variants}")
    print(f"Test fraction: {args.test_fraction}")
    print(f"Seed:          {args.seed}")
    print()

    # ---- Initialize behavior generator ----
    generator = BehaviorPromptGenerator(filter_conversational=True)
    all_behaviors = generator.available_behaviors
    print(f"Available conversational behaviors: {len(all_behaviors)}")

    # ---- Train / test split on behaviors ----
    train_behaviors, test_behaviors = stratified_behavior_split(
        all_behaviors, generator,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Train behaviors: {len(train_behaviors)}")
    print(f"Test behaviors:  {len(test_behaviors)}")

    # ---- Generate utterances ----
    if args.mode == "template":
        print("\nGenerating template-based utterances...")
        train_utterances = generate_template_utterances(
            train_behaviors, generator, n_variants=args.n_variants)
        test_utterances = generate_template_utterances(
            test_behaviors, generator, n_variants=args.n_variants)
    else:
        print("\nGenerating LLM-based utterances (this may take a while)...")
        train_utterances = generate_llm_utterances(
            train_behaviors, generator, n_variants=args.n_variants)
        test_utterances = generate_llm_utterances(
            test_behaviors, generator, n_variants=args.n_variants)

    print(f"\nTrain utterances: {len(train_utterances)}")
    print(f"Test utterances:  {len(test_utterances)}")

    # ---- Print EPA distribution summary ----
    for split_name, utterances in [("Train", train_utterances),
                                    ("Test", test_utterances)]:
        epas = np.array([[u["target_epa"]["e"], u["target_epa"]["p"],
                          u["target_epa"]["a"]] for u in utterances])
        print(f"\n{split_name} EPA distribution:")
        for i, dim in enumerate(["E", "P", "A"]):
            print(f"  {dim}: mean={epas[:, i].mean():.2f}, "
                  f"std={epas[:, i].std():.2f}, "
                  f"range=[{epas[:, i].min():.2f}, {epas[:, i].max():.2f}]")

    # ---- Save ----
    model_name = "meta-llama/Llama-3.1-8B-Instruct" if args.mode == "llm" \
        else "template-based (no model)"

    save_dataset(
        str(TRAIN_PATH), train_utterances, "train",
        model_name=model_name,
        n_behaviors=len(train_behaviors),
        n_variants=args.n_variants,
    )
    print(f"\nSaved train set: {TRAIN_PATH}")

    save_dataset(
        str(TEST_PATH), test_utterances, "test",
        model_name=model_name,
        n_behaviors=len(test_behaviors),
        n_variants=args.n_variants,
    )
    print(f"Saved test set:  {TEST_PATH}")

    # ---- Verify ----
    print("\n--- Verification ---")
    for path in [TRAIN_PATH, TEST_PATH]:
        with open(path, "r") as f:
            data = json.load(f)
        meta = data["metadata"]
        n = len(data["utterances"])
        sample = data["utterances"][0]
        print(f"  {path.name}: {n} utterances, "
              f"fields: {list(sample.keys())}")
        assert "text" in sample, f"Missing 'text' field in {path.name}"
        assert "target_epa" in sample, f"Missing 'target_epa' field in {path.name}"
        assert all(k in sample["target_epa"] for k in ["e", "p", "a"]), \
            f"Missing EPA keys in {path.name}"
    print("  All checks passed ✓")


if __name__ == "__main__":
    main()
