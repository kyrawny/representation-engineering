"""
Dataset Creation for EPA Direction Extraction.

Provides functions to create contrastive datasets for each EPA dimension using
minimal user-tag-only prompts following the RepE honesty pattern.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

from .prompt_formatting import (
    format_llama3_prompt,
    get_epa_extraction_template,
)


def load_act_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load ACT training data from data directory.

    Args:
        data_dir: Path to ``data/act`` directory.

    Returns:
        Tuple of (user_inputs, truncated_outputs).
    """
    with open(os.path.join(data_dir, "user_inputs.json"), "r") as f:
        user_inputs = json.load(f)

    with open(os.path.join(data_dir, "all_truncated_outputs.json"), "r") as f:
        truncated_outputs = json.load(f)

    # Filter out empty or very short truncated outputs
    truncated_outputs = [t for t in truncated_outputs if len(t) >= 2]

    return user_inputs, truncated_outputs


def create_epa_dataset(
    data_dir: str,
    dimension: str,
    n_train: int = 256,
    seed: int = 42,
) -> Dict:
    """
    Create a contrastive dataset for a specific EPA dimension.

    Uses minimal user-tag-only prompts following the RepE honesty pattern.
    No system prompt is used — the contrastive instruction is embedded
    directly in the user message as a single short sentence.

    Args:
        data_dir: Path to ``data/act`` directory.
        dimension: One of ``'evaluation'``, ``'potency'``, ``'activity'``.
        n_train: Number of training pairs.
        seed: Random seed.

    Returns:
        Dict with ``'train': {'data': List[str], 'labels': List[List[bool]]}``.
    """
    random.seed(seed)
    np.random.seed(seed)

    _user_inputs, truncated_outputs = load_act_data(data_dir)

    # Get minimal extraction templates (no system prompt)
    template_pos = get_epa_extraction_template(dimension, "positive")
    template_neg = get_epa_extraction_template(dimension, "negative")

    train_data: List[str] = []
    train_labels: List[List[bool]] = []

    # Create contrastive pairs
    for _i in range(n_train):
        truncated = random.choice(truncated_outputs)

        # Create positive and negative prompts with same truncated output.
        # No system prompt — the template IS the user message.
        pos_prompt = format_llama3_prompt(None, template_pos, truncated)
        neg_prompt = format_llama3_prompt(None, template_neg, truncated)

        # Shuffle for balanced labels
        pair = [pos_prompt, neg_prompt]
        random.shuffle(pair)

        train_labels.append([pair[0] == pos_prompt, pair[1] == pos_prompt])
        train_data.extend(pair)

    return {"train": {"data": train_data, "labels": train_labels}}


def create_all_epa_datasets(
    data_dir: str,
    n_train: int = 256,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Create datasets for all three EPA dimensions.

    Returns:
        Dict mapping dimension name to dataset dict.
    """
    return {
        dim: create_epa_dataset(data_dir, dim, n_train, seed)
        for dim in ["evaluation", "potency", "activity"]
    }
