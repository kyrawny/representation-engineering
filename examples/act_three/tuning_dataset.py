"""
Tuning Dataset Generation.

Generates utterances for EPA calibration using an LLM and assigns Likert-scale
ground-truth labels for every EPA dimension based on ACT dictionary values.

Provides:
- ``epa_value_to_likert()`` — continuous EPA → 7-point Likert label
- ``epa_to_description()`` — EPA values → natural language description
- ``stratified_behavior_split()`` — train/test split with EPA-space coverage
- ``generate_tuning_dataset()`` — full dataset generation orchestrator
- ``save_dataset()`` / ``load_dataset()`` — JSON I/O
"""

import json
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .act_core import EPA
from .epa_calibration import BehaviorPromptGenerator
from .model_registry import format_chat_prompt

# ---------------------------------------------------------------------------
# Likert scale mapping
# ---------------------------------------------------------------------------

LIKERT_OPTIONS = [
    "Very low",
    "Low",
    "Somewhat low",
    "Neutral",
    "Somewhat high",
    "High",
    "Very high",
]

LIKERT_VALUES = {label: i + 1 for i, label in enumerate(LIKERT_OPTIONS)}


def epa_value_to_likert(value: float) -> str:
    """Map a continuous EPA value to a 7-point Likert label."""
    if value <= -3.0:
        return "Very low"
    elif value <= -1.5:
        return "Low"
    elif value <= -0.5:
        return "Somewhat low"
    elif value <= 0.5:
        return "Neutral"
    elif value <= 1.5:
        return "Somewhat high"
    elif value <= 3.0:
        return "High"
    else:
        return "Very high"


# ---------------------------------------------------------------------------
# Natural language description of EPA values
# ---------------------------------------------------------------------------

def epa_to_description(epa: EPA) -> str:
    """
    Convert EPA values to a human-readable description.

    Used as part of the LLM prompt when generating utterances with a
    specific affective tone.
    """

    def _scale_term(value: float, neg_terms: List[str], pos_terms: List[str]) -> str:
        if value <= -3.0:
            return neg_terms[0]
        elif value <= -1.5:
            return neg_terms[1]
        elif value <= -0.5:
            return neg_terms[2]
        elif value <= 0.5:
            return neg_terms[3]
        elif value <= 1.5:
            return pos_terms[0]
        elif value <= 3.0:
            return pos_terms[1]
        else:
            return pos_terms[2]

    e_desc = _scale_term(
        epa.e,
        ["very bad", "bad", "somewhat bad", "neither good nor bad"],
        ["somewhat good", "good", "very good"],
    )
    p_desc = _scale_term(
        epa.p,
        ["very impotent", "impotent", "somewhat impotent", "neither potent nor impotent"],
        ["somewhat potent", "potent", "very potent"],
    )
    a_desc = _scale_term(
        epa.a,
        ["very passive", "passive", "somewhat passive", "neither active nor passive"],
        ["somewhat active", "active", "very active"],
    )
    return f"{e_desc}, {p_desc}, and {a_desc}"


# ---------------------------------------------------------------------------
# LLM utterance generation
# ---------------------------------------------------------------------------

def make_utterance_generator(
    model,
    tokenizer,
) -> Callable[[str, EPA], str]:
    """
    Create a function that generates a single conversational utterance for a
    given behavior and EPA using the provided *model* and *tokenizer*.

    The returned callable has signature ``(behavior: str, epa: EPA) -> str``
    and can be passed to ``BehaviorPromptGenerator.set_llm_generate_function()``.
    """

    def _generate(behavior: str, epa: EPA) -> str:
        epa_description = epa_to_description(epa)
        behavior_display = behavior.replace("_", " ")
        prompt = format_chat_prompt(
            tokenizer,
            None,
            (
                "You are simulating a human person engaging in a conversation with "
                "another human person. Generate a single conversational utterance that "
                "when observed by a third party, is most appropriately described as "
                f"'you {behavior_display} them' towards the other person, where "
                f"'{behavior_display}' is a verb or action, with an affective tone "
                f"that is {epa_description}. Respond with only the utterance, no "
                "explanation or quotation marks."
            ),
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip().strip("\"'")

    return _generate


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def stratified_behavior_split(
    behaviors: List[str],
    generator: BehaviorPromptGenerator,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split behaviours into train/test sets ensuring both cover the full EPA
    space.  Uses k-means clustering on EPA vectors to achieve proportional
    allocation from each cluster.

    Args:
        behaviors: List of behaviour names.
        generator: ``BehaviorPromptGenerator`` with EPA lookup.
        test_fraction: Fraction of behaviours held out for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_behaviors, test_behaviors).
    """
    from sklearn.cluster import KMeans

    rng = random.Random(seed)

    epa_vecs, valid_behaviors = [], []
    for b in behaviors:
        epa = generator.get_behavior_epa(b)
        if epa is not None:
            epa_vecs.append([epa.e, epa.p, epa.a])
            valid_behaviors.append(b)

    epa_arr = np.array(epa_vecs)
    n_clusters = min(10, len(valid_behaviors))

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(epa_arr)

    clusters: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(valid_behaviors[idx])

    train_behaviors: List[str] = []
    test_behaviors: List[str] = []
    for _cluster_id, members in clusters.items():
        rng.shuffle(members)
        n_take = max(1, round(len(members) * test_fraction))
        test_behaviors.extend(members[:n_take])
        train_behaviors.extend(members[n_take:])

    return train_behaviors, test_behaviors


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_tuning_dataset(
    model,
    tokenizer,
    model_name: str,
    n_variants: int = 5,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate a calibration dataset of (utterance, target EPA) pairs.

    For each behaviour from the ACT dictionary, generates *n_variants*
    utterances using the language model, and labels each with ground-truth
    EPA and Likert targets from the dictionary.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        model_name: Name of the model (for metadata).
        n_variants: Number of utterance variants per behaviour.
        test_fraction: Fraction of behaviours for the test split.
        seed: Random seed.

    Returns:
        Tuple of (train_utterances, test_utterances) where each entry is a
        list of dicts with keys ``text``, ``behavior``, ``variant``,
        ``target_epa``, ``likert_targets``.
    """
    from tqdm.auto import tqdm

    generate_fn = make_utterance_generator(model, tokenizer)

    generator = BehaviorPromptGenerator()
    generator.set_llm_generate_function(generate_fn)

    behaviors = generator.available_behaviors
    train_behaviors, test_behaviors = stratified_behavior_split(
        behaviors, generator, test_fraction=test_fraction, seed=seed,
    )

    def _gen_utterances(behavior_list: List[str], desc: str) -> List[Dict]:
        utterances: List[Dict] = []
        for behavior in tqdm(behavior_list, desc=f"Generating {desc}"):
            epa = generator.get_behavior_epa(behavior)
            if epa is None:
                continue
            for v in range(n_variants):
                text = generator.generate_utterance(behavior, variant=v)
                if text is None or text.startswith("["):
                    continue
                utterances.append(
                    {
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
                    }
                )
        return utterances

    train_utterances = _gen_utterances(train_behaviors, "train")
    test_utterances = _gen_utterances(test_behaviors, "test")

    return train_utterances, test_utterances


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_dataset(
    path: str,
    utterances: List[Dict],
    split_name: str,
    model_name: str,
    n_behaviors: int,
    n_variants: int,
) -> None:
    """Save a tuning dataset split to JSON."""
    dataset = {
        "metadata": {
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "split": split_name,
            "n_behaviors": n_behaviors,
            "n_variants": n_variants,
            "n_utterances": len(utterances),
        },
        "utterances": utterances,
    }
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_dataset(path: str) -> Dict:
    """Load a tuning dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)
