"""
Prompt Formatting for Llama 3.1 Instruct models.

Provides:
- Llama 3.1 prompt token constants
- EPA dimension definitions and extraction adjectives
- Prompt formatting for system/user/assistant turns
- Formatting helpers for EPA reading and extraction
"""

from typing import Optional, Dict, List, Tuple

# =============================================================================
# Llama 3.1 Instruct Prompt Tokens
# =============================================================================

LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_SYSTEM_START = "<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA3_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"


# =============================================================================
# EPA Dimension Definitions
# =============================================================================

EPA_DIMENSIONS = {
    "evaluation": {
        "positive": "good",
        "negative": "bad",
        "positive_terms": ["good", "moral", "altruistic", "kind", "virtuous"],
        "negative_terms": ["bad", "immoral", "selfish", "cruel", "wicked"],
        "neutral_dims": [("potent", "impotent"), ("active", "inactive")],
        "description": "morality, altruism, and social desirability",
    },
    "potency": {
        "positive": "potent",
        "negative": "impotent",
        "positive_terms": ["commanding", "authoritative", "dominant", "assertive",
                           "forceful", "imposing", "decisive"],
        "negative_terms": ["submissive", "meek", "deferential", "timid",
                           "yielding", "compliant", "hesitant"],
        "neutral_dims": [("good", "bad"), ("active", "inactive")],
        "description": "power, authority, dominance, and strength",
    },
    "activity": {
        "positive": "active",
        "negative": "inactive",
        "positive_terms": ["active", "energetic", "lively", "dynamic", "animated"],
        "negative_terms": ["inactive", "lethargic", "sluggish", "passive", "calm"],
        "neutral_dims": [("good", "bad"), ("potent", "impotent")],
        "description": "energy level, speed, volatility, and liveliness",
    },
}

# Simple extraction template adjectives (used in get_epa_extraction_template).
# Map EPA dimensions to natural-sounding adjective pairs for the
# minimal "Pretend you're a {adj} person" template.
_EXTRACTION_ADJECTIVES = {
    "evaluation": {"positive": "a good", "negative": "a bad"},
    "potency": {"positive": "an authoritative and commanding",
                "negative": "a meek and submissive"},
    "activity": {"positive": "a lively", "negative": "a quiet"},
}

# Dimension abbreviation keys
DIMENSION_NAMES = ["evaluation", "potency", "activity"]
DIM_KEY = {"evaluation": "e", "potency": "p", "activity": "a"}


# =============================================================================
# Prompt Formatting Functions
# =============================================================================

def format_llama3_prompt(
    system_prompt: Optional[str],
    user_input: str,
    assistant_start: str = "",
    include_bos: bool = True,
) -> str:
    """
    Format a complete Llama 3.1 Instruct prompt.

    System prompt is optional — pass None or "" to omit the system block
    entirely. This is important for extraction prompts, which should use
    user-tag-only format.

    Args:
        system_prompt: System prompt content, or None/"" to skip.
        user_input: User message content.
        assistant_start: Truncated assistant response start.
        include_bos: Whether to include BOS token.

    Returns:
        Formatted prompt string.
    """
    parts = []
    if include_bos:
        parts.append(LLAMA3_BOS)

    # Only include system block if system_prompt is provided and non-empty
    if system_prompt:
        parts.append(LLAMA3_SYSTEM_START)
        parts.append(system_prompt)
        parts.append(LLAMA3_EOT)

    parts.append(LLAMA3_USER_START)
    parts.append(user_input)
    parts.append(LLAMA3_EOT)
    parts.append(LLAMA3_ASSISTANT_START)
    parts.append(assistant_start)

    return "".join(parts)


def format_for_reading(
    text: str,
    neutral_context: str = "Pretend you're a person making a statement in a conversation.",
) -> str:
    """
    Format text as if it were an assistant response for EPA reading.

    Places *text* in the assistant position after a neutral user message.
    No system prompt — matches the training format used in
    ``create_epa_dataset()``, where directions were extracted with
    ``format_llama3_prompt(None, template, text)``.

    Args:
        text: The text to read EPA values from.
        neutral_context: Neutral user message to provide context.

    Returns:
        Formatted prompt with text in assistant position.
    """
    return format_llama3_prompt(None, neutral_context, text)


def get_epa_extraction_template(dimension: str, target: str) -> str:
    """
    Generate a minimal extraction template for an EPA dimension.

    Follows the RepE honesty pattern: a single short sentence where the only
    semantic difference between positive and negative is the target adjective.

    Example::

        >>> get_epa_extraction_template("evaluation", "positive")
        "Pretend you're a good person making a statement in a conversation."

    Args:
        dimension: One of ``'evaluation'``, ``'potency'``, ``'activity'``.
        target: Either ``'positive'`` or ``'negative'``.

    Returns:
        Minimal template string (used as user message, no system prompt).
    """
    adj = _EXTRACTION_ADJECTIVES[dimension][target]
    return f"Pretend you're {adj} person making a statement in a conversation."
