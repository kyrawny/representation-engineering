"""
Model Registry — model-agnostic configuration and prompt formatting.

Provides:
- A registry of known models with their short names and metadata.
- ``format_chat_prompt()`` — model-agnostic prompt formatting using
  HuggingFace's ``tokenizer.apply_chat_template()``.
- ``get_short_name()`` — returns a filesystem-safe short name for a model.
- ``get_model_config()`` — returns the registry entry for a model.

Unknown models are handled gracefully by deriving a slug from the model
name and using the tokenizer's built-in chat template.
"""

import re
from typing import Any, Dict, Optional


# =========================================================================
# Known model registry
# =========================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "short_name": "llama3",
        "num_hidden_layers": 32,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "short_name": "ministral",
        "num_hidden_layers": 36,
    },
}


def _slugify(model_name: str) -> str:
    """Convert a HuggingFace model name to a filesystem-safe slug.

    Examples::

        >>> _slugify("meta-llama/Llama-3.1-8B-Instruct")
        'llama-3.1-8b-instruct'
        >>> _slugify("mistralai/Ministral-8B-Instruct-2412")
        'ministral-8b-instruct-2412'
    """
    # Take the part after the last '/'
    name = model_name.rsplit("/", 1)[-1]
    # Lowercase and replace non-alphanumeric chars with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def get_short_name(model_name: str) -> str:
    """Return the short name for a model (used for directory names).

    If the model is in the registry, returns its registered short name.
    Otherwise, derives a slug from the model name.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Short string suitable for use in file paths.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]["short_name"]
    return _slugify(model_name)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Return the configuration dict for a model.

    Returns the registry entry if the model is known, or a default
    config derived from the model name.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Dict with at least ``'short_name'`` key.
    """
    if model_name in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[model_name])
    return {
        "short_name": _slugify(model_name),
    }


# =========================================================================
# Model-agnostic prompt formatting
# =========================================================================

def format_chat_prompt(
    tokenizer,
    system_prompt: Optional[str],
    user_input: str,
    assistant_start: str = "",
) -> str:
    """Format a chat prompt using the tokenizer's built-in chat template.

    This is the model-agnostic replacement for ``format_llama3_prompt()``.
    It uses HuggingFace's ``tokenizer.apply_chat_template()`` to produce
    the correct special tokens for any supported model.

    Args:
        tokenizer: HuggingFace tokenizer with a chat template.
        system_prompt: System prompt content, or None/\"\" to skip.
        user_input: User message content.
        assistant_start: Optional text to prepend to the assistant turn
            (e.g. a truncated response for contrastive extraction).

    Returns:
        Formatted prompt string with model-appropriate special tokens.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if assistant_start:
        prompt += assistant_start

    return prompt


def format_for_reading_generic(
    tokenizer,
    text: str,
    neutral_context: str = "Pretend you're a person making a statement in a conversation.",
) -> str:
    """Format text for EPA reading using the tokenizer's chat template.

    Places *text* in the assistant position after a neutral user message.
    No system prompt — matches the training format used during direction
    extraction.

    Args:
        tokenizer: HuggingFace tokenizer with a chat template.
        text: The text to read EPA values from.
        neutral_context: Neutral user message to provide context.

    Returns:
        Formatted prompt with text in assistant position.
    """
    return format_chat_prompt(tokenizer, None, neutral_context, text)
