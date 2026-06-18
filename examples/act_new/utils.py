"""
Utility functions for Affect Control Theory (ACT) EPA vector extraction.

This module provides:
- Dataset creation for E, P, A contrastive prompts (minimal format)
- Visualization functions for t-SNE, LAT scans, and per-token detection
- Helper functions for Llama 3.1 prompt formatting
- EPA reading and steering utilities

This is an independent, simplified version with fixes for EPA steering
overexaggeration. Key differences from the original:
1. Minimal extraction prompts matching the RepE honesty pattern
2. No system prompt in extraction (user-tag-only format)
3. L2-normalized direction vectors in make_epa_activations
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

# Llama 3.1 Instruct prompt tokens
LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_SYSTEM_START = "<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA3_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"

# EPA dimension definitions
EPA_DIMENSIONS = {
    "evaluation": {
        "positive": "good",
        "negative": "bad",
        "positive_terms": ["good", "moral", "altruistic", "kind", "virtuous"],
        "negative_terms": ["bad", "immoral", "selfish", "cruel", "wicked"],
        "neutral_dims": [("potent", "impotent"), ("active", "inactive")],
        "description": "morality, altruism, and social desirability"
    },
    "potency": {
        "positive": "potent",
        "negative": "impotent",
        "positive_terms": ["potent", "powerful", "dominant", "authoritative", "strong"],
        "negative_terms": ["impotent", "weak", "submissive", "powerless", "meek"],
        "neutral_dims": [("good", "bad"), ("active", "inactive")],
        "description": "power, authority, dominance, and strength"
    },
    "activity": {
        "positive": "active",
        "negative": "inactive",
        "positive_terms": ["active", "energetic", "lively", "dynamic", "animated"],
        "negative_terms": ["inactive", "lethargic", "sluggish", "passive", "calm"],
        "neutral_dims": [("good", "bad"), ("potent", "impotent")],
        "description": "energy level, speed, volatility, and liveliness"
    }
}

# Simple extraction template adjectives (used in get_epa_extraction_template)
# These map EPA dimensions to natural-sounding adjective pairs for the
# minimal "Pretend you're a {adj} person" template.
_EXTRACTION_ADJECTIVES = {
    "evaluation": {"positive": "a good", "negative": "a bad"},
    "potency": {"positive": "a powerful", "negative": "a weak"},
    "activity": {"positive": "a lively", "negative": "a quiet"},
}


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_llama3_prompt(
    system_prompt: Optional[str],
    user_input: str,
    assistant_start: str = "",
    include_bos: bool = True
) -> str:
    """
    Format a complete Llama 3.1 Instruct prompt.
    
    System prompt is optional — pass None or "" to omit the system block entirely.
    This is important for extraction prompts, which should use user-tag-only format.
    
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


def format_for_reading(text: str, neutral_context: str = "Pretend you're a person making a statement in a conversation.") -> str:
    """
    Format text as if it were an assistant response for EPA reading.
    
    Args:
        text: The text to read EPA values from.
        neutral_context: Neutral user message to provide context.
    
    Returns:
        Formatted prompt with text in assistant position.
    """
    # No system prompt — matches the training format used in create_epa_dataset(),
    # where directions were extracted with format_llama3_prompt(None, template, text).
    return format_llama3_prompt(None, neutral_context, text)


# =============================================================================
# Extraction Template (Minimal, following RepE honesty pattern)
# =============================================================================

def get_epa_extraction_template(dimension: str, target: str) -> str:
    """
    Generate a minimal extraction template for an EPA dimension.
    
    Follows the RepE honesty pattern: a single short sentence where the only
    semantic difference between positive and negative is the target adjective.
    
    Example output:
        "Pretend you're a good person making a statement in a conversation."
    
    Args:
        dimension: One of 'evaluation', 'potency', 'activity'
        target: Either 'positive' or 'negative'
    
    Returns:
        Minimal template string (used as user message, no system prompt).
    """
    adj = _EXTRACTION_ADJECTIVES[dimension][target]
    return f"Pretend you're {adj} person making a statement in a conversation."


# =============================================================================
# Dataset Creation
# =============================================================================

def load_act_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load ACT training data from data directory.
    
    Args:
        data_dir: Path to data/act directory.
    
    Returns:
        Tuple of (user_inputs, truncated_outputs)
    """
    with open(os.path.join(data_dir, "user_inputs.json"), 'r') as f:
        user_inputs = json.load(f)
    
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as f:
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
    Create contrastive dataset for a specific EPA dimension.
    
    Uses minimal user-tag-only prompts following the RepE honesty pattern.
    No system prompt is used — the contrastive instruction is embedded directly
    in the user message as a single short sentence.
    
    Args:
        data_dir: Path to data/act directory.
        dimension: One of 'evaluation', 'potency', 'activity'.
        n_train: Number of training pairs.
        seed: Random seed.
    
    Returns:
        Dict with 'train': {'data': List[str], 'labels': List[List[bool]]}
    """
    random.seed(seed)
    np.random.seed(seed)
    
    _user_inputs, truncated_outputs = load_act_data(data_dir)
    
    # Get minimal extraction templates (no system prompt)
    template_pos = get_epa_extraction_template(dimension, "positive")
    template_neg = get_epa_extraction_template(dimension, "negative")
    
    train_data = []
    train_labels = []
    
    # Create contrastive pairs
    for i in range(n_train):
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
    
    return {
        'train': {'data': train_data, 'labels': train_labels}
    }


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


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_tsne_epa(
    hidden_states_pos: np.ndarray,
    hidden_states_neg: np.ndarray,
    dimension: str,
    layer: int,
    perplexity: int = 30,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Create t-SNE visualization for EPA dimension (Figure 14 style).
    
    Args:
        hidden_states_pos: Hidden states for positive class (n_samples, hidden_dim)
        hidden_states_neg: Hidden states for negative class (n_samples, hidden_dim)
        dimension: EPA dimension name
        layer: Layer number for title
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib and sklearn required for plotting")
        return None
    
    dim_info = EPA_DIMENSIONS[dimension]
    
    # Combine data
    all_states = np.vstack([hidden_states_pos, hidden_states_neg])
    labels = np.array([1] * len(hidden_states_pos) + [0] * len(hidden_states_neg))
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(all_states)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    colors = ['#e74c3c', '#27ae60']  # Red for negative, Green for positive
    
    for label, color, name in [(0, colors[0], dim_info['negative']), 
                                (1, colors[1], dim_info['positive'])]:
        mask = labels == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                  c=color, label=name.capitalize(), alpha=0.6, s=30)
    
    ax.set_title(f"t-SNE: {dimension.capitalize()} (Layer {layer})")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    plt.tight_layout()
    return fig


def plot_lat_scan(
    scores_by_layer: Dict[int, np.ndarray],
    tokens: List[str],
    dimension: str,
    start_idx: int = 0,
    n_tokens: int = 40,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Create LAT scan heatmap visualization (Figure 8 style).
    
    Args:
        scores_by_layer: Dict mapping layer index to per-token scores
        tokens: List of token strings
        dimension: EPA dimension name
        start_idx: Starting token index
        n_tokens: Number of tokens to show
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return None
    
    layers = sorted(scores_by_layer.keys())
    
    # Build matrix (layers x tokens)
    matrix = []
    for layer in layers:
        scores = scores_by_layer[layer]
        if len(scores.shape) > 1:
            scores = scores.flatten()
        layer_scores = scores[start_idx:start_idx + n_tokens]
        matrix.append(layer_scores)
    
    matrix = np.array(matrix)
    
    # Normalize
    bound = np.percentile(np.abs(matrix), 95)
    matrix = np.clip(matrix, -bound, bound)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    sns.heatmap(matrix, cmap='coolwarm', center=0, 
                vmin=-bound, vmax=bound, ax=ax,
                xticklabels=5, yticklabels=5)
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(f"LAT Scan: {dimension.capitalize()}")
    
    # Set y-axis labels
    y_labels = [str(layers[i]) for i in range(0, len(layers), max(1, len(layers)//10))]
    
    plt.tight_layout()
    return fig


def plot_per_token_detection(
    tokens: List[str],
    scores: np.ndarray,
    dimension: str,
    threshold: float = 0.0,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Visualize per-token EPA intensity with colored text (Figure 9 style).
    
    Args:
        tokens: List of token strings
        scores: Per-token scores
        dimension: EPA dimension name
        threshold: Normalization threshold
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, LinearSegmentedColormap
    except ImportError:
        print("matplotlib required for plotting")
        return None
    
    dim_info = EPA_DIMENSIONS[dimension]
    
    # Clean tokens
    clean_tokens = []
    for token in tokens:
        t = token.replace('▁', ' ').replace('Ġ', ' ').replace('Ċ', '\n')
        t = t.replace('<|', '').replace('|>', '')  # Remove special tokens
        clean_tokens.append(t)
    
    # Normalize scores
    scores = np.array(scores) - threshold
    scores = scores / (np.std(scores) + 1e-8)
    mag = max(0.5, np.percentile(np.abs(scores), 90))
    scores = np.clip(scores, -mag, mag)
    
    # Create colormap (red for negative, green for positive)
    cmap = LinearSegmentedColormap.from_list('epa', 
        ['#e74c3c', '#f5f5dc', '#27ae60'], N=256)
    norm = Normalize(vmin=-mag, vmax=mag)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    x, y = 10, 9
    max_x = 990
    
    for i, (token, score) in enumerate(zip(clean_tokens, scores)):
        if not token.strip():
            continue
            
        color = cmap(norm(score))
        
        text = ax.text(x, y, token, fontsize=10,
                      bbox=dict(facecolor=color, edgecolor='none', 
                               alpha=0.8, pad=1))
        
        # Get text width
        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        text_width = bbox.width
        
        x += text_width + 3
        if x > max_x:
            x = 10
            y -= 1.2
            if y < 1:
                break
    
    ax.set_title(f"Per-Token {dimension.capitalize()} Detection\n"
                f"(Green = {dim_info['positive']}, Red = {dim_info['negative']})")
    
    plt.tight_layout()
    return fig


def plot_epa_scores(
    e_score: float,
    p_score: float,
    a_score: float,
    title: str = "EPA Profile",
    figsize: Tuple[int, int] = (8, 4),
):
    """
    Plot EPA scores as a bar chart.
    
    Args:
        e_score: Evaluation score
        p_score: Potency score
        a_score: Activity score
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return None
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    dimensions = ['Evaluation', 'Potency', 'Activity']
    scores = [e_score, p_score, a_score]
    
    colors = ['#27ae60' if s > 0 else '#e74c3c' for s in scores]
    
    bars = ax.barh(dimensions, scores, color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("Score")
    ax.set_title(title)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.1 if width > 0 else width - 0.3, 
               bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Helper Functions for Reading and Control
# =============================================================================

def make_epa_activations(
    rep_readers: Dict[str, 'RepReader'],
    layers: List[int],
    e_coeff: float = 0.0,
    p_coeff: float = 0.0,
    a_coeff: float = 0.0,
    device = None,
    dtype = None,
    normalize: bool = True,
) -> Dict[int, 'torch.Tensor']:
    """
    Create combined EPA activation dictionary for control.
    
    Direction vectors are L2-normalized by default so that the coefficient
    directly controls the magnitude of the perturbation (a coefficient of 1.0
    adds a unit vector). Without normalization, PCA direction vectors of
    4096-dimensional hidden states can have very large norms, causing even
    small coefficients to produce exaggerated steering effects.
    
    Args:
        rep_readers: Dict mapping dimension name to RepReader
        layers: List of layer indices
        e_coeff: Evaluation control coefficient
        p_coeff: Potency control coefficient
        a_coeff: Activity control coefficient
        device: Target device
        dtype: Target dtype
        normalize: If True, L2-normalize direction vectors before scaling.
                   This makes the coefficient directly interpretable as
                   the perturbation magnitude. Default True.
    
    Returns:
        Dict mapping layer index to activation tensor
    """
    import torch
    
    # Determine hidden dimension from the first available RepReader direction
    # so we can initialize zero-valued tensors for all layers.
    hidden_dim = None
    for reader in rep_readers.values():
        for layer, direction in reader.directions.items():
            hidden_dim = direction.shape[-1]
            break
        if hidden_dim is not None:
            break
    
    # Pre-populate activations with zeros for every requested layer.
    # This ensures the rep-control pipeline always finds an entry for
    # each layer, even when all EPA coefficients are zero (baseline).
    activations = {}
    if hidden_dim is not None:
        for layer in layers:
            t = torch.zeros(1, hidden_dim)
            if dtype is not None:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            activations[layer] = t
    
    for dim, coeff in [("evaluation", e_coeff), 
                       ("potency", p_coeff), 
                       ("activity", a_coeff)]:
        if coeff == 0.0 or dim not in rep_readers:
            continue
            
        reader = rep_readers[dim]
        for layer in layers:
            if layer not in reader.directions:
                continue
                
            sign = reader.direction_signs.get(layer, 1)
            # Ensure sign is a scalar (it may be a numpy array)
            if hasattr(sign, 'item'):
                sign = sign.item()
            sign = float(sign)
            direction = torch.tensor(reader.directions[layer])
            
            # L2-normalize so coefficient has predictable magnitude effect
            if normalize:
                norm = direction.norm(p=2)
                if norm > 0:
                    direction = direction / norm
            
            if dtype is not None:
                direction = direction.to(dtype)
            if device is not None:
                direction = direction.to(device)
            
            activation = coeff * sign * direction
            
            if layer in activations:
                activations[layer] = activations[layer] + activation
            else:
                activations[layer] = activation
    
    return activations


def read_epa_scores(
    pipeline,
    rep_readers: Dict[str, 'RepReader'],
    text: str,
    layers: List[int] = None,
    neutral_context: str = "What do you think?",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    **tokenizer_kwargs,
) -> Dict[str, float]:
    """
    Read EPA scores from text.
    
    Args:
        pipeline: RepReadingPipeline
        rep_readers: Dict mapping dimension name to RepReader
        text: Text to analyze
        layers: Layers to average for final scoring (if None, uses all layers from rep_reader)
        neutral_context: Context prompt for formatting (used if user_prompt not provided)
        system_prompt: Optional custom system prompt for formatting
        user_prompt: Optional custom user prompt for formatting (overrides neutral_context)
        **tokenizer_kwargs: Tokenizer arguments
    
    Returns:
        Dict with 'evaluation', 'potency', 'activity' scores
    """
    # Format text as assistant response with custom prompts if provided
    if system_prompt is not None or user_prompt is not None:
        # Use custom prompts
        sys = system_prompt if system_prompt is not None else "You are in a conversation."
        usr = user_prompt if user_prompt is not None else neutral_context
        formatted = format_llama3_prompt(sys, usr, text)
    else:
        # Use default formatting
        formatted = format_for_reading(text, neutral_context)
    
    tokenizer_kwargs.setdefault('padding', True)
    tokenizer_kwargs.setdefault('truncation', True)
    
    scores = {}
    
    for dim, reader in rep_readers.items():
        # Use all layers from the rep_reader (required by pipeline)
        all_layers = list(reader.directions.keys())
        
        dim_scores = pipeline(
            [formatted],
            hidden_layers=all_layers,
            rep_reader=reader,
            **tokenizer_kwargs
        )
        
        # Determine which layers to average over
        layers_to_avg = layers if layers is not None else all_layers
        
        # Average across specified layers (only those that exist in the result)
        layer_scores = [dim_scores[0][layer] for layer in layers_to_avg if layer in dim_scores[0]]
        scores[dim] = float(np.mean(layer_scores)) if layer_scores else 0.0
    
    return scores


def read_epa(
    pipeline,
    rep_readers: Dict[str, 'RepReader'],
    text: str,
    layers: List[int],
    neutral_context: str = "What do you think?",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    **tokenizer_kwargs,
) -> 'EPA':
    """
    Read raw EPA values from text using the extracted directions.
    
    This is a convenience wrapper around read_epa_scores that returns an EPA object.
    
    Args:
        pipeline: RepReadingPipeline
        rep_readers: Dict mapping dimension name to RepReader
        text: Text to analyze
        layers: Layers to average for final scoring
        neutral_context: Context prompt for formatting (used if user_prompt not provided)
        system_prompt: Optional custom system prompt for formatting
        user_prompt: Optional custom user prompt for formatting (overrides neutral_context)
        **tokenizer_kwargs: Tokenizer arguments (e.g., padding=True, truncation=True)
    
    Returns:
        EPA object with raw (uncalibrated) values.
    """
    # Import EPA here to avoid circular imports
    from examples.act.act_core import EPA
    
    scores = read_epa_scores(
        pipeline=pipeline,
        rep_readers=rep_readers,
        text=text,
        layers=layers,
        neutral_context=neutral_context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **tokenizer_kwargs,
    )
    return EPA(
        e=scores.get('evaluation', 0.0),
        p=scores.get('potency', 0.0),
        a=scores.get('activity', 0.0)
    )


def steer_generation(
    model,
    tokenizer,
    rep_readers: Dict[str, 'RepReader'],
    layers: List[int],
    prompt: str,
    target_epa: Tuple[float, float, float],
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rep_control_pipeline = None,
    normalize: bool = True,
    do_sample: bool = True,
    **generation_kwargs,
) -> str:
    """
    Generate text with EPA steering.
    
    Uses RepControlPipeline from the repe library to apply steering vectors.
    Direction vectors are L2-normalized by default.
    
    Args:
        model: The language model for generation
        tokenizer: The tokenizer
        rep_readers: Dict mapping dimension name to RepReader
        layers: Layer indices to apply steering to
        prompt: The input prompt for the LLM
        target_epa: Target EPA values (e, p, a) for steering
        steering_coefficient: Multiplier for steering strength (default 1.0)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        rep_control_pipeline: Optional pre-configured RepControlPipeline.
        normalize: If True, L2-normalize direction vectors. Default True.
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Generated text (response only, without prompt)
    """
    import torch
    from transformers import pipeline
    
    e_target, p_target, a_target = target_epa
    
    # Create activation vectors for the target EPA
    activations = make_epa_activations(
        rep_readers=rep_readers,
        layers=layers,
        e_coeff=e_target * steering_coefficient,
        p_coeff=p_target * steering_coefficient,
        a_coeff=a_target * steering_coefficient,
        device=model.device,
        dtype=model.dtype,
        normalize=normalize,
    )
    
    # Create or use existing RepControlPipeline
    if rep_control_pipeline is None:
        rep_control_pipeline = pipeline(
            "rep-control",
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            control_method="reading_vec",
        )
    
    # Generate with representation control
    outputs = rep_control_pipeline(
        prompt,
        activations=activations,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        **generation_kwargs,
    )
    
    # Extract response text (pipeline returns list of dicts with 'generated_text')
    generated_text = outputs[0]['generated_text']
    
    # Remove the prompt from the generated text
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    return response
