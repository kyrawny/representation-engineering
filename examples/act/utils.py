"""
Utility functions for Affect Control Theory (ACT) EPA vector extraction.

This module provides:
- Dataset creation for E, P, A contrastive prompts
- Visualization functions for t-SNE, LAT scans, and per-token detection
- Helper functions for Llama 3.1 prompt formatting
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


def get_epa_system_prompt(dimension: str, target: str) -> str:
    """
    Generate system prompt for EPA dimension extraction.
    
    Args:
        dimension: One of 'evaluation', 'potency', 'activity'
        target: Either 'positive' or 'negative' for the target polarity
    
    Returns:
        Formatted system prompt string.
    """
    dim_info = EPA_DIMENSIONS[dimension]
    target_term = dim_info["positive"] if target == "positive" else dim_info["negative"]
    neutral_1, neutral_2 = dim_info["neutral_dims"]
    
    system_prompt = f"""You are a human participating in a verbal conversation. Your goal is to generate a response that is a pure representation of a specific social dimension from Affect Control Theory.

The EPA Framework:
- Evaluation (Good vs. Bad): Refers to morality, altruism, and social desirability.
- Potency (Potent vs. Impotent): Refers to power, authority, dominance, and strength.
- Activity (Active vs. Inactive): Refers to energy level, speed, volatility, and liveliness.

Your Task: Respond to the user while being perceived as extremely {target_term}.

Constraints:
- Neutrality: On the other two dimensions, you must remain strictly neutral. Do not lean toward being {neutral_1[0]} or {neutral_1[1]}. Do not lean toward being {neutral_2[0]} or {neutral_2[1]}.
- Subtlety: Avoid using the target word itself. Instead, embody the essence of the dimension through your tone, word choice, and social posture.
- Naturalism: Maintain the flow of a standard verbal interaction. Do not include any non-verbal cues or actions, such as emojis or tags indicating gestures and laughter. Only include the transcribed text of the response, with no additional formatting."""
    
    return system_prompt


def format_llama3_prompt(
    system_prompt: str,
    user_input: str,
    assistant_start: str = "",
    include_bos: bool = True
) -> str:
    """
    Format a complete Llama 3.1 Instruct prompt.
    
    Args:
        system_prompt: System prompt content.
        user_input: User message content.
        assistant_start: Truncated assistant response start.
        include_bos: Whether to include BOS token.
    
    Returns:
        Formatted prompt string.
    """
    parts = []
    if include_bos:
        parts.append(LLAMA3_BOS)
    
    parts.append(LLAMA3_SYSTEM_START)
    parts.append(system_prompt)
    parts.append(LLAMA3_EOT)
    parts.append(LLAMA3_USER_START)
    parts.append(user_input)
    parts.append(LLAMA3_EOT)
    parts.append(LLAMA3_ASSISTANT_START)
    parts.append(assistant_start)
    
    return "".join(parts)


def format_for_reading(text: str, neutral_context: str = "What do you think?") -> str:
    """
    Format text as if it were an assistant response for EPA reading.
    
    Args:
        text: The text to read EPA values from.
        neutral_context: Neutral user message to provide context.
    
    Returns:
        Formatted prompt with text in assistant position.
    """
    system = "You are in a conversation."
    return format_llama3_prompt(system, neutral_context, text)


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
    
    user_inputs, truncated_outputs = load_act_data(data_dir)
    
    # Get system prompts for positive and negative
    system_pos = get_epa_system_prompt(dimension, "positive")
    system_neg = get_epa_system_prompt(dimension, "negative")
    
    train_data = []
    train_labels = []
    
    # Create contrastive pairs
    for i in range(n_train):
        user_input = random.choice(user_inputs)
        truncated = random.choice(truncated_outputs)
        
        # Create positive and negative prompts with same content
        pos_prompt = format_llama3_prompt(system_pos, user_input, truncated)
        neg_prompt = format_llama3_prompt(system_neg, user_input, truncated)
        
        # Shuffle for balanced labels
        pair = [pos_prompt, neg_prompt]
        is_first_positive = True
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
) -> Dict[int, 'torch.Tensor']:
    """
    Create combined EPA activation dictionary for control.
    
    Args:
        rep_readers: Dict mapping dimension name to RepReader
        layers: List of layer indices
        e_coeff: Evaluation control coefficient
        p_coeff: Potency control coefficient
        a_coeff: Activity control coefficient
        device: Target device
        dtype: Target dtype
    
    Returns:
        Dict mapping layer index to activation tensor
    """
    import torch
    
    activations = {}
    
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
    **tokenizer_kwargs,
) -> Dict[str, float]:
    """
    Read EPA scores from text.
    
    Args:
        pipeline: RepReadingPipeline
        rep_readers: Dict mapping dimension name to RepReader
        text: Text to analyze
        layers: Layers to average for final scoring (if None, uses all layers from rep_reader)
        neutral_context: Context prompt for formatting
        **tokenizer_kwargs: Tokenizer arguments
    
    Returns:
        Dict with 'evaluation', 'potency', 'activity' scores
    """
    # Format text as assistant response
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
