"""
Consolidated utility functions for representation engineering.

This module provides reusable utilities for:
- Creating contrastive datasets for direction extraction
- Visualization of concept detection and layer activations
- Evaluation of RepReader performance
"""

import numpy as np
import random
import json
import os
from typing import List, Dict, Tuple, Optional, Union, Callable
from transformers import PreTrainedTokenizer
import torch

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# Dataset Creation Functions
# =============================================================================

def create_contrastive_dataset(
    positive_template: str,
    negative_template: str,
    scenarios: List[str],
    user_tag: str = "",
    assistant_tag: str = "",
    n_train: int = None,
    shuffle_pairs: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Create a contrastive dataset for concept direction extraction.
    
    Args:
        positive_template: Template with {scenario} placeholder for positive examples.
                          E.g., "Act as an honest person. {scenario}"
        negative_template: Template with {scenario} placeholder for negative examples.
        scenarios: List of scenario strings to fill templates.
        user_tag: User instruction tag (e.g., "[INST]")
        assistant_tag: Assistant response tag (e.g., "[/INST]")
        n_train: Number of training pairs (default: all)
        shuffle_pairs: Whether to shuffle order within pairs for label balance
        seed: Random seed for reproducibility
    
    Returns:
        Dict with 'train': {'data': List[str], 'labels': List[List[bool]]}
    """
    random.seed(seed)
    
    # Format templates with tags
    pos_template = f"{user_tag} {positive_template} {assistant_tag} ".strip() + " "
    neg_template = f"{user_tag} {negative_template} {assistant_tag} ".strip() + " "
    
    # Create pairs
    pairs = []
    labels = []
    
    for scenario in scenarios:
        pos = pos_template.replace("{scenario}", scenario)
        neg = neg_template.replace("{scenario}", scenario)
        pair = [pos, neg]
        
        if shuffle_pairs:
            is_first_positive = pair[0] == pos
            random.shuffle(pair)
            labels.append([pair[0] == pos, pair[1] == pos])
        else:
            labels.append([True, False])
        
        pairs.append(pair)
    
    # Limit to n_train
    if n_train is not None:
        pairs = pairs[:n_train]
        labels = labels[:n_train]
    
    # Flatten pairs
    train_data = [item for pair in pairs for item in pair]
    
    return {
        'train': {'data': train_data, 'labels': labels}
    }


def load_honesty_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    n_train: int = 512,
    seed: int = 42,
) -> Dict:
    """
    Load and format honesty detection dataset from CSV.
    
    Args:
        data_path: Path to CSV with 'label' and 'statement' columns.
        tokenizer: Tokenizer for truncating statements.
        user_tag: Instruction user tag.
        assistant_tag: Instruction assistant tag.
        n_train: Number of training examples.
        seed: Random seed.
    
    Returns:
        Dict with 'train' and 'test' data.
    """
    if pd is None:
        raise ImportError("pandas is required for load_honesty_dataset")
    
    random.seed(seed)
    
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    
    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []
    
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)
            
            honest_statements.append(
                f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement
            )
            untruthful_statements.append(
                f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement
            )
    
    # Create training data
    combined_data = [[h, u] for h, u in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:n_train]
    
    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data_flat = [item for pair in train_data for item in pair]
    
    # Create test data
    test_data_pairs = [[h, u] for h, u in zip(honest_statements[n_train:n_train*2], 
                                               untruthful_statements[n_train:n_train*2])]
    test_data_flat = [item for pair in test_data_pairs for item in pair]
    
    return {
        'train': {'data': train_data_flat, 'labels': train_labels},
        'test': {'data': test_data_flat, 'labels': [[True, False]] * len(test_data_pairs)}
    }


def load_emotion_dataset(
    data_dir: str,
    emotion: str,
    user_tag: str = "",
    assistant_tag: str = "",
    n_samples: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Load emotion dataset for a specific emotion.
    
    Args:
        data_dir: Directory containing emotion JSON files.
        emotion: One of 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise'
        user_tag: Instruction user tag.
        assistant_tag: Instruction assistant tag.
        n_samples: Maximum samples per emotion.
        seed: Random seed.
    
    Returns:
        Dict with 'train' data.
    """
    random.seed(seed)
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    if emotion not in emotions:
        raise ValueError(f"emotion must be one of {emotions}")
    
    template = "{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} "
    
    # Load current emotion scenarios
    with open(os.path.join(data_dir, f'{emotion}.json')) as f:
        current_scenarios = list(set(json.load(f)))[:n_samples]
    
    # Load other emotion scenarios for contrast
    other_scenarios = []
    for e in emotions:
        if e != emotion:
            with open(os.path.join(data_dir, f'{e}.json')) as f:
                other_scenarios.extend(list(set(json.load(f)))[:n_samples // 5])
    random.shuffle(other_scenarios)
    other_scenarios = other_scenarios[:len(current_scenarios)]
    
    # Create pairs
    pairs = []
    labels = []
    for curr, other in zip(current_scenarios, other_scenarios):
        pair = [
            template.format(emotion=emotion, scenario=curr, user_tag=user_tag, assistant_tag=assistant_tag),
            template.format(emotion=emotion, scenario=other, user_tag=user_tag, assistant_tag=assistant_tag),
        ]
        true_s = pair[0]
        random.shuffle(pair)
        labels.append([s == true_s for s in pair])
        pairs.append(pair)
    
    train_data = [item for pair in pairs for item in pair]
    
    return {
        'train': {'data': train_data, 'labels': labels}
    }


def load_bias_dataset(
    n_train: int = 150,
    user_tag: str = "",
    assistant_tag: str = "",
    bias_type: str = "race",
    seed: int = 42,
) -> Dict:
    """
    Load StereoSet bias dataset for bias direction extraction.
    
    Args:
        n_train: Number of training pairs.
        user_tag: Instruction user tag.
        assistant_tag: Instruction assistant tag.
        bias_type: Type of bias ('race', 'gender', 'profession', 'religion')
        seed: Random seed.
    
    Returns:
        Dict with 'train' and 'test' data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required for load_bias_dataset")
    
    random.seed(seed)
    
    template = '{user_tag} Consider the bias of the following scenario.\nScenario: {scenario}\nThe amount of bias in the scenario is {assistant_tag} '
    
    dataset = load_dataset('stereoset', 'intrasentence')['validation'].shuffle(seed=seed)
    dataset = [d for d in dataset if d['bias_type'] == bias_type]
    
    pairs = []
    labels = []
    
    for d in dataset[:n_train]:
        sentences = d['sentences']
        anti_idx = sentences['gold_label'].index(0)
        stereo_idx = sentences['gold_label'].index(1)
        
        context = d['context'] + " " if 'BLANK' not in d['context'] else ""
        
        pair = [
            template.format(scenario=context + sentences['sentence'][anti_idx], 
                          user_tag=user_tag, assistant_tag=assistant_tag),
            template.format(scenario=context + sentences['sentence'][stereo_idx],
                          user_tag=user_tag, assistant_tag=assistant_tag),
        ]
        true_s = pair[0]
        random.shuffle(pair)
        labels.append([s == true_s for s in pair])
        pairs.append(pair)
    
    train_data = [item for pair in pairs for item in pair]
    
    # Test data
    test_pairs = []
    for d in dataset[n_train:]:
        sentences = d['sentences']
        anti_idx = sentences['gold_label'].index(0)
        stereo_idx = sentences['gold_label'].index(1)
        context = d['context'] + " " if 'BLANK' not in d['context'] else ""
        test_pairs.extend([
            template.format(scenario=context + sentences['sentence'][anti_idx],
                          user_tag=user_tag, assistant_tag=assistant_tag),
            template.format(scenario=context + sentences['sentence'][stereo_idx],
                          user_tag=user_tag, assistant_tag=assistant_tag),
        ])
    
    return {
        'train': {'data': train_data, 'labels': labels},
        'test': {'data': test_pairs, 'labels': [[True, False]] * (len(test_pairs) // 2)}
    }


# =============================================================================
# Activation Creation Functions
# =============================================================================

def make_activations(
    rep_reader,
    layers: List[int],
    coefficient: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[int, torch.Tensor]:
    """
    Create activation dictionary from a RepReader for use with RepControlPipeline.
    
    Args:
        rep_reader: RepReader with extracted directions.
        layers: List of layer indices to create activations for.
        coefficient: Scaling coefficient (positive = amplify, negative = suppress).
        device: Target device for tensors.
        dtype: Target dtype for tensors.
    
    Returns:
        Dict mapping layer index to activation tensor.
    """
    activations = {}
    for layer in layers:
        if layer not in rep_reader.directions:
            continue
        sign = rep_reader.direction_signs.get(layer, 1)
        direction = torch.tensor(rep_reader.directions[layer], dtype=dtype)
        if device is not None:
            direction = direction.to(device)
        activations[layer] = coefficient * sign * direction
    return activations


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_concept_detection(
    tokens: List[str],
    scores: np.ndarray,
    threshold: float = 0.0,
    title: str = "Concept Detection",
    colormap: str = 'RdYlGn',
    start_token: str = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional['plt.Figure']:
    """
    Visualize per-token concept detection scores with colored text.
    
    Args:
        tokens: List of token strings.
        scores: Array of scores per token.
        threshold: Threshold for normalization (mean to subtract).
        title: Plot title.
        colormap: Matplotlib colormap name.
        start_token: Token string to start visualization from.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure or None if plotting not available.
    """
    if not HAS_PLOTTING:
        print("matplotlib/seaborn required for plotting")
        return None
    
    # Clean tokens for display
    clean_tokens = []
    for token in tokens:
        t = token.replace('▁', ' ').replace('Ġ', ' ').replace('Ċ', '\n')
        clean_tokens.append(t)
    
    # Normalize scores
    scores = np.array(scores)
    scores = scores - threshold
    scores = scores / (np.std(scores) + 1e-8)
    mag = max(0.3, np.abs(scores).std())
    scores = np.clip(scores, -mag, mag)
    
    # Create figure
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=-mag, vmax=mag)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Find start position
    start_idx = 0
    if start_token:
        for i, t in enumerate(tokens):
            if start_token in t:
                start_idx = i
                break
    
    x, y = 10, 9
    max_x = 990
    
    for i in range(start_idx, len(clean_tokens)):
        token = clean_tokens[i]
        score = scores[i] if i < len(scores) else 0
        color = cmap(norm(score))
        
        text = ax.text(x, y, token, fontsize=11,
                      bbox=dict(facecolor=color, edgecolor='none', alpha=0.7, pad=1))
        
        # Get text width
        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        text_width = bbox.width
        
        x += text_width + 5
        if x > max_x:
            x = 10
            y -= 1.5
            if y < 1:
                break
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_layer_scan(
    tokens: List[str],
    scores_by_layer: Dict[int, np.ndarray],
    start_token_idx: int = 0,
    n_tokens: int = 40,
    title: str = "Layer Activation Scan",
    figsize: Tuple[int, int] = (10, 6),
) -> Optional['plt.Figure']:
    """
    Visualize concept activation as heatmap across layers and tokens.
    
    Args:
        tokens: List of token strings.
        scores_by_layer: Dict mapping layer index to score array.
        start_token_idx: Index of first token to show.
        n_tokens: Number of tokens to display.
        title: Plot title.
        figsize: Figure size.
    
    Returns:
        Matplotlib Figure or None if plotting not available.
    """
    if not HAS_PLOTTING:
        print("matplotlib/seaborn required for plotting")
        return None
    
    layers = sorted(scores_by_layer.keys())
    
    # Build matrix
    matrix = []
    for layer in layers:
        scores = scores_by_layer[layer]
        if len(scores.shape) > 1:
            scores = scores.flatten()
        layer_scores = scores[start_token_idx:start_token_idx + n_tokens]
        matrix.append(layer_scores)
    
    matrix = np.array(matrix)
    
    # Normalize
    bound = np.mean(np.abs(matrix)) + 2 * np.std(matrix)
    matrix = np.clip(matrix, -bound, bound)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    sns.heatmap(matrix, cmap='coolwarm', center=0, 
                vmin=-bound, vmax=bound, ax=ax)
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    
    # Y-axis labels
    ax.set_yticks(np.arange(len(layers)) + 0.5)
    ax.set_yticklabels([str(l) for l in layers])
    
    plt.tight_layout()
    return fig


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_accuracy(
    rep_reader,
    pipeline,
    test_data: List[str],
    test_labels: List[List[bool]],
    hidden_layers: List[int],
    batch_size: int = 8,
    **pipeline_kwargs,
) -> Dict[int, float]:
    """
    Evaluate RepReader classification accuracy on test data.
    
    Args:
        rep_reader: Trained RepReader.
        pipeline: RepReadingPipeline.
        test_data: Flattened test prompts.
        test_labels: Labels for each test pair.
        hidden_layers: Layers to evaluate.
        batch_size: Batch size for inference.
        **pipeline_kwargs: Additional kwargs for pipeline (padding, truncation, etc.)
    
    Returns:
        Dict mapping layer index to accuracy score.
    """
    from itertools import islice
    
    # Get predictions
    H_tests = pipeline(
        test_data,
        rep_token=-1,
        hidden_layers=hidden_layers,
        rep_reader=rep_reader,
        batch_size=batch_size,
        **pipeline_kwargs
    )
    
    results = {}
    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        
        # Unflatten into choice groups
        idx = 0
        unflattened = []
        for label_group in test_labels:
            group_size = len(label_group)
            unflattened.append(H_test[idx:idx + group_size])
            idx += group_size
        
        # Calculate accuracy
        sign = rep_reader.direction_signs.get(layer, 1)
        eval_func = np.argmin if sign == -1 else np.argmax
        
        correct = 0
        for i, scores in enumerate(unflattened):
            if i < len(test_labels):
                predicted = eval_func(scores)
                actual = test_labels[i].index(True) if True in test_labels[i] else 0
                if predicted == actual:
                    correct += 1
        
        results[layer] = correct / len(unflattened) if unflattened else 0.0
    
    return results


def find_best_layer(
    accuracy_by_layer: Dict[int, float],
) -> Tuple[int, float]:
    """
    Find the layer with highest accuracy.
    
    Args:
        accuracy_by_layer: Dict mapping layer index to accuracy.
    
    Returns:
        Tuple of (best_layer, best_accuracy).
    """
    best_layer = max(accuracy_by_layer, key=accuracy_by_layer.get)
    return best_layer, accuracy_by_layer[best_layer]


# =============================================================================
# Convenience Functions
# =============================================================================

def get_default_layers(model, n_layers: int = 19) -> List[int]:
    """
    Get default layer indices for a model.
    
    Args:
        model: HuggingFace model.
        n_layers: Number of layers to return (from end).
    
    Returns:
        List of negative layer indices.
    """
    total_layers = model.config.num_hidden_layers
    n_layers = min(n_layers, total_layers - 1)
    return list(range(-1, -n_layers - 1, -1))


def get_prompt_tags(model_name: str) -> Tuple[str, str]:
    """
    Get appropriate prompt tags for common models.
    
    Args:
        model_name: Model name or path.
    
    Returns:
        Tuple of (user_tag, assistant_tag).
    """
    model_lower = model_name.lower()
    
    if "llama-2" in model_lower and "chat" in model_lower:
        return "[INST]", "[/INST]"
    elif "llama-3" in model_lower and "instruct" in model_lower:
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "mistral" in model_lower and "instruct" in model_lower:
        return "[INST]", "[/INST]"
    elif "vicuna" in model_lower:
        return "USER:", "ASSISTANT:"
    else:
        return "", ""
