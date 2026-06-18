"""
Visualization Functions for EPA Analysis.

Provides:
- t-SNE visualization of contrastive hidden states
- LAT scan heatmaps across layers and tokens
- Per-token EPA intensity with colored text
- EPA score bar charts
"""

from typing import Dict, List, Tuple

import numpy as np

from .prompt_formatting import EPA_DIMENSIONS


def plot_tsne_epa(
    hidden_states_pos: np.ndarray,
    hidden_states_neg: np.ndarray,
    dimension: str,
    layer: int,
    perplexity: int = 30,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Create t-SNE visualization for an EPA dimension (Figure 14 style).

    Args:
        hidden_states_pos: Hidden states for positive class (n_samples, hidden_dim).
        hidden_states_neg: Hidden states for negative class (n_samples, hidden_dim).
        dimension: EPA dimension name.
        layer: Layer number for title.
        perplexity: t-SNE perplexity parameter.
        figsize: Figure size.

    Returns:
        matplotlib Figure, or None if dependencies are missing.
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
    labels = np.array(
        [1] * len(hidden_states_pos) + [0] * len(hidden_states_neg)
    )

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(all_states)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    colors = ["#e74c3c", "#27ae60"]  # Red for negative, Green for positive

    for label, color, name in [
        (0, colors[0], dim_info["negative"]),
        (1, colors[1], dim_info["positive"]),
    ]:
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=color,
            label=name.capitalize(),
            alpha=0.6,
            s=30,
        )

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
        scores_by_layer: Dict mapping layer index to per-token scores.
        tokens: List of token strings.
        dimension: EPA dimension name.
        start_idx: Starting token index.
        n_tokens: Number of tokens to show.
        figsize: Figure size.

    Returns:
        matplotlib Figure, or None if dependencies are missing.
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
        layer_scores = scores[start_idx : start_idx + n_tokens]
        matrix.append(layer_scores)

    matrix = np.array(matrix)

    # Normalize
    bound = np.percentile(np.abs(matrix), 95)
    matrix = np.clip(matrix, -bound, bound)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0,
        vmin=-bound,
        vmax=bound,
        ax=ax,
        xticklabels=5,
        yticklabels=5,
    )
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(f"LAT Scan: {dimension.capitalize()}")
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
        tokens: List of token strings.
        scores: Per-token scores.
        dimension: EPA dimension name.
        threshold: Normalization threshold.
        figsize: Figure size.

    Returns:
        matplotlib Figure, or None if dependencies are missing.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ImportError:
        print("matplotlib required for plotting")
        return None

    dim_info = EPA_DIMENSIONS[dimension]

    # Clean tokens
    clean_tokens = []
    for token in tokens:
        t = token.replace("\u2581", " ").replace("\u0120", " ").replace("\u010a", "\n")
        t = t.replace("<|", "").replace("|>", "")
        clean_tokens.append(t)

    # Normalize scores
    scores = np.array(scores) - threshold
    scores = scores / (np.std(scores) + 1e-8)
    mag = max(0.5, np.percentile(np.abs(scores), 90))
    scores = np.clip(scores, -mag, mag)

    # Create colormap (red for negative, green for positive)
    cmap = LinearSegmentedColormap.from_list(
        "epa", ["#e74c3c", "#f5f5dc", "#27ae60"], N=256
    )
    norm = Normalize(vmin=-mag, vmax=mag)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 10)
    ax.axis("off")

    x, y = 10, 9
    max_x = 990

    for token, score in zip(clean_tokens, scores):
        if not token.strip():
            continue

        color = cmap(norm(score))
        text = ax.text(
            x,
            y,
            token,
            fontsize=10,
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.8, pad=1),
        )

        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer).transformed(
            ax.transData.inverted()
        )
        text_width = bbox.width

        x += text_width + 3
        if x > max_x:
            x = 10
            y -= 1.2
            if y < 1:
                break

    ax.set_title(
        f"Per-Token {dimension.capitalize()} Detection\n"
        f"(Green = {dim_info['positive']}, Red = {dim_info['negative']})"
    )
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
    Plot EPA scores as a horizontal bar chart.

    Args:
        e_score: Evaluation score.
        p_score: Potency score.
        a_score: Activity score.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure, or None if dependencies are missing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    dimensions = ["Evaluation", "Potency", "Activity"]
    scores = [e_score, p_score, a_score]
    colors = ["#27ae60" if s > 0 else "#e74c3c" for s in scores]

    bars = ax.barh(dimensions, scores, color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("Score")
    ax.set_title(title)

    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(
            width + 0.1 if width > 0 else width - 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return fig
