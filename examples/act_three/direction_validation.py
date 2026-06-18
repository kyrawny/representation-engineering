"""
Direction Validation & Layer Selection.

After extracting EPA direction vectors, this module:
1. Tests per-layer classification accuracy for each EPA dimension.
2. Selects steering layers using the RepE paper methodology:
   - Layers with high accuracy (≥ threshold) across all dimensions.
   - Spaced selection (every Nth qualifying layer) to prevent cascading.
   - Avoidance of the last layers for stability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .prompt_formatting import DIMENSION_NAMES


@dataclass
class LayerSelectionResult:
    """Result of the layer selection procedure."""

    # Per-dimension per-layer accuracy: {dim: {layer: accuracy}}
    layer_accuracies: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # Best layer and accuracy per dimension
    best_layers: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    # All layers meeting the accuracy threshold for every dimension
    qualifying_layers: List[int] = field(default_factory=list)

    # Final spaced steering layers
    steering_layers: List[int] = field(default_factory=list)

    # Parameters used
    min_accuracy: float = 0.90
    layer_step: int = 3
    skip_last_n: int = 2


def compute_layer_accuracies(
    rep_pipeline,
    rep_readers: Dict[str, Any],
    datasets: Dict[str, Dict],
    hidden_layers: List[int],
    n_test: int = 50,
    batch_size: int = 8,
) -> Dict[str, Dict[int, float]]:
    """
    Compute per-layer classification accuracy for each EPA dimension.

    For each dimension, feeds the first *n_test* prompts through the
    rep-reading pipeline and checks whether the direction correctly
    identifies the positive example in each pair.

    Args:
        rep_pipeline: A ``rep-reading`` HuggingFace pipeline.
        rep_readers: Dict mapping dimension name to RepReader.
        datasets: Contrastive datasets from ``create_all_epa_datasets()``.
        hidden_layers: Layer indices.
        n_test: Number of prompts to use for testing.
        batch_size: Batch size for the pipeline.

    Returns:
        Dict ``{dimension: {layer: accuracy}}``.
    """
    layer_accuracies: Dict[str, Dict[int, float]] = {}

    for dimension in DIMENSION_NAMES:
        data = datasets[dimension]
        reader = rep_readers[dimension]

        test_data = data["train"]["data"][:n_test]
        test_labels = data["train"]["labels"][: n_test // 2]

        scores = rep_pipeline(
            test_data,
            hidden_layers=hidden_layers,
            rep_reader=reader,
            batch_size=batch_size,
            padding=True,
            truncation=True,
        )

        layer_acc: Dict[int, float] = {}
        for layer in hidden_layers:
            correct = 0
            total = 0

            for j in range(0, len(scores), 2):
                pair_idx = j // 2
                if pair_idx >= len(test_labels):
                    break

                score1 = scores[j][layer]
                score2 = scores[j + 1][layer]

                if test_labels[pair_idx][0]:  # First is positive
                    correct += 1 if score1 > score2 else 0
                else:
                    correct += 1 if score2 > score1 else 0
                total += 1

            layer_acc[layer] = correct / total if total > 0 else 0.0

        layer_accuracies[dimension] = layer_acc

    return layer_accuracies


def select_steering_layers(
    layer_accuracies: Dict[str, Dict[int, float]],
    hidden_layers: List[int],
    min_accuracy: float = 0.90,
    layer_step: int = 3,
    skip_last_n: int = 2,
) -> LayerSelectionResult:
    """
    Select steering layers following the RepE paper methodology.

    Methodology (arXiv 2310.01405v4, Appendix C.2):
    1. Identify layers with accuracy ≥ *min_accuracy* across **all** dimensions.
    2. Skip the last *skip_last_n* layers (closest to output) for stability.
    3. Take every *layer_step*-th remaining layer to prevent cascading.

    Args:
        layer_accuracies: Per-layer accuracies from ``compute_layer_accuracies()``.
        hidden_layers: All candidate layers.
        min_accuracy: Minimum accuracy threshold (default 0.90).
        layer_step: Spacing between selected layers (default 3).
        skip_last_n: Number of qualifying layers nearest the output to skip (default 2).

    Returns:
        ``LayerSelectionResult`` with all intermediate and final results.
    """
    result = LayerSelectionResult(
        layer_accuracies=layer_accuracies,
        min_accuracy=min_accuracy,
        layer_step=layer_step,
        skip_last_n=skip_last_n,
    )

    # Best layer per dimension
    for dim in DIMENSION_NAMES:
        accs = layer_accuracies[dim]
        best_layer = max(accs, key=lambda l: accs[l])
        result.best_layers[dim] = (best_layer, accs[best_layer])

    # Qualifying layers: accuracy ≥ threshold for ALL dimensions
    qualifying = []
    for layer in hidden_layers:
        if all(
            layer_accuracies[dim].get(layer, 0) >= min_accuracy
            for dim in DIMENSION_NAMES
        ):
            qualifying.append(layer)

    qualifying.sort()

    # Skip the last N layers (closest to output)
    if len(qualifying) > skip_last_n:
        qualifying = qualifying[:-skip_last_n]

    result.qualifying_layers = qualifying

    # Apply spacing
    result.steering_layers = qualifying[::layer_step]

    return result
