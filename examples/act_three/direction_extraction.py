"""
EPA Direction Extraction via PCA.

Provides functions to extract contrastive EPA direction vectors from a language
model using PCA on the difference between positive/negative hidden states.
Supports saving and loading directions as pickle files.
"""

import pickle
from typing import Dict, List, Optional, Any

from .prompt_formatting import DIMENSION_NAMES


def extract_epa_directions(
    model,
    tokenizer,
    datasets: Dict[str, Dict],
    hidden_layers: Optional[List[int]] = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Extract EPA direction vectors for all three dimensions using PCA.

    Uses the ``rep-reading`` pipeline from the ``repe`` library to compute
    contrastive PCA directions across layers.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer (``padding_side="left"``).
        datasets: Dict from ``create_all_epa_datasets()`` with keys
            ``'evaluation'``, ``'potency'``, ``'activity'``.
        hidden_layers: Layer indices to extract from. If None, uses all layers
            as negative indices ``[-1, -2, ..., -num_layers+1]``.
        batch_size: Batch size for the pipeline.
        max_length: Maximum sequence length for tokenisation.

    Returns:
        Dict mapping dimension name to a ``RepReader`` object (containing
        ``directions`` and ``direction_signs`` per layer).
    """
    from transformers import pipeline as hf_pipeline
    from repe import repe_pipeline_registry

    repe_pipeline_registry()

    rep_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    if hidden_layers is None:
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    rep_readers: Dict[str, Any] = {}

    for dimension in DIMENSION_NAMES:
        data = datasets[dimension]

        rep_reader = rep_pipeline.get_directions(
            data["train"]["data"],
            rep_token=-1,
            hidden_layers=hidden_layers,
            n_difference=1,
            train_labels=data["train"]["labels"],
            direction_method="pca",
            direction_finder_kwargs={"n_components": 1},
            batch_size=batch_size,
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        rep_readers[dimension] = rep_reader

    return rep_readers


def save_directions(
    rep_readers: Dict[str, Any],
    hidden_layers: List[int],
    model_name: str,
    path: str = "epa_directions.pkl",
) -> None:
    """
    Save extracted EPA directions to disk.

    Args:
        rep_readers: Dict of RepReader objects (one per dimension).
        hidden_layers: Layer indices used during extraction.
        model_name: Name of the model (for provenance).
        path: Output pickle file path.
    """
    with open(path, "wb") as f:
        pickle.dump(
            {
                "rep_readers": rep_readers,
                "hidden_layers": hidden_layers,
                "model_name": model_name,
            },
            f,
        )


def load_directions(path: str = "epa_directions.pkl") -> Dict:
    """
    Load previously saved EPA directions.

    Args:
        path: Path to the pickle file.

    Returns:
        Dict with keys ``'rep_readers'``, ``'hidden_layers'``, ``'model_name'``.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
