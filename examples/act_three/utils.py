"""
Backward-compatibility shim for ``utils.py``.

Re-exports all public symbols from the sub-modules that the original
monolithic ``utils.py`` provided, so that existing code like::

    from utils import format_llama3_prompt, create_all_epa_datasets, ...

continues to work unchanged.
"""

# Prompt formatting
from .prompt_formatting import (
    LLAMA3_BOS,
    LLAMA3_SYSTEM_START,
    LLAMA3_USER_START,
    LLAMA3_ASSISTANT_START,
    LLAMA3_EOT,
    EPA_DIMENSIONS,
    format_llama3_prompt,
    format_for_reading,
    get_epa_extraction_template,
)

# Dataset creation
from .dataset import (
    load_act_data,
    create_epa_dataset,
    create_all_epa_datasets,
)

# Visualization
from .visualization import (
    plot_tsne_epa,
    plot_lat_scan,
    plot_per_token_detection,
    plot_epa_scores,
)

# Reading and steering helpers
from .epa_steerer import (
    make_epa_activations,
    steer_generation,
)

# EPA reading
from .epa_reader import EPAReader

__all__ = [
    # Prompt tokens
    "LLAMA3_BOS",
    "LLAMA3_SYSTEM_START",
    "LLAMA3_USER_START",
    "LLAMA3_ASSISTANT_START",
    "LLAMA3_EOT",
    "EPA_DIMENSIONS",
    # Formatting
    "format_llama3_prompt",
    "format_for_reading",
    "get_epa_extraction_template",
    # Dataset
    "load_act_data",
    "create_epa_dataset",
    "create_all_epa_datasets",
    # Visualization
    "plot_tsne_epa",
    "plot_lat_scan",
    "plot_per_token_detection",
    "plot_epa_scores",
    # Activations & steering
    "make_epa_activations",
    "steer_generation",
    # Reader
    "EPAReader",
]
