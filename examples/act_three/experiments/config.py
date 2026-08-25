"""
Shared configuration for the experiment suite.

All paths, identity pairs, model defaults, and experiment constants
are defined here so that individual experiment scripts stay DRY.

The ``set_model()`` function reconfigures all paths for a given model.
Call it before ``load_experiment_components()`` to switch models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================================================================
# Paths (project layout)
# =========================================================================

# Resolve project layout relative to *this* file
_THIS_DIR = Path(__file__).resolve().parent           # experiments/
ACT_THREE_DIR = _THIS_DIR.parent                       # act_three/
EXAMPLES_DIR = ACT_THREE_DIR.parent                    # examples/
REPO_ROOT = EXAMPLES_DIR.parent                        # representation-engineering/

DATA_DIR = REPO_ROOT / "data" / "act"
MODELS_DIR = ACT_THREE_DIR / "models"

# ACT dictionaries (model-independent)
IDENTITIES_CSV = str(DATA_DIR / "MTurkInteract_Identities.csv")
BEHAVIORS_CSV = str(DATA_DIR / "MTurkInteract_Behaviors.csv")


# =========================================================================
# Model configuration
# =========================================================================

# Default model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
READING_METHOD = "ElasticNet"

# Whether to prefer orthogonalised directions (if the file exists)
USE_ORTHOGONAL_DIRECTIONS = True


def _resolve_model_dir(model_name: str) -> Path:
    """Return the model-specific artefact directory."""
    from ..model_registry import get_short_name
    short = get_short_name(model_name)
    return MODELS_DIR / short


def _load_model_config(model_dir: Path) -> dict:
    """Load per-model config.json if it exists."""
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


# ---- Per-model artefact paths (populated by set_model) ----

# Pre-computed artefacts (defaults for Llama)
_model_dir = _resolve_model_dir(MODEL_NAME)
DIRECTIONS_PATH = str(_model_dir / "epa_directions.pkl")
DIRECTIONS_PATH_ORTHO = str(_model_dir / "epa_directions_ortho.pkl")
READING_RESULTS_PATH = str(_model_dir / "epa_reading_tuning_v2_results.json")
STEERING_RESULTS_PATH = str(_model_dir / "epa_tuning_results.json")

# Calibration datasets
TUNING_TRAIN_PATH = str(_model_dir / "epa_tuning_dataset_train.json")
TUNING_TEST_PATH = str(_model_dir / "epa_tuning_dataset_test.json")

# Default results directory
RESULTS_DIR = _THIS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Per-dimension optimised steering coefficients
PER_DIM_COEFFICIENTS = {
    "evaluation": 0.5,
    "potency": 0.2,
    "activity": 0.2,
}


def set_model(model_name: str) -> None:
    """Reconfigure all paths and coefficients for a given model.

    Call this before ``load_experiment_components()`` to switch models.
    Results will be saved to ``results/{short_name}/``.

    Args:
        model_name: HuggingFace model identifier.
    """
    global MODEL_NAME, DIRECTIONS_PATH, DIRECTIONS_PATH_ORTHO
    global READING_RESULTS_PATH, STEERING_RESULTS_PATH
    global TUNING_TRAIN_PATH, TUNING_TEST_PATH
    global RESULTS_DIR, FIGURES_DIR
    global PER_DIM_COEFFICIENTS, READING_METHOD, USE_ORTHOGONAL_DIRECTIONS

    from ..model_registry import get_short_name

    MODEL_NAME = model_name
    short = get_short_name(model_name)
    model_dir = MODELS_DIR / short

    DIRECTIONS_PATH = str(model_dir / "epa_directions.pkl")
    DIRECTIONS_PATH_ORTHO = str(model_dir / "epa_directions_ortho.pkl")
    READING_RESULTS_PATH = str(model_dir / "epa_reading_tuning_v2_results.json")
    STEERING_RESULTS_PATH = str(model_dir / "epa_tuning_results.json")
    TUNING_TRAIN_PATH = str(model_dir / "epa_tuning_dataset_train.json")
    TUNING_TEST_PATH = str(model_dir / "epa_tuning_dataset_test.json")

    RESULTS_DIR = _THIS_DIR / "results" / short
    FIGURES_DIR = RESULTS_DIR / "figures"

    # Load per-model config if available
    cfg = _load_model_config(model_dir)
    if "per_dim_coefficients" in cfg:
        PER_DIM_COEFFICIENTS = cfg["per_dim_coefficients"]
    if "reading_method" in cfg:
        READING_METHOD = cfg["reading_method"]
    if "use_orthogonal_directions" in cfg:
        USE_ORTHOGONAL_DIRECTIONS = cfg["use_orthogonal_directions"]


def get_model_dir(model_name: Optional[str] = None) -> Path:
    """Return the artefact directory for a model.

    Args:
        model_name: HuggingFace model identifier. If None, uses the
            current ``MODEL_NAME``.

    Returns:
        Path to ``models/{short_name}/``.
    """
    name = model_name or MODEL_NAME
    return _resolve_model_dir(name)


# =========================================================================
# Identity pairs
# =========================================================================

# Each entry: (pair_name, agent_term, user_term)
# Terms must exist in MTurkInteract_Identities.csv
IDENTITY_PAIRS: List[Tuple[str, str, str]] = [
    ("counselor_client",    "counselor",    "client"),
    ("boss_subordinate",    "boss",         "subordinate"),
    ("teacher_student",     "teacher",      "student"),
    ("doctor_patient",      "doctor",       "patient"),
    ("friend_friend",       "friend",       "friend"),
    ("client_counselor",    "client",       "counselor"),
    ("subordinate_boss",    "subordinate",  "boss"),
    ("student_teacher",     "student",      "teacher"),
    ("patient_doctor",      "patient",      "doctor"),
    ("daughter_mother",     "daughter",     "mother"),
    ("mother_daughter",     "mother",       "daughter"),
    ("receptionist_visitor","receptionist", "visitor"),
    ("visitor_receptionist","visitor",      "receptionist"),
]


# =========================================================================
# Generation defaults
# =========================================================================

GENERATION_DEFAULTS: Dict = {
    "max_new_tokens": 256,
    "do_sample": False,
    "repetition_penalty": 1.2,
}


# =========================================================================
# Experiment sizing
# =========================================================================

# Full experiment: all scenarios × all identity pairs
QUICK_N_SCENARIOS = 3
QUICK_N_IDENTITY_PAIRS = 3      # first N pairs from IDENTITY_PAIRS

# Coefficient sweep values (focused on coherent range; text degenerates above ~2.0)
COEFF_SWEEP_VALUES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

# Checkpoint frequency (trials between saves)
CHECKPOINT_EVERY = 50


# =========================================================================
# Dimension names (re-exported for convenience)
# =========================================================================

DIMENSION_NAMES = ["evaluation", "potency", "activity"]
