"""
Shared configuration for the experiment suite.

All paths, identity pairs, model defaults, and experiment constants
are defined here so that individual experiment scripts stay DRY.
"""

from pathlib import Path
from typing import Dict, List, Tuple


# =========================================================================
# Paths
# =========================================================================

# Resolve project layout relative to *this* file
_THIS_DIR = Path(__file__).resolve().parent           # experiments/
ACT_THREE_DIR = _THIS_DIR.parent                       # act_three/
EXAMPLES_DIR = ACT_THREE_DIR.parent                    # examples/
REPO_ROOT = EXAMPLES_DIR.parent                        # representation-engineering/

DATA_DIR = REPO_ROOT / "data" / "act"

# Pre-computed artefacts
DIRECTIONS_PATH = str(ACT_THREE_DIR / "epa_directions.pkl")
DIRECTIONS_PATH_ORTHO = str(ACT_THREE_DIR / "epa_directions_ortho.pkl")
READING_RESULTS_PATH = str(ACT_THREE_DIR / "epa_reading_tuning_v2_results.json")
STEERING_RESULTS_PATH = str(ACT_THREE_DIR / "epa_tuning_results.json")

# Whether to prefer orthogonalised directions (if the file exists)
USE_ORTHOGONAL_DIRECTIONS = True

# Calibration datasets
TUNING_TRAIN_PATH = str(ACT_THREE_DIR / "epa_tuning_dataset_train.json")
TUNING_TEST_PATH = str(ACT_THREE_DIR / "epa_tuning_dataset_test.json")

# ACT dictionaries
IDENTITIES_CSV = str(DATA_DIR / "MTurkInteract_Identities.csv")
BEHAVIORS_CSV = str(DATA_DIR / "MTurkInteract_Behaviors.csv")

# Default results directory
RESULTS_DIR = _THIS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# =========================================================================
# Model
# =========================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
READING_METHOD = "ElasticNet"


# =========================================================================
# Identity pairs
# =========================================================================

# Each entry: (pair_name, agent_term, user_term)
# Terms must exist in MTurkInteract_Identities.csv
IDENTITY_PAIRS: List[Tuple[str, str, str]] = [
    ("counselor_client",   "counselor",   "client"),
    ("boss_subordinate",   "boss",        "subordinate"),
    ("teacher_student",    "teacher",     "student"),
    ("doctor_patient",     "doctor",      "patient"),
    ("friend_friend",      "friend",      "friend"),
]


# =========================================================================
# Generation defaults
# =========================================================================

GENERATION_DEFAULTS: Dict = {
    "max_new_tokens": 128,
    "do_sample": False,
    "repetition_penalty": 1.2,
}


# =========================================================================
# Experiment sizing
# =========================================================================

# Full experiment: all scenarios × all identity pairs
QUICK_N_SCENARIOS = 10
QUICK_N_IDENTITY_PAIRS = 2      # first N pairs from IDENTITY_PAIRS

# Coefficient sweep values (focused on coherent range; text degenerates above ~2.0)
COEFF_SWEEP_VALUES = [
    0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0,
]

# Checkpoint frequency (trials between saves)
CHECKPOINT_EVERY = 50


# =========================================================================
# Per-dimension optimised steering coefficients
# =========================================================================

# Empirically derived from per-dimension coefficient sweep
# (results/per_dim_sign_check.json). Steering reverses above ~1.5
# due to text degeneration, so these stay in the coherent range.
# EPASteerer multiplies coeff * target_value * direction per layer,
# with typical |target| in [0.5, 2.0] and 9-11 selected layers.
PER_DIM_COEFFICIENTS = {
    "evaluation": 0.5,     # sweet spot: 0.75-1.0 (reverses at 2.0)
    "potency": 0.2,         # sweet spot: ~0.5 (overshoots at 1.0)
    "activity": 0.2,       # sweet spot: 0.75-1.0 (reverses at 2.0)
}


# =========================================================================
# Dimension names (re-exported for convenience)
# =========================================================================

DIMENSION_NAMES = ["evaluation", "potency", "activity"]
