"""
Settings for the 70B scale test.

Everything a collaborating lab may need to change is either here or in an
environment variable (see README.md).  Values marked "paper" reproduce the
Llama-3.1-8B setting reported in the paper and must not be changed.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ACT_THREE_DIR = THIS_DIR.parent
REPO_ROOT = ACT_THREE_DIR.parent.parent
MODELS_DIR = ACT_THREE_DIR / "models"
DATA_DIR = REPO_ROOT / "data" / "act"
IDENTITIES_CSV = DATA_DIR / "MTurkInteract_Identities.csv"
SCALE_RESULTS_DIR = ACT_THREE_DIR / "experiments" / "results" / "scale"
REFERENCE_DIR = THIS_DIR / "reference"          # 8B reference checks written by stage R

# Calibration data for the reader (identical text for every model; reused so
# that reader quality is comparable across scales).
TUNING_TRAIN = MODELS_DIR / "llama3" / "epa_tuning_dataset_train.json"
TUNING_TEST = MODELS_DIR / "llama3" / "epa_tuning_dataset_test.json"

# 8B result log used as the comparison point (paper Table 3, Llama column).
# Tracked in git (an exception in experiments/.gitignore).
REFERENCE_8B_LOG = ACT_THREE_DIR / "experiments" / "results" / "08_hybrid_steering.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
REFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_REGISTERED_SHORT_NAMES = {          # mirrors model_registry.MODEL_REGISTRY
    "meta-llama/Llama-3.1-8B-Instruct": "llama3",
    "mistralai/Ministral-8B-Instruct-2410": "ministral",
}
# Artefact folders of the paper's models: never overwritten by this package.
PROTECTED_SHORT_NAMES = {"llama3", "ministral", "mistral-7b-instruct-v0-3"}


def short_name(model_name: str) -> str:
    """Same rule as model_registry.get_short_name (kept torch-free)."""
    if model_name in _REGISTERED_SHORT_NAMES:
        return _REGISTERED_SHORT_NAMES[model_name]
    name = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def model_dir(model_name: str) -> Path:
    return MODELS_DIR / short_name(model_name)


def results_dir(model_name: str) -> Path:
    return SCALE_RESULTS_DIR / short_name(model_name)


# ---------------------------------------------------------------------------
# Method settings (paper; do not change)
# ---------------------------------------------------------------------------
READING_METHOD = "ElasticNet"
USE_ORTHOGONAL_DIRECTIONS = False            # models/llama3/config.json
REFERENCE_COEFFICIENTS = {"evaluation": 0.5, "potency": 0.2, "activity": 0.2}
REFERENCE_CLAMP_LIMIT = 1.0                  # EPASteerer default
DIMS = ["evaluation", "potency", "activity"]

# ---------------------------------------------------------------------------
# Steering-strength calibration (pre-specified; see README "Strength")
# ---------------------------------------------------------------------------
# Strength s multiplies BOTH the per-dimension coefficients and the clamp
# limit, so every perturbation is exactly s times the 8B perturbation for the
# same target.  s = 1 is the literal 8B setting.
STRENGTH_GRID = [0.5, 1, 2, 4, 8, 16, 32]
MAX_PPL_RATIO = 1.25        # geometric-mean self-perplexity, steered / unsteered (8B paper setting: about 1.19)
MAX_DEGENERATE_FRAC = 0.0   # no degenerate outputs allowed at the chosen strength
LENGTH_RATIO_BOUNDS = (0.5, 2.0)
DEGENERATE_MIN_CHARS = 20
DEGENERATE_MAX_BIGRAM_REPEAT = 0.2
CALIBRATION_PAIRS = ["counselor_client", "boss_subordinate", "doctor_patient"]
CALIBRATION_PER_CATEGORY = 3                 # 7 categories x 3 scenarios x 3 pairs = 63 trials

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
CORE_CONDITIONS = ["unsteered", "repe_only", "pe_only", "hybrid"]
OPTIONAL_CONDITIONS = ["repe_const", "repe_only_s1", "hybrid_s1"]
TEXT500 = 500                                 # the 8B logs keep the first 500 characters

# ---------------------------------------------------------------------------
# Verdict (pre-specified; see README "What counts as the effect holding")
# ---------------------------------------------------------------------------
N_BOOT = 2000
ALPHA = 0.05
