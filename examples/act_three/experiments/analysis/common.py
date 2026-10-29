"""
Shared loaders for the revision analyses.

Every analysis reads *raw* trial logs (never aggregate fields written by the
experiment scripts) so that numbers in the paper can be traced back to a
single source (WS0).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ANALYSIS_DIR.parent
ACT_THREE_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = ACT_THREE_DIR.parent.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
OUT_DIR = RESULTS_DIR / "analysis"
FIG_DIR = OUT_DIR / "figures"
PAPER_DIR = ACT_THREE_DIR / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
SURVEY_DIR = EXPERIMENTS_DIR / "results_05_02_02"
DATA_DIR = REPO_ROOT / "data" / "act"

DIMS = ["evaluation", "potency", "activity"]
DIM_SHORT = {"evaluation": "E", "potency": "P", "activity": "A"}
CONDITIONS = ["unsteered", "pe_only", "repe_only", "hybrid"]
COND_LABEL = {"unsteered": "Unst.", "pe_only": "PE", "repe_only": "RepE", "hybrid": "Hybrid"}

# Raw logs per model (exp 08 contains all four conditions on 1,300 trials)
HYBRID_LOGS: Dict[str, Path] = {
    "llama3": RESULTS_DIR / "08_hybrid_steering.json",
    "ministral": RESULTS_DIR / "ministral" / "08_hybrid_steering.json",
}
READING_LOGS: Dict[str, Path] = {
    "llama3": RESULTS_DIR / "01_reading_quality.json",
}
COHERENCE_LOGS: Dict[str, Path] = {
    "llama3": RESULTS_DIR / "07_coherence_evaluation.json",
    "ministral": RESULTS_DIR / "ministral" / "07_coherence_evaluation.json",
}
ABLATION_LOGS: Dict[str, Path] = {
    "llama3": RESULTS_DIR / "11_dimension_ablation.json",
    "ministral": RESULTS_DIR / "ministral" / "11_dimension_ablation.json",
}
MODEL_LABEL = {"llama3": "Llama-3.1-8B", "ministral": "Ministral-8B"}


def ensure_dirs() -> None:
    for d in (OUT_DIR, FIG_DIR, GENERATED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, name: str) -> Path:
    ensure_dirs()
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_np_default)
    return path


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def scenario_category(scenario_id: str) -> str:
    """'pos_e_01' -> 'pos_e', 'mixed_ep_03' -> 'mixed_ep'."""
    return scenario_id.rsplit("_", 1)[0]


def load_hybrid_trials(model: str) -> pd.DataFrame:
    """Tidy long table: one row per (trial, condition, dimension)."""
    d = load_json(HYBRID_LOGS[model])
    rows: List[dict] = []
    for i, t in enumerate(d["trials"]):
        for cond in CONDITIONS:
            c = t["conditions"][cond]
            for dim in DIMS:
                rows.append({
                    "model": model,
                    "trial": i,
                    "scenario_id": t["scenario_id"],
                    "category": scenario_category(t["scenario_id"]),
                    "pair": t["pair"],
                    "agent": t["agent_identity"],
                    "user": t["user_identity"],
                    "condition": cond,
                    "dim": dim,
                    "target": t["target_epa"][dim],
                    "observed": c["epa"][dim],
                    "distance": abs(c["epa"][dim] - t["target_epa"][dim]),
                    "logged_distance": c["distances"][dim],
                })
    return pd.DataFrame(rows)


def load_hybrid_raw(model: str) -> dict:
    return load_json(HYBRID_LOGS[model])
