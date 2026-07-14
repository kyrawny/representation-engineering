"""
Experiment setup — shared model / reader / steerer loading.

Centralises the ~40 lines of boilerplate that every experiment needs
so individual scripts can call ``load_experiment_components()`` once
and immediately start running trials.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import config as C


@dataclass
class ExperimentComponents:
    """Everything an experiment script needs."""
    model: Any
    tokenizer: Any
    rep_readers: Dict[str, Any]
    hidden_layers: List[int]
    reader: Any                     # EPAReader
    steerer: Optional[Any]          # EPASteerer (None if not requested)
    rep_reading_pipeline: Any
    coefficients: Any               # ACTCoefficients
    identities_df: pd.DataFrame


def load_experiment_components(
    load_steerer: bool = True,
    steering_from_tuning: bool = True,
) -> ExperimentComponents:
    """
    Load model, directions, reader, and optionally steerer.

    Args:
        load_steerer: Whether to configure the EPASteerer.
        steering_from_tuning: If True, load per-dimension tuned
            hyperparameters from ``epa_tuning_results.json``.
            If False, derive from reader config with a default coeff.

    Returns:
        Populated ``ExperimentComponents``.
    """
    # Ensure repo root is on sys.path
    repo_str = str(C.REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline
    from repe import repe_pipeline_registry

    from examples.act_three import (
        EPAReader,
        EPASteerer,
        load_directions,
        get_default_coefficients,
        DIMENSION_NAMES,
    )

    # ---- Model ----
    print(f"Loading model: {C.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # ---- RepE pipelines ----
    repe_pipeline_registry()
    rep_reading_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    # ---- Directions ----
    directions_path = C.DIRECTIONS_PATH
    if C.USE_ORTHOGONAL_DIRECTIONS:
        import os
        if os.path.exists(C.DIRECTIONS_PATH_ORTHO):
            directions_path = C.DIRECTIONS_PATH_ORTHO
            print("Using orthogonalised directions")
        else:
            print("Orthogonal directions not found, using original directions")

    saved = load_directions(directions_path)
    rep_readers = saved["rep_readers"]
    hidden_layers = saved["hidden_layers"]
    print(f"Loaded directions for {len(rep_readers)} EPA dimensions, "
          f"{len(hidden_layers)} layers")

    # ---- Reader ----
    reader = EPAReader.from_tuning_results(
        C.READING_RESULTS_PATH, rep_readers, method=C.READING_METHOD,
    )
    print(f"Reader configured ({C.READING_METHOD})")

    # ---- Steerer ----
    steerer = None
    if load_steerer:
        if steering_from_tuning:
            # Use the reader's selected layers (from ElasticNet tuning)
            # and the per-dimension coefficients from config.py
            steering_configs = {}
            for dim_name in DIMENSION_NAMES:
                dim_cfg = reader.config.dimensions[dim_name]
                selected_layers = sorted(dim_cfg.selected_layers.keys())
                signs = {}
                for layer in selected_layers:
                    sign_val = rep_readers[dim_name].direction_signs.get(layer, 1)
                    if hasattr(sign_val, "item"):
                        sign_val = sign_val.item()
                    signs[layer] = float(sign_val)
                steering_configs[dim_name] = {
                    "layers": selected_layers,
                    "signs": signs,
                    "base_coeff": C.PER_DIM_COEFFICIENTS.get(dim_name, 2.0),
                }
            steerer = EPASteerer(
                model=model,
                tokenizer=tokenizer,
                rep_readers=rep_readers,
                steering_configs=steering_configs,
            )
            print(f"Steerer coefficients: "
                  f"E={steering_configs['evaluation']['base_coeff']}, "
                  f"P={steering_configs['potency']['base_coeff']}, "
                  f"A={steering_configs['activity']['base_coeff']}")
        else:
            steerer = EPASteerer.from_reader(
                reader=reader,
                rep_readers=rep_readers,
                model=model,
                tokenizer=tokenizer,
                base_coeff=2.0,
            )
        print("Steerer configured")

    # ---- ACT coefficients ----
    coefficients = get_default_coefficients()

    # ---- Identities ----
    identities_df = pd.read_csv(C.IDENTITIES_CSV)

    return ExperimentComponents(
        model=model,
        tokenizer=tokenizer,
        rep_readers=rep_readers,
        hidden_layers=hidden_layers,
        reader=reader,
        steerer=steerer,
        rep_reading_pipeline=rep_reading_pipeline,
        coefficients=coefficients,
        identities_df=identities_df,
    )


# =========================================================================
# Helpers
# =========================================================================

def get_identity_epa(df: pd.DataFrame, term: str):
    """Look up an identity term in the ACT dictionary and return its EPA."""
    from examples.act_three import EPA
    row = df[df["term"] == term].iloc[0]
    return EPA(e=float(row["E"]), p=float(row["P"]), a=float(row["A"]))


def make_system_prompt(agent_term: str, user_term: str) -> str:
    """Build a conversational preamble system prompt.

    Gives the model an identity context so it doesn't respond as a
    generic AI assistant ("we haven't had a conversation yet...").

    Args:
        agent_term: The agent identity (e.g. "counselor", "teacher").
        user_term: The user identity (e.g. "client", "student").

    Returns:
        System prompt string.
    """
    return (
        f"Pretend you're a {agent_term} replying to a {user_term} "
        f"in a conversation. Respond concisely in-character."
    )



def ensure_results_dir(subdir: Optional[str] = None) -> Path:
    """Create and return the results directory (or a subdirectory)."""
    d = C.RESULTS_DIR if subdir is None else C.RESULTS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(data: dict, filename: str, subdir: Optional[str] = None) -> Path:
    """Save a dict as pretty-printed JSON in the results directory."""
    d = ensure_results_dir(subdir)
    path = d / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"Results saved to {path}")
    return path


def load_results(filename: str, subdir: Optional[str] = None) -> dict:
    """Load results JSON from the results directory."""
    d = C.RESULTS_DIR if subdir is None else C.RESULTS_DIR / subdir
    path = d / filename
    with open(path, "r") as f:
        return json.load(f)


def _json_default(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def clean_response(text: str) -> str:
    """Strip Llama special tokens from a generated response."""
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>"]:
        text = text.replace(tok, "")
    return text.strip()


def generate_unsteered(
    model, tokenizer, prompt: str, **gen_kwargs,
) -> str:
    """Generate a response without any RepE steering."""
    defaults = dict(C.GENERATION_DEFAULTS)
    defaults.update(gen_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=defaults.get("max_new_tokens", 128),
            do_sample=defaults.get("do_sample", False),
            repetition_penalty=defaults.get("repetition_penalty", 1.2),
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    response = full_text[len(prompt):].strip()
    return clean_response(response)
