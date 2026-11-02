"""
Shared helpers for the 70B scale test.

Design rule: the *method* (direction extraction, reader, ACT target, steering
formula, prompts, decoding) is imported unchanged from ``examples.act_three``.
This module only adds what a large multi-GPU run needs: configurable model
loading, a faster batched reader that is numerically equivalent to
``EPAReader.read_epa``, resumable logs, sharding and run metadata.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import settings as S

if str(S.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(S.REPO_ROOT))

import torch  # noqa: E402

DIMS = S.DIMS


# ===========================================================================
# Model loading (environment-configurable; same defaults as the paper code)
# ===========================================================================

def model_load_kwargs() -> Dict[str, Any]:
    """Keyword arguments for ``AutoModelForCausalLM.from_pretrained``.

    Environment variables:
        ACT70B_DTYPE       float16 (paper default) | bfloat16
        ACT70B_QUANT       none (default) | 8bit | 4bit   (bitsandbytes)
        ACT70B_MAX_MEMORY  e.g. "0:76GiB,1:76GiB,2:76GiB,3:76GiB"
    """
    dtype = os.environ.get("ACT70B_DTYPE", "float16")
    quant = os.environ.get("ACT70B_QUANT", "none").lower()
    max_mem = os.environ.get("ACT70B_MAX_MEMORY", "").strip()
    kw: Dict[str, Any] = {"device_map": "auto", "torch_dtype": getattr(torch, dtype)}
    if max_mem:
        mm = {}
        for part in max_mem.split(","):
            k, v = part.split(":", 1)
            mm[int(k) if k.strip().isdigit() else k.strip()] = v.strip()
        kw["max_memory"] = mm
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
    elif quant != "none":
        raise ValueError(f"ACT70B_QUANT must be none|8bit|4bit, got {quant!r}")
    return kw


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token          # as in experiments/setup.py
    print(f"[load] {model_name} in {time.time() - t0:.0f}s; "
          f"device map spans {len(set(getattr(model, 'hf_device_map', {'': 0}).values()))} device(s)")
    return model, tokenizer


# ===========================================================================
# Components (mirrors experiments/setup.py::load_experiment_components)
# ===========================================================================

@dataclass
class Components:
    model: Any
    tokenizer: Any
    rep_readers: Dict[str, Any]
    reader: Any                 # EPAReader
    fast_reader: Any            # FastReader
    steerer: Optional[Any]      # EPASteerer
    rep_pipeline: Any
    act_coefficients: Any
    identities_df: Any
    model_name: str
    strength: float
    config: Dict[str, Any]


def read_model_config(model_name: str) -> Dict[str, Any]:
    p = S.model_dir(model_name) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def directions_path(model_name: str, cfg: Dict[str, Any]) -> Path:
    d = S.model_dir(model_name)
    use_ortho = cfg.get("use_orthogonal_directions", S.USE_ORTHOGONAL_DIRECTIONS)
    p = d / ("epa_directions_ortho.pkl" if use_ortho else "epa_directions.pkl")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run stage 1 (extract + tune) first.")
    return p


def load_components(model_name: str, need_steerer: bool = True,
                    strength: Optional[float] = None,
                    model_and_tokenizer=None) -> Components:
    from transformers import pipeline as hf_pipeline
    from repe import repe_pipeline_registry
    import pandas as pd
    from examples.act_three import EPAReader, EPASteerer, load_directions, get_default_coefficients

    cfg = read_model_config(model_name)
    model, tokenizer = model_and_tokenizer or load_model(model_name)
    repe_pipeline_registry()
    rep_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    saved = load_directions(str(directions_path(model_name, cfg)))
    rep_readers = saved["rep_readers"]
    tuning = S.model_dir(model_name) / "epa_reading_tuning_v2_results.json"
    reader = EPAReader.from_tuning_results(str(tuning), rep_readers,
                                           method=cfg.get("reading_method", S.READING_METHOD))

    if strength is None:
        strength = float(cfg.get("strength", 1.0))
    steerer = None
    if need_steerer:
        steering_configs = {}
        for dim in DIMS:
            dim_cfg = reader.config.dimensions[dim]
            layers = sorted(dim_cfg.selected_layers.keys())
            signs = {}
            for layer in layers:
                sv = rep_readers[dim].direction_signs.get(layer, 1)
                signs[layer] = float(sv.item() if hasattr(sv, "item") else np.asarray(sv).ravel()[0])
            steering_configs[dim] = {"layers": layers, "signs": signs,
                                     "base_coeff": S.REFERENCE_COEFFICIENTS[dim]}
        steerer = EPASteerer(model=model, tokenizer=tokenizer, rep_readers=rep_readers,
                             steering_configs=steering_configs,
                             clamp_limit=S.REFERENCE_CLAMP_LIMIT)
        apply_strength(steerer, strength)

    return Components(
        model=model, tokenizer=tokenizer, rep_readers=rep_readers, reader=reader,
        fast_reader=FastReader(reader, rep_pipeline), steerer=steerer,
        rep_pipeline=rep_pipeline, act_coefficients=get_default_coefficients(),
        identities_df=pd.read_csv(S.IDENTITIES_CSV), model_name=model_name,
        strength=float(strength), config=cfg)


def apply_strength(steerer, s: float) -> None:
    """Scale coefficients AND clamp limit by s.

    EPASteerer computes  m = L * tanh(c * t * sign / L)  per layer.  With
    c -> s*c and L -> s*L this becomes  s * L * tanh(c * t * sign / L),
    i.e. exactly s times the paper's perturbation for the same target.
    """
    for dim in DIMS:
        steerer.steering_configs[dim]["base_coeff"] = s * S.REFERENCE_COEFFICIENTS[dim]
    steerer.clamp_limit = s * S.REFERENCE_CLAMP_LIMIT


class strength_context:
    """Temporarily use another strength (e.g. the literal s = 1 conditions)."""

    def __init__(self, steerer, s: float, restore: float):
        self.steerer, self.s, self.restore = steerer, s, restore

    def __enter__(self):
        apply_strength(self.steerer, self.s)

    def __exit__(self, *exc):
        apply_strength(self.steerer, self.restore)


# ===========================================================================
# Faster reader, numerically equivalent to EPAReader.read_epa
# ===========================================================================

class FastReader:
    """Reads E, P and A from ONE forward pass per text (the original runs three).

    Replicates ``RepReader.transform`` exactly: subtract the PCA training mean
    (``H_train_means``), project onto ``directions[layer][0] / ||.||`` in
    float32, then apply the reader's layer weights, signs and calibration.
    ``s00_preflight`` checks agreement with ``EPAReader.read_epa``.
    """

    def __init__(self, reader, rep_pipeline, batch_size: int = 8):
        self.reader, self.pipe, self.batch_size = reader, rep_pipeline, batch_size
        self.layers = sorted({l for d in DIMS
                              for l in reader.config.dimensions[d].selected_layers})
        self.proj = {}
        for d in DIMS:
            rr = reader.rep_readers[d]
            for l in reader.config.dimensions[d].selected_layers:
                v = np.asarray(rr.directions[l], dtype=np.float32)
                v = v[0] if v.ndim > 1 else v
                v = torch.from_numpy(v.copy())
                mean = None
                if hasattr(rr, "H_train_means"):
                    mean = torch.from_numpy(
                        np.asarray(rr.H_train_means[l], dtype=np.float32).reshape(-1).copy())
                self.proj[(d, l)] = (v / v.norm(), mean)

    def _hidden(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        from examples.act_three.prompt_formatting import format_for_reading
        formatted = [format_for_reading(t) for t in texts]
        out = self.pipe(formatted, hidden_layers=self.layers, rep_reader=None,
                        batch_size=self.batch_size, padding=True, truncation=True)
        rows: Dict[int, list] = {l: [] for l in self.layers}
        for item in out:
            for l in self.layers:
                h = torch.as_tensor(item[l]).float()
                rows[l].append(h.reshape(-1, h.shape[-1]))
        H = {l: torch.cat(rows[l], dim=0) for l in self.layers}
        assert all(H[l].shape[0] == len(texts) for l in self.layers), "reader output size mismatch"
        return H

    def read(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []
        texts = [t if t.strip() else " " for t in texts]
        H = self._hidden(texts)
        res = [dict() for _ in texts]
        for d in DIMS:
            cfg = self.reader.config.dimensions[d]
            total = torch.zeros(len(texts))
            wsum = 0.0
            for l, w in cfg.selected_layers.items():
                v, mean = self.proj[(d, l)]
                h = H[l] - mean if mean is not None else H[l]
                total += float(w) * float(cfg.layer_signs.get(l, 1.0)) * (h @ v)
                wsum += float(w)
            raw = total / wsum if wsum > 0 else total * 0
            cal = cfg.calibration_slope * raw + cfg.calibration_intercept
            for i in range(len(texts)):
                res[i][d] = float(cal[i])
        return res


def safe_read_batch_size(model_name: str, requested: int) -> int:
    """Use batch size 1 if preflight found batched reading outside tolerance."""
    p = S.results_dir(model_name) / "preflight.json"
    if p.exists():
        fr = json.loads(p.read_text()).get("checks", {}).get("fast_reader")
        if fr and not fr.get("ok", True):
            print("[reader] preflight: batched reading outside tolerance -> batch size 1")
            return 1
    return requested


def verify_fast_reader(comp: Components, texts: List[str], tol: float = 0.05) -> Dict[str, Any]:
    """Compare FastReader with the original EPAReader.read_epa (batch size 1)."""
    fast = comp.fast_reader.read(texts)
    slow = [comp.reader.read_epa(comp.rep_pipeline, t) for t in texts]
    diffs = [abs(f[d] - s[d]) for f, s in zip(fast, slow) for d in DIMS]
    out = {"n": len(texts), "max_abs_diff": float(max(diffs)), "mean_abs_diff": float(np.mean(diffs)),
           "tolerance": tol, "ok": bool(max(diffs) <= tol)}
    if not out["ok"]:
        comp.fast_reader.batch_size = 1
        fast1 = comp.fast_reader.read(texts)
        d1 = [abs(f[d] - s[d]) for f, s in zip(fast1, slow) for d in DIMS]
        out["batch1_max_abs_diff"] = float(max(d1))
        out["ok_batch1"] = bool(max(d1) <= tol)
    return out


# ===========================================================================
# Method imports (unchanged code paths)
# ===========================================================================

def method_api():
    """Return the functions the paper's experiment 08 uses, unchanged."""
    from examples.act_three import EPA, get_response_epa_for_deflection_minimization
    from examples.act_three.model_registry import format_chat_prompt
    from examples.act_three.experiments.config import IDENTITY_PAIRS, GENERATION_DEFAULTS
    from examples.act_three.experiments.setup import (
        make_system_prompt, generate_unsteered, clean_response, get_identity_epa)
    from examples.act_three.experiments.scenarios import get_scenarios
    exp08 = importlib.import_module("examples.act_three.experiments.08_hybrid_steering")
    return dict(EPA=EPA, target_fn=get_response_epa_for_deflection_minimization,
                format_chat_prompt=format_chat_prompt, IDENTITY_PAIRS=IDENTITY_PAIRS,
                GENERATION_DEFAULTS=GENERATION_DEFAULTS, make_system_prompt=make_system_prompt,
                affective_prompt=exp08._make_affective_system_prompt,
                generate_unsteered=generate_unsteered, clean_response=clean_response,
                get_identity_epa=get_identity_epa, get_scenarios=get_scenarios)


def coherence_api():
    m = importlib.import_module("examples.act_three.experiments.07_coherence_evaluation")
    return m.compute_perplexity_self, m.compute_text_metrics, getattr(m, "compute_perplexity_gpt2", None)


def compute_target(api, comp: Components, agent: str, user: str, user_epa: Dict[str, float]):
    a = api["get_identity_epa"](comp.identities_df, agent)
    u = api["get_identity_epa"](comp.identities_df, user)
    b = api["EPA"](e=user_epa["evaluation"], p=user_epa["potency"], a=user_epa["activity"])
    t = api["target_fn"](agent_identity=a, user_identity=u, user_behavior_epa=b,
                         coefficients=comp.act_coefficients)
    return ({"evaluation": a.e, "potency": a.p, "activity": a.a},
            {"evaluation": u.e, "potency": u.p, "activity": u.a},
            {"evaluation": t.e, "potency": t.p, "activity": t.a})


# ===========================================================================
# Trials, subsets, sharding
# ===========================================================================

def all_trials(api) -> List[Dict[str, Any]]:
    """Same order as experiment 08: scenario-major, then identity pairs."""
    trials = []
    for s in api["get_scenarios"]():
        for pair, agent, user in api["IDENTITY_PAIRS"]:
            trials.append({"index": len(trials), "key": f"{s['id']}|{pair}",
                           "scenario_id": s["id"], "category": s["category"],
                           "user_statement": s["text"], "pair": pair,
                           "agent_identity": agent, "user_identity": user})
    return trials


def subset(trials: List[Dict[str, Any]], which: str) -> List[Dict[str, Any]]:
    if which == "all":
        return trials
    by_cat: Dict[str, List[str]] = {}
    for t in trials:
        ids = by_cat.setdefault(t["category"], [])
        if t["scenario_id"] not in ids:
            ids.append(t["scenario_id"])
    if which == "half":       # every other scenario within each category (650 trials)
        keep = {sid for ids in by_cat.values() for sid in sorted(ids)[::2]}
        return [t for t in trials if t["scenario_id"] in keep]
    if which == "calibration":
        keep = {sid for ids in by_cat.values() for sid in sorted(ids)[:S.CALIBRATION_PER_CATEGORY]}
        return [t for t in trials if t["scenario_id"] in keep and t["pair"] in S.CALIBRATION_PAIRS]
    raise ValueError(which)


def shard(trials: List[Dict[str, Any]], index: int, num: int) -> List[Dict[str, Any]]:
    if not 0 <= index < num:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [t for t in trials if t["index"] % num == index]


# ===========================================================================
# Resumable JSONL logs
# ===========================================================================

def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class JsonlLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.done.add(json.loads(line)["key"])
                except Exception:
                    pass  # tolerate a truncated last line after a crash

    def append(self, rec: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_np_default, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add(rec["key"])


def read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ===========================================================================
# Run metadata
# ===========================================================================

def env_report() -> Dict[str, Any]:
    import transformers
    rep: Dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "env": {k: os.environ.get(k) for k in ("ACT70B_DTYPE", "ACT70B_QUANT", "ACT70B_MAX_MEMORY")},
    }
    for mod in ("accelerate", "tokenizers", "bitsandbytes", "sklearn", "scipy", "pandas", "numpy"):
        try:
            rep[mod] = importlib.import_module(mod).__version__
        except Exception:
            rep[mod] = None
    if torch.cuda.is_available():
        rep["gpus"] = [{"name": torch.cuda.get_device_name(i),
                        "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                       for i in range(torch.cuda.device_count())]
    try:
        rep["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=S.REPO_ROOT, text=True).strip()
        rep["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "examples/act_three", "repe"],
            cwd=S.REPO_ROOT, text=True).strip())
    except Exception:
        rep["git_commit"] = None
    lock = S.THIS_DIR / "requirements-lock.txt"
    if lock.exists():
        pinned = {}
        for line in lock.read_text().splitlines():
            if "==" in line:
                k, v = line.split("==", 1)
                pinned[k.strip().lower()] = v.strip()
        installed = {"torch": rep["torch"], "transformers": rep["transformers"],
                     "tokenizers": rep.get("tokenizers"), "accelerate": rep.get("accelerate"),
                     "scikit-learn": rep.get("sklearn"), "numpy": rep.get("numpy"),
                     "python": rep["python"]}
        rep["version_mismatches"] = {
            k: {"pinned": pinned[k], "installed": v} for k, v in installed.items()
            if k in pinned and v and not str(v).split("+")[0] == pinned[k].split("+")[0]}
    return rep


def geo_mean(x: Iterable[float]) -> float:
    x = [v for v in x if v and math.isfinite(v) and v > 0]
    return float(math.exp(np.mean(np.log(x)))) if x else float("nan")
