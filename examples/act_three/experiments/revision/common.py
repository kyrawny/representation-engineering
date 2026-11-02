"""
Shared infrastructure for the revision experiments (plan task T1).

Nothing here changes the method: generation goes through the same
``EPASteerer.generate`` / ``generate_unsteered`` calls and the same
``GENERATION_DEFAULTS`` as experiment 08.  New conditions only differ in the
target, the system prompt or the injected vector.

Run scripts from the repo root as
``python -m examples.act_three.experiments.revision.<script>``.
"""

import importlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..analysis.common import load_hybrid_raw, load_json
from ..config import GENERATION_DEFAULTS, IDENTITY_PAIRS
from ..scenarios import SCENARIOS
from ..setup import clean_response, generate_unsteered

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REV_RESULTS = EXPERIMENTS_DIR / "results" / "revision"
ANALYSIS_RESULTS = EXPERIMENTS_DIR / "results" / "analysis"

MODEL_IDS = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
DIMS = ["evaluation", "potency", "activity"]

_CATEGORY = {s["id"]: s["category"] for s in SCENARIOS}


def rev_dir(model_short: str) -> Path:
    d = REV_RESULTS / model_short
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_08(model_short: str) -> dict:
    return load_hybrid_raw(model_short)


def scenario_category7(scenario_id: str) -> str:
    """Category in scenarios.py (7 categories; 'mixed' is one category)."""
    return _CATEGORY[scenario_id]


def subset_650(trials: List[dict]) -> List[dict]:
    """Stratified half-subset: within each of the 7 categories, sort scenario ids and
    keep every other one (indices 0, 2, 4, ...), all 13 identity pairs each.

    Following the plan's rule literally gives 52 scenarios (categories with an odd
    count keep ceil(n/2)), i.e. 676 trials rather than the nominal 650.
    """
    by_cat: Dict[str, set] = {}
    for t in trials:
        by_cat.setdefault(scenario_category7(t["scenario_id"]), set()).add(t["scenario_id"])
    keep = set()
    for cat, ids in by_cat.items():
        keep.update(sorted(ids)[::2])
    return [t for t in trials if t["scenario_id"] in keep]


def quick_subset(trials: List[dict]) -> List[dict]:
    """21 trials: 3 scenarios per category, one pair each (pairs cycle so all are exercised)."""
    pair_names = [p[0] for p in IDENTITY_PAIRS]
    by_cat: Dict[str, list] = {}
    for sid in sorted({t["scenario_id"] for t in trials}):
        by_cat.setdefault(scenario_category7(sid), []).append(sid)
    want = {}
    i = 0
    for cat in sorted(by_cat):
        for sid in by_cat[cat][:3]:
            want[sid] = pair_names[i % len(pair_names)]
            i += 1
    return [t for t in trials if want.get(t["scenario_id"]) == t["pair"]]


def trial_key(t: dict, cond: str) -> str:
    return f"{t['scenario_id']}|{t['pair']}|{cond}"


class JsonlLog:
    """Append-only JSONL log with a resume set (plan G5)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        self.records: List[dict] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a crash; regenerate
                    self.done.add(r["key"])
                    self.records.append(r)

    def append(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_np_default) + "\n")
            f.flush()
        self.done.add(rec["key"])
        self.records.append(rec)


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# Prompts and generation
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, system_prompt: Optional[str], user_msg: str,
                 history: Optional[List[dict]] = None) -> str:
    """Chat-template prompt; identical to ``format_chat_prompt`` when history is empty."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _gen_kwargs(max_new_tokens: Optional[int]) -> dict:
    kw = dict(GENERATION_DEFAULTS)
    if max_new_tokens is not None:
        kw["max_new_tokens"] = max_new_tokens
    return kw


def gen_plain(comp, system_prompt, user_msg, history=None, max_new_tokens=None) -> str:
    prompt = build_prompt(comp.tokenizer, system_prompt, user_msg, history)
    return clean_response(generate_unsteered(
        comp.model, comp.tokenizer, prompt, **_gen_kwargs(max_new_tokens)))


def gen_epa(comp, system_prompt, user_msg, target_epa: dict, history=None, max_new_tokens=None) -> str:
    prompt = build_prompt(comp.tokenizer, system_prompt, user_msg, history)
    return clean_response(comp.steerer.generate(
        prompt=prompt, target_epa=target_epa, **_gen_kwargs(max_new_tokens)))


def gen_vector(comp, system_prompt, user_msg, activations: dict, history=None,
               max_new_tokens=None) -> str:
    """Steer with an arbitrary {layer: tensor} dict through the same control pipeline."""
    prompt = build_prompt(comp.tokenizer, system_prompt, user_msg, history)
    out = comp.steerer.control_pipeline(
        prompt, activations=activations, batch_size=1, **_gen_kwargs(max_new_tokens))
    text = out[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return clean_response(text.strip())


def read_one(comp, text: str) -> dict:
    return comp.reader.read_epa(comp.rep_reading_pipeline, text)


def read_all(comp, texts: List[str], batch_size: int = 8) -> List[dict]:
    return comp.reader.read_epa_batch(comp.rep_reading_pipeline, texts, batch_size)


def constant_target(model_short: str) -> dict:
    return load_json(ANALYSIS_RESULTS / "act_targets.json")[model_short]["constant_target_B8"]


def make_record(t: dict, cond: str, system_prompt: str, user_msg: str, target_used,
                text: str, own_epa, n_tokens: int, seconds: float, **extra) -> dict:
    rec = {
        "key": trial_key(t, cond), "scenario_id": t["scenario_id"], "pair": t["pair"],
        "agent": t["agent_identity"], "user": t["user_identity"], "condition": cond,
        "system_prompt": system_prompt, "user_msg": user_msg, "target_epa_used": target_used,
        "text": text, "text500": text[:500], "own_epa": own_epa,
        "n_tokens": n_tokens, "gen_seconds": round(seconds, 3),
    }
    rec.update(extra)
    return rec


def count_tokens(comp, text: str) -> int:
    return len(comp.tokenizer(text, add_special_tokens=False)["input_ids"])


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0


def text_metrics(text: str) -> dict:
    """Surface metrics from experiment 07 (bigram repetition for the degeneracy check)."""
    m = importlib.import_module("examples.act_three.experiments.07_coherence_evaluation")
    return m.compute_text_metrics(text)


def self_perplexity(comp, texts: List[str]) -> List[float]:
    m = importlib.import_module("examples.act_three.experiments.07_coherence_evaluation")
    return m.compute_perplexity_self(comp.model, comp.tokenizer, texts)


def affective_prompt(agent: str, user: str, target: dict) -> str:
    m = importlib.import_module("examples.act_three.experiments.08_hybrid_steering")
    return m._make_affective_system_prompt(agent, user, target)


def load_components(model_short: str, load_steerer: bool = True):
    """Load model + readers + steerer with the orthogonalised directions the paper describes and the logged
    experiment-08 run used (verified: 10/10 logged texts reproduce exactly).  The flag is forced at runtime because
    models/llama3/config.json (added after that run) says false; config.json itself is not edited (plan G1)."""
    return load_components_ortho(model_short, True, load_steerer)


# ---------------------------------------------------------------------------
# Non-ACT targets (T2)
# ---------------------------------------------------------------------------

def shuffled_target_map(trials: List[dict]) -> Dict[str, dict]:
    """Derangement of ACT targets across identity pairs within each scenario (seed 0).

    Returns {f"{scenario_id}|{pair}": {"source_pair": ..., "target": {...}}}.  Computed over the
    full trial list so the mapping does not depend on which subset is run.
    """
    rng = np.random.default_rng(0)
    by_scn: Dict[str, List[dict]] = {}
    for t in trials:
        by_scn.setdefault(t["scenario_id"], []).append(t)
    out = {}
    for sid in sorted(by_scn):
        grp = by_scn[sid]
        idx = np.arange(len(grp))
        perm = idx.copy()
        while len(idx) > 1 and np.any(perm == idx):
            perm = rng.permutation(idx)
        for i, j in zip(idx, perm):
            out[f"{sid}|{grp[i]['pair']}"] = {"source_pair": grp[j]["pair"],
                                              "target": dict(grp[j]["target_epa"])}
    return out


def idonly_target(t: dict, coefficients) -> dict:
    """ACT target for the true identities with the user behaviour at the EPA origin (no read step)."""
    from examples.act_three import EPA, get_response_epa_for_deflection_minimization
    a, u = t["agent_identity_epa"], t["user_identity_epa"]
    res = get_response_epa_for_deflection_minimization(
        agent_identity=EPA(e=a["evaluation"], p=a["potency"], a=a["activity"]),
        user_identity=EPA(e=u["evaluation"], p=u["potency"], a=u["activity"]),
        user_behavior_epa=EPA(e=0.0, p=0.0, a=0.0),
        coefficients=coefficients)
    return {"evaluation": res.e, "potency": res.p, "activity": res.a}


def load_components_ortho(model_short: str, ortho: bool, load_steerer: bool = True):
    """Like ``load_components`` but forces the orthogonalised-directions flag at runtime
    (config.json is NOT edited).  ``ortho=None`` keeps the config.json value."""
    from .. import config as C
    from ..setup import load_experiment_components
    C.set_model(MODEL_IDS[model_short])
    if ortho is not None:
        C.USE_ORTHOGONAL_DIRECTIONS = ortho
    return load_experiment_components(load_steerer=load_steerer, model_name=None)
