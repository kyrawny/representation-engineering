"""
CAA (Rimsky et al., 2024) sycophancy vector: extraction helpers and injection through the
same ``rep-control`` pipeline that ACT-RepE uses (plan T3b).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import common as R

CAA_DATA = R.REV_RESULTS / "data" / "caa" / "generate_dataset.json"
N_EXTRACT, N_HELDOUT = 500, 50
LAYER_CANDIDATES = [10, 12, 14, 16]          # 0-based decoder layers


def load_items():
    items = json.load(open(CAA_DATA, encoding="utf-8"))
    rng = random.Random(0)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    ext = [items[i] for i in idx[:N_EXTRACT]]
    held = [items[i] for i in idx[N_EXTRACT:N_EXTRACT + N_HELDOUT]]
    return ext, held


def _letter(s: str) -> str:
    return s.strip().strip("()").strip()[0]


def prefix_ids(comp, question: str):
    """Chat prompt for the question, ending with the assistant turn opened by '('."""
    prompt = R.build_prompt(comp.tokenizer, None, question) + "("
    return comp.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]


def letter_id(comp, letter: str) -> int:
    ids = comp.tokenizer.encode(letter, add_special_tokens=False)
    return ids[-1]


@torch.no_grad()
def letter_hidden_states(comp, question: str, letter: str) -> torch.Tensor:
    """Residual stream (all decoder layers) at the letter token: [n_layers, H] (float32, cpu)."""
    ids = prefix_ids(comp, question)
    ids = torch.cat([ids, torch.tensor([[letter_id(comp, letter)]])], dim=1).to(comp.model.device)
    out = comp.model(ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states[1:]])   # hidden_states[l+1] = output of layer l
    return hs.float().cpu()


def extract_vectors(comp, items):
    pos, neg = [], []
    for it in items:
        m, n = _letter(it["answer_matching_behavior"]), _letter(it["answer_not_matching_behavior"])
        pos.append(letter_hidden_states(comp, it["question"], m))
        neg.append(letter_hidden_states(comp, it["question"], n))
    return (torch.stack(pos).mean(0) - torch.stack(neg).mean(0))        # [n_layers, H]


def neg_id(comp, layer0: int) -> int:
    return layer0 - comp.model.config.num_hidden_layers


def zero_activations(comp):
    H = comp.model.config.hidden_size
    return {l: torch.zeros(1, H, dtype=comp.model.dtype, device=comp.model.device)
            for l in comp.steerer.all_layers}


def vector_activations(comp, layer0: int, vec: torch.Tensor, mult: float) -> dict:
    acts = zero_activations(comp)
    acts[neg_id(comp, layer0)] = (mult * vec).to(comp.model.dtype).to(comp.model.device).reshape(1, -1)
    return acts


@torch.no_grad()
def match_logit_diff(comp, items, activations=None) -> float:
    """Mean log P(matching letter) - log P(non-matching letter) at the position after '('."""
    pipe = comp.steerer.control_pipeline
    vals = []
    for it in items:
        m, n = _letter(it["answer_matching_behavior"]), _letter(it["answer_not_matching_behavior"])
        ids = prefix_ids(comp, it["question"]).to(comp.model.device)
        if activations is not None:
            pipe.wrapped_model.reset()
            pipe.wrapped_model.set_controller(pipe.layers, activations, pipe.block_name)
        logits = comp.model(ids).logits[0, -1].float()
        if activations is not None:
            pipe.wrapped_model.reset()
        lp = torch.log_softmax(logits, -1)
        vals.append((lp[letter_id(comp, m)] - lp[letter_id(comp, n)]).item())
    return float(np.mean(vals))


def load_caa(comp, model_short: str):
    d = R.rev_dir(model_short)
    meta = json.load(open(d / "caa_meta.json", encoding="utf-8"))
    vecs = torch.load(d / "caa_vectors.pt")["vectors"]
    layer0, mult = meta["chosen_layer0"], meta["chosen_multiplier"]
    return vector_activations(comp, layer0, vecs[layer0], mult), meta
