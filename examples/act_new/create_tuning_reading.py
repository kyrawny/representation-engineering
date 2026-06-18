"""Generate hyperparameter_tuning_reading.ipynb
Per-layer reading weight tuning using representation reading (not Likert).
"""
import json

def S(code):
    lines = code.split('\n')
    if lines and lines[0].strip() == '': lines = lines[1:]
    while lines and lines[-1].strip() == '': lines.pop()
    return [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]

def code_cell(cid, code):
    return {"cell_type":"code","execution_count":None,"id":cid,"metadata":{},"outputs":[],"source":S(code)}

def md_cell(cid, md):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":S(md)}

cells = []

# ── Header ──
cells.append(md_cell("header", """# EPA Hyperparameter Tuning via Representation Reading

Uses **representation reading** (projecting hidden states onto extracted directions) to find
optimal per-layer weights for reading EPA values from text.

Much faster than Likert-based tuning: one forward pass per utterance extracts scores from ALL layers.

**Phase 1** -- Individual layer sweep: correlation of each layer's raw reading with ground-truth EPA.
**Simple Independent** -- Combine effective layers with uniform weights.
**Phase 2** -- Greedy forward selection with per-layer weight optimization.

**Prerequisites:** `epa_tuning_dataset.json` (from `hyperparameter_tuning.ipynb` Cell 3) and
`epa_directions.pkl` must exist in the working directory.

**Estimated time:** ~5-10 minutes total (vs ~50 min for Likert-based)."""))

# ── Cell 1: Setup ──
cells.append(code_cell("cell-setup", r"""# === Cell 1: Setup & Imports ===
import sys; sys.path.append('../..')
import os, json, pickle
from datetime import datetime
import torch, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry
repe_pipeline_registry()
from examples.act_new.act_core import EPA
from examples.act_new.utils import format_for_reading, format_llama3_prompt
print("All imports OK.")"""))

# ── Cell 2: Load Model & Directions ──
cells.append(code_cell("cell-load-model", r"""# === Cell 2: Load Model & Directions ===
with open("epa_directions.pkl", 'rb') as f:
    directions_data = pickle.load(f)
rep_readers = directions_data['rep_readers']
hidden_layers = directions_data['hidden_layers']
model_name = directions_data['model_name']

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Create reading pipeline
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

print(f"Loaded: {model_name}, {len(hidden_layers)} layers ({hidden_layers[0]}..{hidden_layers[-1]})")

# Verify layer keys
sample_reader = list(rep_readers.values())[0]
print(f"RepReader direction layers: {sorted(sample_reader.directions.keys())[:5]}...")"""))

# ── Cell 3: Load Dataset ──
cells.append(code_cell("cell-load-dataset", r"""# === Cell 3: Load Dataset & Prepare Ground Truth ===
DATASET_PATH = "epa_tuning_dataset.json"
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
utterances = dataset['utterances']
DIMENSIONS = ["evaluation", "potency", "activity"]
DIM_KEY = {"evaluation": "e", "potency": "p", "activity": "a"}

# Ground truth EPA values (continuous, from ACT dictionary)
ground_truth = {dim: [] for dim in DIMENSIONS}
for u in utterances:
    for dim in DIMENSIONS:
        ground_truth[dim].append(u['target_epa'][DIM_KEY[dim]])

print(f"Loaded {len(utterances)} utterances")
for dim in DIMENSIONS:
    vals = ground_truth[dim]
    print(f"  {dim:12s}: range=[{min(vals):.2f}, {max(vals):.2f}], mean={np.mean(vals):.2f}")"""))

# ── Cell 4: Extract Per-Layer Readings ──
cells.append(code_cell("cell-extract", r"""# === Cell 4: Extract Per-Layer Reading Scores for All Utterances ===
# One forward pass per utterance extracts readings from ALL layers simultaneously.
# This is the key speed advantage over Likert-based tuning.

all_layers = sorted(rep_readers[DIMENSIONS[0]].directions.keys())
print(f"Extracting readings from {len(all_layers)} layers for {len(utterances)} utterances...")

# Format utterances for reading
formatted_texts = []
for u in utterances:
    formatted_texts.append(format_for_reading(u['text']))

# Extract per-layer scores for each dimension
# raw_scores[dim][layer] = array of shape (n_utterances,)
raw_scores = {}
for dim in DIMENSIONS:
    reader = rep_readers[dim]
    dim_scores = rep_reading_pipeline(
        formatted_texts,
        hidden_layers=all_layers,
        rep_reader=reader,
        batch_size=8,
        padding=True,
        truncation=True,
    )
    # dim_scores is a list of dicts, one per utterance
    # each dict maps layer -> scalar score
    layer_arrays = {layer: [] for layer in all_layers}
    for utt_scores in dim_scores:
        for layer in all_layers:
            layer_arrays[layer].append(float(utt_scores[layer]))
    raw_scores[dim] = {layer: np.array(vals) for layer, vals in layer_arrays.items()}
    print(f"  {dim}: extracted {len(all_layers)} layers x {len(utterances)} utterances")

print("\nExtraction complete!")
# Quick sanity: show range of readings for first layer
for dim in DIMENSIONS:
    first_layer = all_layers[0]
    vals = raw_scores[dim][first_layer]
    print(f"  {dim} layer {first_layer}: range=[{vals.min():.3f}, {vals.max():.3f}]")"""))

# ── Cell 5: Phase 1 Individual Layer Sweep ──
cells.append(code_cell("cell-phase1", r"""# === Cell 5: Phase 1 -- Individual Layer Correlation ===
# For each layer, compute Spearman correlation between raw reading and ground truth.
# No coefficients needed here -- just measuring which layers have the best signal.

gt_arrays = {dim: np.array(ground_truth[dim]) for dim in DIMENSIONS}

phase1 = {dim: {} for dim in DIMENSIONS}
for dim in DIMENSIONS:
    for layer in all_layers:
        readings = raw_scores[dim][layer]
        rho, pval = spearmanr(readings, gt_arrays[dim])
        phase1[dim][layer] = {"rho": float(rho), "pval": float(pval)}

# Print ranked results
print("Phase 1: Per-Layer Spearman Correlation with Ground-Truth EPA")
for dim in DIMENSIONS:
    ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]['rho']), reverse=True)
    print(f"\n{'='*60}")
    print(f"{dim.upper()} -- Top 15 layers:")
    print(f"{'='*60}")
    for layer, info in ranked[:15]:
        sign = "+" if info['rho'] > 0 else "-"
        bar = "#" * int(abs(info['rho']) * 40)
        print(f"  Layer {layer:4d}: rho={info['rho']:+.3f} (p={info['pval']:.4f}) {bar}")"""))

# ── Cell 6: Phase 1 Visualization ──
cells.append(code_cell("cell-phase1-viz", r"""# === Cell 6: Phase 1 Visualization ===

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for idx, dim in enumerate(DIMENSIONS):
    sorted_layers = sorted(all_layers)
    rhos = [phase1[dim][l]['rho'] for l in sorted_layers]
    colors = ['#2196F3' if r > 0 else '#F44336' for r in rhos]
    axes[idx].barh([str(l) for l in sorted_layers], rhos, color=colors)
    axes[idx].set_xlabel("Spearman rho")
    axes[idx].set_ylabel("Layer (neg idx)")
    axes[idx].set_title(f"{dim.capitalize()}")
    axes[idx].axvline(0, color='gray', ls='--', alpha=0.5)
    axes[idx].invert_yaxis()
plt.suptitle("Phase 1: Per-Layer Correlation with Ground-Truth EPA", fontsize=13)
plt.tight_layout()
plt.savefig("reading_phase1.png", dpi=150, bbox_inches='tight'); plt.show()
print("Saved reading_phase1.png")"""))

# ── Cell 7: Simple Independent (Uniform Weight Average) ──
cells.append(code_cell("cell-simple-indep", r"""# === Cell 7: Simple Independent -- Uniform Weighted Average ===
# Average the readings from the top-K layers (by absolute correlation).
# Also test weighted average using sign-corrected readings.

TOP_K_VALUES = [1, 3, 5, 10, 15, 20, len(all_layers)]
INTERFERENCE_PENALTY = 0.5

print("Simple Independent: Uniform average of top-K layers")
print(f"{'K':>4s}  {'Eval rho':>10s}  {'Pote rho':>10s}  {'Acti rho':>10s}")
print("-" * 50)

simple_results = {}
for K in TOP_K_VALUES:
    combined = {}
    for dim in DIMENSIONS:
        # Select top-K layers by absolute correlation
        ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]['rho']), reverse=True)[:K]
        top_layers = [l for l, _ in ranked]
        # Average readings (sign-corrected so positive = higher EPA)
        layer_readings = []
        for layer in top_layers:
            sign = 1.0 if phase1[dim][layer]['rho'] > 0 else -1.0
            layer_readings.append(sign * raw_scores[dim][layer])
        avg_reading = np.mean(layer_readings, axis=0)
        rho, _ = spearmanr(avg_reading, gt_arrays[dim])
        combined[dim] = {"rho": float(rho), "top_layers": top_layers}
    simple_results[K] = combined
    print(f"{K:4d}  {combined['evaluation']['rho']:+10.3f}  "
          f"{combined['potency']['rho']:+10.3f}  "
          f"{combined['activity']['rho']:+10.3f}")

print("\nNote: |rho| may decrease with more layers if weak layers add noise.")"""))

# ── Cell 8: Phase 2 Greedy with Per-Layer Weights ──
cells.append(code_cell("cell-phase2", r"""# === Cell 8: Phase 2 -- Greedy Forward Selection with Per-Layer Weights ===
# Start from the single best layer, greedily add layers with optimized weights.
# "Weight" here scales each layer's contribution to the weighted average.

TOP_K = 15         # candidate layers from Phase 1
MAX_LAYERS = 10    # max layers to select
MIN_IMPROVEMENT = 0.002  # min rho improvement to accept a layer
WEIGHT_CANDIDATES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def compute_weighted_reading(selected_lw, dim):
    # Compute weighted average of sign-corrected layer readings.
    # selected_lw: dict {layer: weight}
    total = np.zeros(len(utterances))
    total_weight = 0.0
    for layer, weight in selected_lw.items():
        sign = 1.0 if phase1[dim][layer]['rho'] > 0 else -1.0
        total += weight * sign * raw_scores[dim][layer]
        total_weight += weight
    return total / total_weight if total_weight > 0 else total

def eval_reading_combo(selected_lw, steer_dim):
    # Evaluate a layer+weight combination across all dimensions.
    corrs = {}; delts = {}
    for dim in DIMENSIONS:
        # Use same layers but recompute weighted avg per dimension
        avg = compute_weighted_reading(selected_lw, dim)
        rho, _ = spearmanr(avg, gt_arrays[dim])
        corrs[dim] = float(rho)
        # Delta from single best layer baseline
        best_single = max(phase1[dim].items(), key=lambda x: abs(x[1]['rho']))
        delts[dim] = float(rho - abs(best_single[1]['rho']))
    others = [d for d in DIMENSIONS if d != steer_dim]
    cross_penalty = np.mean([max(0, -delts[d]) for d in others])
    score = corrs[steer_dim] - INTERFERENCE_PENALTY * cross_penalty
    return {"correlations": corrs, "deltas": delts, "score": score}

print(f"Phase 2: Greedy selection (top-{TOP_K}, max {MAX_LAYERS} layers)")
print(f"Weight candidates: {WEIGHT_CANDIDATES}")
print("")

greedy_results = {}
for steer_dim in DIMENSIONS:
    print(f"\n{'='*60}")
    print(f"Greedy: {steer_dim.upper()}")
    print(f"{'='*60}")

    # Rank by absolute correlation on the steered dimension
    ranked = sorted(phase1[steer_dim].items(),
                    key=lambda x: abs(x[1]['rho']), reverse=True)[:TOP_K]
    candidates = [l for l, _ in ranked]
    print(f"Candidate layers: {candidates}")

    # Start with best single layer
    best_layer = candidates[0]
    selected = {best_layer: 1.0}
    ev = eval_reading_combo(selected, steer_dim)
    cur_score = ev['score']
    print(f"Step 0: layer {best_layer} w=1.0 => rho={ev['correlations'][steer_dim]:+.3f}, score={cur_score:.4f}")

    history = [{"step": 0, "selected": dict(selected), "score": cur_score, **ev}]

    for step in range(1, MAX_LAYERS):
        remaining = [l for l in candidates if l not in selected]
        if not remaining: break
        best_add_score = cur_score
        best_add_layer = None
        best_add_weight = None
        for cand in remaining:
            for w in WEIGHT_CANDIDATES:
                trial = {**selected, cand: w}
                ev = eval_reading_combo(trial, steer_dim)
                if ev['score'] > best_add_score:
                    best_add_score = ev['score']
                    best_add_layer = cand
                    best_add_weight = w
        if best_add_layer is None or (best_add_score - cur_score) < MIN_IMPROVEMENT:
            print(f"Step {step}: no improvement >= {MIN_IMPROVEMENT}, stopping.")
            break
        selected[best_add_layer] = best_add_weight
        cur_score = best_add_score
        ev = eval_reading_combo(selected, steer_dim)
        print(f"Step {step}: +layer {best_add_layer} w={best_add_weight} => "
              f"rho={ev['correlations'][steer_dim]:+.3f}, score={cur_score:.4f}")
        history.append({"step": step, "selected": dict(selected), "score": cur_score, **ev})

    greedy_results[steer_dim] = {"selected": dict(selected), "history": history, "score": cur_score}
    print(f"\nFinal: {len(selected)} layers, score={cur_score:.4f}")
    for l, w in sorted(selected.items()):
        sign_str = "+" if phase1[steer_dim][l]['rho'] > 0 else "-"
        print(f"  Layer {l:4d}: weight={w}, sign={sign_str}, "
              f"individual rho={phase1[steer_dim][l]['rho']:+.3f}")

print("\nPhase 2 complete!")"""))

# ── Cell 9: Final Evaluation ──
cells.append(code_cell("cell-final-eval", r"""# === Cell 9: Final Evaluation & Comparison ===

print("=== Final Results ===\n")

# Compare: single best layer vs simple top-5 vs greedy
methods = {}

# 1. Single best layer
methods["Single Best"] = {}
for dim in DIMENSIONS:
    best = max(phase1[dim].items(), key=lambda x: abs(x[1]['rho']))
    methods["Single Best"][dim] = {"rho": abs(best[1]['rho']), "layers": {best[0]: 1.0}}

# 2. Simple top-5
methods["Top-5 Avg"] = {}
for dim in DIMENSIONS:
    ranked = sorted(phase1[dim].items(), key=lambda x: abs(x[1]['rho']), reverse=True)[:5]
    top5 = {l: 1.0 for l, _ in ranked}
    avg = compute_weighted_reading(top5, dim)
    rho, _ = spearmanr(avg, gt_arrays[dim])
    methods["Top-5 Avg"][dim] = {"rho": float(rho), "layers": top5}

# 3. Greedy
methods["Greedy"] = {}
for dim in DIMENSIONS:
    sel = greedy_results[dim]['selected']
    avg = compute_weighted_reading(sel, dim)
    rho, _ = spearmanr(avg, gt_arrays[dim])
    methods["Greedy"][dim] = {"rho": float(rho), "layers": sel}

# Print comparison table
print(f"{'Method':<15s}  {'Eval':>8s}  {'Pote':>8s}  {'Acti':>8s}")
print("-" * 45)
for name, dims in methods.items():
    print(f"{name:<15s}  {dims['evaluation']['rho']:+8.3f}  "
          f"{dims['potency']['rho']:+8.3f}  "
          f"{dims['activity']['rho']:+8.3f}")

# Cross-dimension check for greedy
print("\n--- Greedy Cross-Dimension Interference ---")
for steer_dim in DIMENSIONS:
    sel = greedy_results[steer_dim]['selected']
    print(f"\n{steer_dim.upper()} layers applied to read other dimensions:")
    for dim in DIMENSIONS:
        avg = compute_weighted_reading(sel, dim)
        rho, _ = spearmanr(avg, gt_arrays[dim])
        marker = " <-- target" if dim == steer_dim else ""
        print(f"  {dim:12s}: rho={rho:+.3f}{marker}")"""))

# ── Cell 10: Visualization ──
cells.append(code_cell("cell-viz", r"""# === Cell 10: Visualization ===

# --- Per-layer weight profile ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, dim in enumerate(DIMENSIONS):
    sel = greedy_results[dim]['selected']
    layers_sorted = sorted(sel.keys())
    weights = [sel[l] for l in layers_sorted]
    signs = [1 if phase1[dim][l]['rho'] > 0 else -1 for l in layers_sorted]
    colors = ['#2196F3' if s > 0 else '#F44336' for s in signs]
    axes[idx].bar([str(l) for l in layers_sorted], weights, color=colors)
    axes[idx].set_xlabel("Layer (neg idx)"); axes[idx].set_ylabel("Weight")
    axes[idx].set_title(f"{dim.capitalize()} (blue=+, red=-)")
    axes[idx].tick_params(axis='x', rotation=45)
plt.suptitle("Greedy-Selected Per-Layer Reading Weights", fontsize=13)
plt.tight_layout()
plt.savefig("reading_weights.png", dpi=150, bbox_inches='tight'); plt.show()

# --- Scatter: predicted vs ground truth ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, dim in enumerate(DIMENSIONS):
    sel = greedy_results[dim]['selected']
    predicted = compute_weighted_reading(sel, dim)
    gt = gt_arrays[dim]
    rho, _ = spearmanr(predicted, gt)
    axes[idx].scatter(gt, predicted, alpha=0.5, s=30, c='#2196F3')
    # Fit line for visual
    z = np.polyfit(gt, predicted, 1)
    x_line = np.linspace(gt.min(), gt.max(), 100)
    axes[idx].plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)
    axes[idx].set_xlabel("Ground-Truth EPA")
    axes[idx].set_ylabel("Weighted Reading")
    axes[idx].set_title(f"{dim.capitalize()}\n{len(sel)} layers, rho={rho:+.3f}")
plt.suptitle("Predicted vs Ground-Truth EPA (Greedy Selection)", fontsize=13)
plt.tight_layout()
plt.savefig("reading_scatter.png", dpi=150, bbox_inches='tight'); plt.show()

# --- Method comparison bar chart ---
method_names = list(methods.keys())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, dim in enumerate(DIMENSIONS):
    rhos = [methods[m][dim]['rho'] for m in method_names]
    axes[idx].bar(method_names, rhos, color=['#9E9E9E', '#FF9800', '#4CAF50'])
    axes[idx].set_ylabel("Spearman rho")
    axes[idx].set_title(f"{dim.capitalize()}")
    axes[idx].set_ylim(0, 1)
plt.suptitle("Reading Accuracy: Single vs Top-5 vs Greedy", fontsize=13)
plt.tight_layout()
plt.savefig("reading_comparison.png", dpi=150, bbox_inches='tight'); plt.show()
print("Visualizations saved.")"""))

# ── Cell 11: Save Results ──
cells.append(code_cell("cell-save", r"""# === Cell 11: Save Results ===
output = {
    "metadata": {
        "model_name": model_name,
        "tuned_at": datetime.now().isoformat(),
        "dataset_path": DATASET_PATH,
        "n_utterances": len(utterances),
        "n_layers": len(all_layers),
        "method": "representation_reading",
    },
    "phase1_correlations": {},
    "greedy_results": {},
    "method_comparison": {},
}

for dim in DIMENSIONS:
    output["phase1_correlations"][dim] = {
        str(l): phase1[dim][l] for l in sorted(phase1[dim].keys())
    }
    gr = greedy_results[dim]
    sel = gr['selected']
    avg = compute_weighted_reading(sel, dim)
    rho, _ = spearmanr(avg, gt_arrays[dim])
    output["greedy_results"][dim] = {
        "selected_layers": {str(k): v for k, v in sel.items()},
        "layer_signs": {str(l): ("+" if phase1[dim][l]['rho'] > 0 else "-") for l in sel},
        "n_layers": len(sel),
        "final_rho": float(rho),
        "score": gr['score'],
    }
    output["method_comparison"][dim] = {
        m: {"rho": info[dim]["rho"], "n_layers": len(info[dim]["layers"])}
        for m, info in methods.items()
    }

with open("epa_reading_tuning_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print("Results saved to epa_reading_tuning_results.json")
print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
for dim in DIMENSIONS:
    gr = output['greedy_results'][dim]
    print(f"\n{dim.upper()}: {gr['n_layers']} layers, rho={gr['final_rho']:+.3f}")
    for l, w in gr['selected_layers'].items():
        sign = gr['layer_signs'][l]
        print(f"  Layer {l}: weight={w}, sign={sign}")"""))

# ── Write notebook ──
nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "repeng", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.13",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py", "mimetype": "text/x-python",
        "nbconvert_exporter": "python", "pygments_lexer": "ipython3"}
}, "nbformat": 4, "nbformat_minor": 5}

with open("hyperparameter_tuning_reading.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
print("Created hyperparameter_tuning_reading.ipynb")
