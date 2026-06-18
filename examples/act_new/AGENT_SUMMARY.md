# ACT Steering with Representation Engineering — Agent Summary

> **Purpose:** Handoff document for another agent to continue development.
> **Last updated:** 2026-02-27
> **Model:** `meta-llama/Llama-3.1-8B-Instruct`

---

## Project Goal

Use **Representation Engineering** (RepE) to steer LLM generation according to **Affect Control Theory** (ACT). The system reads and controls three EPA dimensions (Evaluation, Potency, Activity) in model hidden states to produce responses that minimize ACT deflection — the discrepancy between expected and observed affective meanings.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  ACT Core (act_core.py)                                     │
│  - EPA dataclass, impression formation, deflection calc     │
│  - Optimal behavior finding via numerical optimization      │
├─────────────────────────────────────────────────────────────┤
│  EPA Directions (epa_directions.pkl)                        │
│  - PCA direction vectors per layer per dimension            │
│  - Extracted via contrastive prompts + RepReadingPipeline   │
│  - Keys: rep_readers, hidden_layers, model_name             │
├─────────────────────────────────────────────────────────────┤
│  Reading: project hidden states onto directions             │
│  Steering: add scaled directions to hidden states           │
├─────────────────────────────────────────────────────────────┤
│  Calibration (epa_calibration.py)                           │
│  - Maps raw vector readings ↔ ACT dictionary EPA values     │
│  - Linear regression or affine calibrators                  │
├─────────────────────────────────────────────────────────────┤
│  Conversation Steering (../act/conversation_steering.py)    │
│  - PID-style deflection control with transient decay        │
│  - ConversationState, DeflectionController, ACTSteeringEngine│
└─────────────────────────────────────────────────────────────┘
```

## Key Files

### Source Modules

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `act_core.py` | ACT math | `EPA`, `ACTCoefficients`, `impression_formation()`, `calculate_deflection()`, `find_optimal_behavior()`, `predict_emotion()` |
| `utils.py` | RepE utilities | `format_llama3_prompt()`, `format_for_reading()`, `create_epa_dataset()`, `make_epa_activations()`, `read_epa_scores()`, `read_epa()`, `plot_tsne_epa()`, `plot_lat_scan()` |
| `epa_calibration.py` | Calibration | `BehaviorPromptGenerator`, `CalibrationCoefficients`, `LinearRegressionCalibrator`, `AffineCalibrator` |
| `../act/conversation_steering.py` | Steering engine | `ACTSteeringEngine`, `ConversationState`, `DeflectionController`, `PromptFormatConfig` |

### Data Files

| File | Format | Contents |
|------|--------|----------|
| `epa_directions.pkl` | pickle | `rep_readers` (dict of RepReader per dim), `hidden_layers` (list of **negative** indices -1..-31), `model_name` |
| `epa_tuning_dataset.json` | JSON | 85 LLM-generated utterances with ground-truth EPA values (from ACT dictionary) and Likert-scale labels |
| `epa_tuning_results.json` | JSON | Results from v1 Likert-based tuning (uniform coefficients) |
| `epa_reading_tuning_results.json` | JSON | Results from representation-reading-based tuning |

### Notebooks

| Notebook | Approach | Status |
|----------|----------|--------|
| `act_epa_pipeline.ipynb` | Direction extraction & t-SNE/LAT visualization | ✅ Complete, produces `epa_directions.pkl` |
| `act_logprob.ipynb` | Early log-probability exploration | ✅ Complete |
| `hyperparameter_tuning.ipynb` | Likert log-prob grid search (uniform coeff, layer subsets) | ✅ Complete, results saved |
| `hyperparameter_tuning_v2.ipynb` | Per-layer Likert tuning: Phase 1 sweep + Phase 2 greedy | ✅ Generated, not yet fully run |
| `hyperparameter_tuning_reading.ipynb` | **Representation reading** tuning (fast) | ✅ Complete, results saved |

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `fix_notebook.py` | Patches v1 notebook: negative↔positive layer index conversion |
| `create_tuning_v2.py` | Generates `hyperparameter_tuning_v2.ipynb` |
| `create_tuning_reading.py` | Generates `hyperparameter_tuning_reading.ipynb` |

## Critical Technical Details

### Layer Index Convention

**This is the most common source of bugs.** The stored directions use **negative** layer indices (`-1` to `-31`). The model wrapping API (`WrappedReadingVecModel.wrap_block`, `set_controller`) uses **positive** indices into `model.model.layers`.

```python
n_layers = len(model.model.layers)  # 32 for Llama-3.1-8B

# Negative → Positive: layer -k = position (n_layers + k)
# Example: layer -15 → position 17
def neg_to_pos(neg_layer):
    return n_layers + neg_layer

# make_epa_activations() needs NEGATIVE keys (match rep_reader.directions)
# wrap_block() / set_controller() need POSITIVE keys
activations_neg = make_epa_activations(rep_readers, layers=[-15, -20], ...)
activations_pos = {neg_to_pos(k): v for k, v in activations_neg.items()}
wrapped_model.wrap_block([17, 12], block_name="decoder_block")
wrapped_model.set_controller([17, 12], activations_pos, "decoder_block")
```

### RepReadingPipeline Constraint

The pipeline asserts `len(rep_reader.directions) == len(hidden_layers)`. You **must** pass ALL layers from the rep_reader, not a subset. Filter layers in the averaging/weighting step afterwards — see `read_epa_scores()` in `utils.py`.

### How Reading Works

For each utterance, per dimension, per layer:
```
score = dot(last_token_hidden_state[layer], pca_direction[dim][layer])
```
One forward pass extracts scores from all layers. Different layers have different correlation with ground-truth EPA values (some even negatively correlated — flip the sign).

### How Steering Works

```python
# 1. Build activation dict (negative keys)
activations = make_epa_activations(rep_readers, layers, e_coeff=0.3, ...)
# 2. Convert keys to positive
activations_pos = {n_layers + k: v for k, v in activations.items()}
# 3. Wrap and apply
wrapped_model.wrap_block(pos_layers, block_name="decoder_block")
wrapped_model.set_controller(pos_layers, activations_pos, "decoder_block")
# 4. Model now steers during any forward pass (generation or scoring)
# 5. Always reset after use:
wrapped_model.reset()
wrapped_model.unwrap()
```

The `make_epa_activations()` function L2-normalizes direction vectors by default, so `coeff=1.0` adds a unit vector perturbation.

### Likert Disambiguation

"Very low" and "Very high" both start with token "Very". The Likert scoring code handles this via 2-token joint probability: `P("Very low") = P("Very") × P("low"|"Very")`.

## Current Results

### Reading-Based Tuning (from `epa_reading_tuning_results.json`)

The representation reading notebook has completed and produced results. Layers with strongest per-dimension signal and the greedy-selected layer combinations are saved.

### Likert-Based Tuning (from `epa_tuning_results.json`)

The v1 Likert notebook completed with the uniform-coefficient grid search. Baseline Spearman correlations:
- Evaluation: ρ = +0.746
- Potency: ρ = +0.403
- Activity: ρ = +0.516

## Imports Pattern

All notebooks in `act_new/` use this import pattern:
```python
import sys; sys.path.append('../..')
from examples.act_new.act_core import EPA
from examples.act_new.epa_calibration import BehaviorPromptGenerator
from examples.act_new.utils import format_llama3_prompt, make_epa_activations
```

The `epa_calibration.py` uses internal `from .act_core import EPA` (relative import), which works because `sys.path` includes the repo root.

## Likely Next Steps

1. **Analyze tuning results** — compare reading-based vs Likert-based optimal layers and decide which to use for steering
2. **Apply best hyperparameters** — update `conversation_steering.py` / demo to use tuned per-layer coefficients instead of uniform
3. **End-to-end demo** — run multi-turn conversations with the tuned steering and evaluate deflection minimization
4. **Calibration refinement** — use the representation reading tuning results to improve `CalibrationCoefficients` (map raw readings → ACT-scale EPA)

## Related Files Outside `act_new/`

- `examples/act/conversation_steering.py` — the full ACT steering engine (`ACTSteeringEngine`)
- `examples/act/demo/demo_server.py` — Flask web demo with real-time steering
- `data/act/MTurkInteract_Behaviors.csv` — ACT behavior dictionary (used by `BehaviorPromptGenerator`)
- `data/act/2010impressionformation.csv` — impression formation coefficients
- `data/act/user_inputs.json` — user input prompts for contrastive dataset generation
- `repe/rep_reading_pipeline.py` — the RepE reading pipeline
- `repe/rep_control_reading_vec.py` — `WrappedReadingVecModel` for steering
