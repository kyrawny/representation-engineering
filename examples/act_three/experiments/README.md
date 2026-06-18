# AAAI 2027 Experiment Suite

Comprehensive experiment suite for evaluating the ACT × RepE affect steering pipeline.

## Quick Start

```bash
# From the repository root directory:

# 1. Install dependencies (if not already)
pip install vaderSentiment seaborn tqdm scikit-learn

# 2. Generate calibration datasets (required before first run)
#    Template mode (no GPU, instant):
python -m examples.act_three.experiments.00_setup_datasets --mode template
#    LLM mode (GPU required, ~30 min, higher quality):
python -m examples.act_three.experiments.00_setup_datasets --mode llm

# 3. Run a quick test to validate setup
python -m examples.act_three.experiments.01_reading_quality --quick

# 4. Run the full experiment suite
python -m examples.act_three.experiments.01_reading_quality
python -m examples.act_three.experiments.02_closed_loop_steering
python -m examples.act_three.experiments.03_prompt_engineering_baseline
python -m examples.act_three.experiments.04_coefficient_sweep
python -m examples.act_three.experiments.05_ablation_normalization
python -m examples.act_three.experiments.06_identity_generalization
python -m examples.act_three.experiments.07_coherence_evaluation --perplexity-model both

# 5. Generate all figures
python -m examples.act_three.experiments.generate_figures
```

## Experiments

| # | Script | Purpose | GPU Time (est.) |
|---|--------|---------|-----------------|
| 00 | `00_setup_datasets.py` | Generate calibration train/test datasets | 0 (template) / ~30 min (LLM) |
| 01 | `01_reading_quality.py` | Calibration scatter plots, per-dim metrics | ~2 min |
| 02 | `02_closed_loop_steering.py` | Core: read → ACT → steer → re-read (500 trials) | ~40 min |
| 03 | `03_prompt_engineering_baseline.py` | Prompt-engineering + VADER baseline | ~40 min |
| 04 | `04_coefficient_sweep.py` | Coefficient vs EPA shift curves | ~20 min |
| 05 | `05_ablation_normalization.py` | L2 normalisation ablation | ~10 min |
| 06 | `06_identity_generalization.py` | Multi-identity-pair evaluation | ~15 min |
| 07 | `07_coherence_evaluation.py` | Perplexity (self + GPT-2), text quality | ~15 min |

All experiments support `--quick` mode for fast debugging (~30s each).

## Common Flags

| Flag | Description |
|------|-------------|
| `--quick` | Run on a small subset (5–10 scenarios) for debugging |
| `--output FILENAME` | Override the default output filename |
| `--resume PATH` | Resume from a checkpoint (experiment 02 only) |
| `--perplexity-model {self,gpt2,both}` | Perplexity evaluator (experiment 07 only) |
| `--n-scenarios N` | Override scenario count (experiments 04, 05, 07) |

## Identity Pairs

| Pair Name | Agent | User | Dynamic |
|-----------|-------|------|---------|
| `counselor_client` | counselor | client | Supportive professional |
| `boss_subordinate` | boss | subordinate | Hierarchical authority |
| `teacher_student` | teacher | student | Educational |
| `doctor_patient` | doctor | patient | Clinical care |
| `friend_friend` | friend | friend | Symmetric peer |

## Output

All results are saved as JSON in `results/`.  
All figures are saved as PDF + PNG in `results/figures/`.

### Generated Figures

| Figure | Source | Description |
|--------|--------|-------------|
| `fig01_reading_scatter` | Exp 01 | Predicted vs ground-truth EPA (3 panels) |
| `fig02_layer_correlations` | Tuning results | Per-layer Spearman ρ heatmap |
| `fig03_steering_scatter` | Exp 02 | Target vs achieved EPA (3 panels) |
| `fig04_steering_bars` | Exp 02 | Hit rates and distance improvement bars |
| `fig05_coefficient_sweep` | Exp 04 | Coefficient vs achieved EPA curves |
| `fig06_identity_heatmap` | Exp 06 | Target EPA by identity pair heatmap |
| `fig07_coherence` | Exp 07 | Coherence violin plots |
| `fig08_baseline_comparison` | Exp 03 | 3-method distance comparison bars |

## Configuration

All shared constants (model name, paths, identity pairs, sweep values) are in
[`config.py`](config.py). Model loading is centralised in [`setup.py`](setup.py).
