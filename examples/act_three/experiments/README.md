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
python -m examples.act_three.experiments.08_hybrid_steering
python -m examples.act_three.experiments.09_direction_quality_comparison
python -m examples.act_three.experiments.10_statistical_significance

# 5. Generate all figures
python -m examples.act_three.experiments.generate_figures
```

## Direction Extraction & Tuning Pipeline

Standalone CLI scripts replace the notebook-based workflow for full reproducibility.

```bash
# 1. Re-extract EPA directions (GPU required, ~15 min)
python -m examples.act_three.extract_directions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output examples/act_three/epa_directions.pkl

# 2. Orthogonalise directions (Gram–Schmidt, CPU only, instant)
python -m examples.act_three.orthogonalise_directions \
    --input examples/act_three/epa_directions.pkl \
    --output examples/act_three/epa_directions_ortho.pkl

# 3. Re-tune the reader (GPU required, ~20 min)
python -m examples.act_three.tune_reader \
    --directions examples/act_three/epa_directions_ortho.pkl \
    --method ElasticNet

# Or extract + orthogonalise in one step:
python -m examples.act_three.extract_directions --orthogonalise
```

### Backups

Original v1 artifacts are preserved in `backups/v1/`:
- `epa_directions.pkl` — original direction vectors
- `epa_reading_tuning_v2_results.json` — original reader tuning
- `epa_tuning_results.json` — original steering tuning

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
| 08 | `08_hybrid_steering.py` | Hybrid PE + RepE steering evaluation | ~60 min |
| 09 | `09_direction_quality_comparison.py` | Direction quality & entanglement comparison | ~1 min (CPU) |
| 10 | `10_statistical_significance.py` | Post-hoc significance tests & LaTeX table | ~1 min (CPU) |

All experiments support `--quick` mode for fast debugging (~30s each).

### Experiment Dependencies

```
00 → 01 → 02 → 03 → 10
              ↘ 04        ↗
              ↘ 05       /
              ↘ 06      /
              ↘ 07     /
              ↘ 08 → 10
09 (standalone — needs direction files only)
```

## Common Flags

| Flag | Description |
|------|-------------|
| `--quick` | Run on a small subset (5–10 scenarios) for debugging |
| `--output FILENAME` | Override the default output filename |
| `--resume PATH` | Resume from a checkpoint (experiment 02 only) |
| `--per-dim-coeff` | Use per-dimension optimised coefficients (experiment 02) |
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

## Statistical Testing

The `stats_utils.py` module provides publication-quality statistical tests:

| Function | Purpose |
|----------|---------|
| `bootstrap_ci(data, fn)` | Bootstrap confidence interval on any statistic |
| `bootstrap_ci_paired(x, y, fn)` | Paired bootstrap CI (preserves index pairing) |
| `paired_permutation_test(x, y)` | Permutation test for paired mean differences |
| `cohens_d(x, y)` | Cohen's d for independent samples |
| `cohens_d_paired(x, y)` | Cohen's d_z for paired samples |
| `wilcoxon_signed_rank(x, y)` | Wilcoxon signed-rank test |
| `mann_whitney_u(x, y)` | Mann–Whitney U test |
| `format_ci(point, lo, hi)` | Format as LaTeX: `$0.574\ [0.521, 0.628]$` |
| `format_p(p)` | Format p-value: `$p < .001$` |
| `significance_stars(p)` | Returns `***` / `**` / `*` / `ns` |

Experiments 02, 03, and 08 automatically compute bootstrap CIs, Wilcoxon
signed-rank tests, permutation tests, and Cohen's d for all comparisons.
Experiment 10 aggregates these into a single LaTeX table.

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
| `fig09_hybrid_comparison` | Exp 08 | 4-method distance comparison with CIs |
| `fig10_direction_comparison` | Exp 09 | Cross-dimension entanglement comparison |
| `fig11_significance_forest` | Exp 10 | Effect sizes with significance colours |
| `fig12_orthogonal_cosines` | Exp 09 | Per-layer cosine similarity before/after |

## Configuration

All shared constants (model name, paths, identity pairs, sweep values, per-dimension
coefficients) are in [`config.py`](config.py). Model loading is centralised in
[`setup.py`](setup.py).

### Key Config Options

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ORTHOGONAL_DIRECTIONS` | `True` | Load orthogonalised directions if available |
| `PER_DIM_COEFFICIENTS` | `{E: 3.0, P: 1.0, A: 2.5}` | Per-dimension steering coefficients |
| `COEFF_SWEEP_VALUES` | `[0.0 .. 5.0]` | Coefficient values for sweep experiments |
