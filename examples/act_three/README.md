# Affect Control Theory + Representation Engineering

This directory contains the codebase for combining Affect Control Theory (ACT) with Representation Engineering (RepE) to steer the socio-affective behavior of Large Language Models (LLMs) at inference time. 

## Prerequisites

1. Ensure you have Python 3.10+ installed.
2. Install the core `repe` package from the repository root:
   ```bash
   pip install -e .
   ```
3. Install the specific dependencies for these experiments:
   ```bash
   pip install scipy scikit-learn seaborn tqdm pandas matplotlib
   ```

## Reproducing the Paper Results

This codebase allows you to completely reproduce the evaluation data and figures from the paper across the supported models (e.g., `meta-llama/Llama-3.1-8B-Instruct` and `mistralai/Ministral-8B-Instruct-2410`).

### 1. Extract EPA Directions

First, you need to extract the underlying linear representations for Evaluation, Potency, and Activity (EPA) from the target model using Contrastive PCA. Then, you need to tune the reader using ElasticNet.

From the repository root, run:

```bash
# For Llama 3.1 8B (Default)
python -m examples.act_three.extract_directions --model meta-llama/Llama-3.1-8B-Instruct --orthogonalise
python -m examples.act_three.tune_reader --model meta-llama/Llama-3.1-8B-Instruct --directions examples/act_three/models/llama/epa_directions_ortho.pkl

# For Ministral 8B
python extract_directions.py --model mistralai/Ministral-8B-Instruct-2410 --orthogonalise
python -m examples.act_three.tune_reader --model mistralai/Ministral-8B-Instruct-2410 --directions examples/act_three/models/ministral/epa_directions_ortho.pkl
```

### 2. Run the Evaluation Pipeline

The experimental scripts are numbered in execution order inside the `experiments/` directory. They utilize the directions extracted in Step 1.

To run the full suite for a specific model, pass the `--model` flag. By default, it targets Llama-3.1.

```bash
cd experiments

# 1. Evaluate the Reading Quality
python 01_reading_quality.py --model meta-llama/Llama-3.1-8B-Instruct

# 2. Run the Steering Experiments
# Unsteered Baseline
python 02_closed_loop_steering.py --model meta-llama/Llama-3.1-8B-Instruct
# Prompt Engineering Baseline
python 03_prompt_engineering_baseline.py --model meta-llama/Llama-3.1-8B-Instruct
# RepE-only and Hybrid Steering
python 08_hybrid_steering.py --model meta-llama/Llama-3.1-8B-Instruct

# 3. Ablation and Coherence Analysis
python 07_coherence_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct
python 11_dimension_ablation.py --model meta-llama/Llama-3.1-8B-Instruct
python 12_inference_speed.py --model meta-llama/Llama-3.1-8B-Instruct
```

Repeat these commands with `--model mistralai/Ministral-8B-Instruct-2410` to generate the corresponding Ministral results. 

*Outputs are saved as JSON files in `experiments/results/<model_short_name>/`.*

### 3. Generate Figures

Once the experiments have generated the raw JSON data, you can generate all the plots (Reading Scatter, Coefficient Sweeps, and Coherence metrics) used in the paper:

```bash
# Still in the experiments/ directory
python generate_figures.py
```
*The figures will be saved to `experiments/results/figures/`.*

## Adapting to New Models

To adapt this codebase to a new open-weights LLM:
1. Extract the directions as shown in Step 1.
2. Edit `models/<model_short_name>/config.json` to define your calibrated steering coefficients (`per_dim_coefficients`) and set whether to use orthogonal directions.
3. Run the evaluation suite. The pipeline dynamically references the config file based on the `--model` argument.
