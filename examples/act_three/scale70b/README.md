# Does ACT steering hold at 70B? Instructions for collaborating labs

This folder runs the paper's Affect Control Theory (ACT) × Representation Engineering (RepE) steering experiment on **Llama-3.1-70B-Instruct**. It tests whether the effect reported for Llama-3.1-8B still appears at scale: steered responses move toward the ACT-predicted Evaluation, Potency, Activity (EPA) targets compared with unsteered responses.

> **Confidential.** The paper is under anonymous review. Please do not share the code, data or results outside your group, and do not post them publicly.

## What the experiment does

For each of 1,300 trials (100 user messages × 13 identity pairs, e.g. *counselor → client*):

1. **Read:** the model's own EPA reader (linear probes on hidden states) reads the user message.
2. **Compute:** ACT's deflection-minimisation equations give the target EPA for the response.
3. **Steer:** the model responds under four conditions:

   | Condition | What it is |
   |---|---|
   | `unsteered` | identity preamble only |
   | `repe_only` | activation steering toward the target |
   | `pe_only` | prompt instruction describing the target |
   | `hybrid` | prompt instruction and activation steering |

   Optional ablations: `repe_const` (steering toward one fixed target), and `repe_only_s1` / `hybrid_s1` (the literal 8B strength).
4. **Score:** every response's EPA is read, and its distance to the target is compared with the unsteered response for the same trial.

All of the method code is imported unchanged from `examples/act_three/`: contrastive-PCA direction extraction, ElasticNet reader, ACT equations, steering formula, prompts and greedy decoding. This folder adds only what a 70B run needs: multi-GPU loading, sharding, resumable logs, a strength calibration rule, an independent reader check and the analysis.

### The one thing that has to be adapted: steering strength

The paper adds unit-norm direction vectors with a clamp of 1.0. Residual-stream norms are larger in a 70B model, so the same absolute perturbation is relatively weaker.

Stage 4 therefore chooses a single **strength multiplier s** with a pre-specified rule. s scales the paper's coefficients and clamp limit together, so the perturbation is exactly s × the 8B perturbation for the same target. The rule is applied in the same way to 8B and 70B:

> Choose the largest s in {0.5, 1, 2, 4, 8, 16, 32} (plus the norm-matched value s_match from stage 3) such that s and every smaller value give, on 63 calibration trials:
> - no degenerate outputs (shorter than 20 characters, or bigram repetition above 0.2);
> - geometric-mean self-perplexity at most 1.25 × unsteered;
> - mean length between 0.5 × and 2 × unsteered.

The EPA effect is **not** used to choose s, so the choice cannot be tuned toward the result. Stage R first applies the same rule to the paper's 8B model, to check that it reproduces the paper's setting (s = 1). The result is saved in `scale70b/reference/calibration_llama3.json`. Optional conditions `*_s1` also report the literal 8B strength.

### What counts as the effect holding (pre-specified)

For `repe_only` and `hybrid` separately, the effect **holds** if the mean paired change in distance to the ACT target versus unsteered (averaged over E, P, A) is **negative with a 95% cluster-bootstrap CI that excludes 0**, under **both**:
1. the 70B model's own reader (as in the paper), and
2. an **independent reader** built on Llama-3.1-8B. This answers the reviewers' concern that the same reader sets the targets and scores the outputs.

`REPORT.md` states the verdict. It also gives effect sizes relative to 8B, tracking slopes, Lin's concordance and coherence.

## Hardware

| Setup | Fits? | Notes |
|---|---|---|
| 4 × 80 GB (A100/H100), fp16 | Yes (recommended) | ~141 GB of weights plus room for activations |
| 2 × 80 GB, fp16 | Tight | set `ACT70B_MAX_MEMORY="0:78GiB,1:78GiB"`; short sequences only |
| 8 × 40–48 GB, fp16 | Yes | `device_map="auto"` splits the layers |
| 2 × 80 GB, 8-bit (`ACT70B_QUANT=8bit`) | Yes | Changes hidden states. Every stage must use the same setting; report it. |
| 1 × 80 GB, 4-bit | Possible | Last resort; report it |

Stages R and 7 also load Llama-3.1-8B (needs ~17 GB on one GPU). Disk: ~170 GB for weights plus about 1 GB of results.

**Expected time** at ~12 tokens/s with batch size 1 on 4 × A100-80GB (preflight prints a projection for your hardware):
- stage R (8B reference, one GPU): about 30 min;
- stages 0–4: about 3 h;
- stage 5 (6,500 generations): about 12 h on one node, or about 3 h with 4 shards;
- stages 6–8: about 1 h.

## Setup

The environment replicates the authors' `repeng` environment, which produced the paper's results: Python 3.11, torch 2.7.1 (CUDA 12.6), transformers 4.52.4.

```bash
git clone <repository URL provided by the authors>
cd representation-engineering
conda env create -f examples/act_three/scale70b/environment.yml   # or mamba / micromamba
conda activate repeng
pip install -e .            # the repo's own 'repe' package; do NOT install repe from PyPI/GitHub
# Optional for 8/4-bit: pip install bitsandbytes
huggingface-cli login       # accept the Llama 3.1 licence for 8B and 70B first
```

Without conda, use `python3.11 -m venv` and then:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r examples/act_three/scale70b/requirements-lock.txt
```
(Delete the `python==` line from the file first.)

| File | Purpose |
|---|---|
| `environment.yml` | Installable Linux/CUDA environment, pinned to the versions that load in the authors' environment |
| `requirements-lock.txt` | The same pins for pip; preflight compares your versions against them |
| `repeng-inventory.txt` | Full inventory of the authors' Windows environment, for provenance |

**Keep `torch`, `transformers`, `tokenizers` and `accelerate` at exactly these versions.** The RepE pipelines import transformers internals that change between releases. If your GPUs need a different CUDA build of torch 2.7.1 (e.g. cu128 for Blackwell), change only the index URL and note it in what you send back. Preflight reports any version mismatches.

## Running

From the repository root:

```bash
bash examples/act_three/scale70b/run_all.sh
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | model to test |
| `STAGES` | `R 0 1 2 3 4 5 6 7 8` | stages to run (`R` = 8B reference) |
| `ACT70B_DTYPE` | `float16` (as in the paper) | use `bfloat16` if preflight reports non-finite logits, for every stage |
| `ACT70B_QUANT` | `none` | `8bit` or `4bit` |
| `ACT70B_MAX_MEMORY` | (auto) | e.g. `0:76GiB,1:76GiB,2:76GiB,3:76GiB` |
| `SHARD_INDEX`, `NUM_SHARDS` | `0`, `1` | split stage 5 across jobs or nodes |
| `EXTRA_CONDITIONS` | `repe_const` | add `repe_only_s1 hybrid_s1` if time allows (+2,600 generations) |

Every stage writes results as it goes and **resumes** if re-run. On a cluster, run stages R and 0–4 once, stage 5 as an array of shards (see `slurm_example.sbatch`), then stages 6–8 once.

| Stage | Script | Output (under `examples/act_three/experiments/results/scale/<model>/`) |
|---|---|---|
| R | `s03_norm_probe.py` and `s04_strength_calibration.py` on Llama-3.1-8B | `scale70b/reference/norm_probe_llama3.json` (8B perturbation size) and `calibration_llama3.json` (rule check; should choose s = 1) |
| 0 | `s00_preflight.py` | `preflight.json`: GPUs, versions, finite logits, steering works across GPUs, reader equivalence, time projection |
| 1 | `run_cli.py` → the paper's `extract_directions.py` and `tune_reader.py` | `examples/act_three/models/<model>/`: directions, reader tuning results (Table 1 row) |
| 2 | `s02_targets.py` | `targets.json`: user-message EPA and ACT targets for 1,300 trials |
| 3 | `s03_norm_probe.py` | `norm_probe.json`: relative perturbation size, s_match |
| 4 | `s04_strength_calibration.py` | `calibration.json`, `models/<model>/config.json` |
| 5 | `s05_main_steering.py` | `main/shard_*.jsonl`: all responses and EPA readings |
| 6 | `s06_coherence.py` | `coherence_main.json` |
| 7 | `s07_cross_read.py` | `read_by_llama3.jsonl` (70B outputs, 8B reader); `../llama3/read_by_<70b>.jsonl` (paper's 8B outputs, 70B reader) |
| 8 | `s08_merge_analyze.py` (CPU only) | `REPORT.md`, `summary.json`, merged `08_hybrid_steering_main.json`, and a `act_scale_results_*.tar.gz` bundle in the repository root |

Quick smoke test before the full run (after stages 0–4):

```bash
python -m examples.act_three.scale70b.s05_main_steering --max-trials 3 --tag smoke
```

## What to send back

The `act_scale_results_<model>_<date>.tar.gz` file created by stage 8. It contains all results, the 8B reference checks, the reader tuning, the calibration, the directions and the logs; no model weights. Please also tell us the GPU type and count, and any deviations (dtype, quantisation, version mismatches, stages skipped).

## Troubleshooting

| Problem | Fix |
|---|---|
| Non-finite logits in fp16 | `ACT70B_DTYPE=bfloat16`, then re-run **from stage 1**. Directions must come from the same dtype. |
| Preflight: "steering_changes_output: false" | The hooks did not fire, probably a transformers version difference in the decoder-layer output format. Send us `preflight.json`. |
| Preflight: FastReader tolerance exceeded | Later stages fall back to `--read-batch-size 1` (slower but exact). If even batch 1 fails, send us `preflight.json`. |
| Out of memory | Lower `ACT70B_MAX_MEMORY` per GPU so accelerate leaves headroom, or add GPUs. |
| Stage 4 picks `None` | Even s = 0.5 degenerated. Send `calibration.json` and `calibration.jsonl`. |
| Job killed | Re-run the same command; finished trials are skipped. |

Questions: contact the authors through the channel they shared this with.
