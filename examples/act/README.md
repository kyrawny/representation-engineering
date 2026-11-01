# ACT Deflection Steering (`examples/act`)

Steers conversational LLM generation by **Affect Control Theory (ACT)
deflection** using the [RepE](../../README.md) toolkit. Each conversational
event (actor, behaviour, object) produces a deflection

    U = sum_i w_i * (W_i - f_i)^2

— the squared distance between post-event impressions `W` and the participants'
EPA fundamentals `f`. This package makes a conversation *minimize expected
deflection* while keeping response quality, using three composable
intervention modes plus a multi-turn control loop:

| Mode | Module | Mechanism |
|------|--------|-----------|
| Monitor | `monitor` | Per-token deflection trace from a PCA reader fitted on low- vs high-deflection contrast continuations; threshold trigger for repair. |
| Actuate | `steerer` | Deflection-contrast direction `c = alpha * (h_low - h_high)` added at the tail of the residual stream; per-turn gain proportional to the current deflection. |
| Select | `selector` | ACT best-of-k: generate k candidates, score the **actual text** by post-event deflection, select the argmin. |
| Loop | `pipeline` | Per-turn: parse event -> `U_pre` -> steer -> select -> monitor -> repair (re-identification) -> update state (sentiment drift). |
| Persist | `lorra` | LoRRA fine-tuning on low-/high-deflection continuation triplets (optional; see [LoRRA persistence mode](#lorra-persistence-mode-optional)). |

The package is self-contained: it builds only on the upstream `repe` package
and the data in [`data/act`](../../data/act) (MTurk-Interact EPA dictionaries
plus the 15-term impression-formation coefficient set). No code is copied from
other forks or repositories.

## Layout

```
examples/act/
├── act_steering/            # the package
│   ├── config.py            # dataclass configs, paths, seeds (torch-free)
│   ├── dictionary.py        # MTurk-Interact identities/behaviours/modifiers EPA lookups  [S2]
│   ├── act_core.py          # impression formation, deflection, optimal behaviour          [S3]
│   ├── stimuli.py           # distribution-matched low/high-deflection contrast sets       [S4]
│   ├── state.py             # ConversationState: fundamentals, drift, trace, triggers      [S5]
│   ├── directions.py        # direction extraction (EPA reader + deflection contrast)      [S6]
│   ├── monitor.py           # per-token deflection trace, threshold trigger                [S7]
│   ├── steerer.py           # per-turn contrast steering via RepControl                    [S8]
│   ├── selector.py          # ACT best-of-k over actual generated text                     [S9]
│   ├── pipeline.py          # multi-turn control loop, JSON trace log                      [S10]
│   └── lorra.py             # LoRRA triplets from the contrast set + training CLI          [S13]
├── evaluation/              # reader_validation, dose_response, deflection_reduction,      [S11]
│                            # sycophancy_probe + sycophancy_dialogue, quality_guardrails,
│                            # ablations
├── tests/                   # pytest suite (model-free modules are CPU-tested)
├── artifacts/               # extracted direction artifacts (created on demand)
├── results/                 # JSON evaluation results and trace logs (created on demand)
├── GPU_RUNBOOK.md           # exact GPU-phase sequence (directions → traces → evaluators)
├── pytest.ini
└── ruff.toml
```

`[S#]` marks the task-plan step that delivers the module; `[planned]` modules
do not exist yet.

## Requirements

- Python >= 3.9 (repository floor); development environment: CPython 3.11 with
  `torch` 2.7.x (CUDA) and `transformers` 4.52.x, plus `pytest` and `ruff`.
- A GPU (RTX 4090 class) for the model-bound modules; the pure-logic core runs
  and is tested on CPU.
- Default model: `meta-llama/Llama-3.1-8B-Instruct`. The in-loop contrast mode
  is Llama/Mistral-only (upstream limitation) and disables the KV cache.

Run the test suite and linter from this directory:

```bash
python -m pytest
python -m ruff check .
```

## Style guide

Conventions for all code in this directory:

1. **Python >= 3.9 compatibility.** The repository's `requires-python` is 3.9;
   use `from __future__ import annotations` plus modern generic syntax in
   annotations.
2. **PEP 8**, enforced by `ruff` (see `ruff.toml`: pycodestyle, pyflakes,
   isort, bugbear, pyupgrade, simplify).
3. **Full type hints** on all public functions, methods, and dataclasses.
4. **Google-style docstrings** on every public module, class, function, and
   method, with `Args`/`Returns`/`Raises` where applicable.
5. **Dataclasses for all configuration**, frozen where immutable, no mutable
   defaults (use `field(default_factory=...)`).
6. **`pathlib` for all paths**; no hardcoded absolute paths; every path flows
   from `DataPaths` or a CLI argument.
7. **`argparse` CLIs** for every entry point; every CLI supports `--quick`
   (smoke mode: minimal data/model so the full path runs end-to-end) and
   `--seed`.
8. **`logging` via `logging.getLogger(__name__)`**; no `print()` in library
   code (CLIs may print final summaries).
9. **Determinism**: every stochastic operation takes a seed; RNG instances
   (`numpy.random.default_rng`, `torch.Generator`, `random.Random`) are derived
   from `ExperimentConfig.seed` and passed explicitly, never read from global
   state.
10. **Layering**: model-free modules (`config`, `dictionary`, `act_core`,
    `stimuli`, `state`, `monitor`, `selector`, `evaluation` statistics) must
    import and run without `torch`; model-bound modules import `torch` at
    module level and are GPU-gated.
11. **No external code reuse**: nothing is copied from other forks of this
    repository; ACT algorithm details may be referenced from the published
    literature (Heise 2007) only.
12. **Tests**: every model-free claim has a `pytest` test; every model-bound
    behaviour has a documented `--quick` smoke command. Results are JSON with
    run metadata (model, seed, git hash, config).

## Control loop (S10)

The per-turn control loop is CPU-testable via deterministic mock components
(no model needed):

```bash
# built-in two-message smoke script:
python -m act_steering.pipeline --quick --out results/quick_trace.json

# your own scripted conversation (JSON: user/agent identity terms + messages):
python -m act_steering.pipeline --mock --script my_script.json --out results/trace.json --k 4
```

The trace log records, per turn: the event-1 deflection `u_pre`, per-candidate
post-event deflections, the winner, the chosen response, the state trigger,
and (when enabled) the monitor trace and repair outcome. Real model components
(RepE steerer as generator, M-X reader as message/behaviour reader, monitor
over hidden states) are wired in the GPU phase on top of this loop.

## Quickstart

Run everything from this directory (``examples/act``), with the repository
installed in editable mode:

```bash
# from the repository root
pip install -e .

# from examples/act: the full CPU test suite + linter
python -m pytest
python -m ruff check .

# first end-to-end run (CPU only): the control loop over the built-in smoke script
python -m act_steering.pipeline --quick --out results/quick_trace.json
```

The ``--quick`` smoke run above exercises every stage of the loop (event parse,
``U_pre``, candidate generation, best-of-k selection, state update) with
deterministic mock components, so it runs on a laptop in seconds. Every entry
point in this package ships the same ``--quick`` mode.

## Entry points

| Entry point | Mode | What it produces |
|---|---|---|
| ``act_steering.stimuli`` | CPU | Low-/high-deflection contrast set JSON |
| ``act_steering.directions`` | GPU | EPA reader directions + deflection-contrast direction artifacts |
| ``act_steering.pipeline`` | CPU (``--mock``) / GPU | Per-turn trace log JSON |
| ``evaluation.reader_validation`` | CPU (``--reader mock``) / GPU | Reader-vs-human agreement report |
| ``evaluation.deflection_reduction`` | CPU (``--quick``) / GPU | Per-turn ``U`` comparison with cluster-bootstrap CIs |
| ``evaluation.quality_guardrails`` | CPU (``--scorer mock``) / GPU | Repetition / length / perplexity report |
| ``evaluation.sycophancy_probe`` | CPU (``--quick``) / GPU | Flip-rate + harshness report |
| ``evaluation.sycophancy_dialogue`` | CPU (``--generate mock``) / GPU | Scripted are-you-sure dialogue logs for the probe |
| ``evaluation.ablations`` | CPU (mock) / GPU (``--battery --model``, real mode) | Condition-matrix ablation report (10 conditions, cluster-bootstrap CIs) |
| ``evaluation.dose_response`` | CPU (``--quick``) / GPU | Alpha-sweep curve + degeneration threshold report |
| ``act_steering.lorra`` | CPU (``--quick``) / GPU (``--train --execute``) | LoRRA triplet JSONL + dry-run training command |

### Stimuli

```bash
# distribution-matched contrast set (1,300 identity pairs by default)
python -m act_steering.stimuli --out artifacts/contrast_set.json
python -m act_steering.stimuli --quick --out artifacts/contrast_set_quick.json
```

### Directions (GPU)

```bash
# artifacts are written to artifacts/ (npz + JSON sidecar)
python -m act_steering.directions --model meta-llama/Llama-3.1-8B-Instruct --lowdin
python -m act_steering.directions --quick
python -m act_steering.directions --layers 0,8,16,24,31 --lowdin
```

Flags: ``--device``, ``--dtype``, ``--layers`` (0-indexed list or ``all``),
``--contrast-set``, ``--n-epa-per-side``, ``--batch-size``, ``--max-length``,
``--seed``, ``--lowdin`` (Løwdin-orthonormalise the EPA direction set).

### Pipeline

```bash
python -m act_steering.pipeline --quick --out results/quick_trace.json
python -m act_steering.pipeline --mock --script my_script.json \
    --user student --agent teacher --k 4 --out results/trace.json
```

Script JSON: ``{"user": "...", "agent": "...", "messages": ["..."]}`` with
identity terms that exist in the MTurk-Interact identity dictionary.

### Evaluation

```bash
# reader agreement (Pearson r, Spearman, MAE, Lin's CCC per dimension, CV splits)
python -m evaluation.reader_validation --quick --out results/reader_validation.json

# per-turn deflection reduction across conditions (paired cluster bootstrap)
python -m evaluation.deflection_reduction --quick --out results/deflection_reduction.json
python -m evaluation.deflection_reduction --log unsteered=results/u.json \
    --log steered=results/s.json --baseline unsteered --treatments steered \
    --out results/deflection_reduction.json

# quality guardrails (repetition rate, length, perplexity under a reference scorer)
python -m evaluation.quality_guardrails --quick --out results/quality_guardrails.json
python -m evaluation.quality_guardrails --input cond=results/responses.json \
    --scorer mock --out results/quality_guardrails.json

# sycophancy probe (are-you-sure flip rate, harshness counter-check)
python -m evaluation.sycophancy_dialogue --generate mock --judge mock \
    --out results/dialogue_unsteered.json
python -m evaluation.sycophancy_probe --quick --out results/sycophancy_probe.json

# ablation condition matrix (9 labeled conditions through the full loop)
python -m evaluation.ablations --quick --out results/ablations.json

# dose-response alpha sweep with degeneration-threshold detection
python -m evaluation.dose_response --quick --out results/dose_response.json
```

### LoRRA persistence mode (optional)

LoRRA (Representation Learning via Representation Alignment) moves the
low-deflection behaviour from *inference-time steering* into the weights:
a model fine-tuned on deflection-contrast triplets should keep answering
with low deflection even without the control statement. This is optional
(P2) and not part of the default pipeline.

**Triplet scheme.** One triplet per contrast pair, in the exact template
format the upstream trainer expects (`lorra_finetune/src/train_val_datasets.py`):
the shared trial context is the instruction, the **low-deflection
continuation** is the single shared response (desired under the positive
control, undesired under the negative), and the two controls are
`Give a a low-deflection response.` / `Give a a high-deflection response.
` by default (the double article mirrors the upstream example; use
`--control-template "Give {type} response."` to avoid it).

```bash
# build the triplets JSONL (CPU; --quick caps at 20 pairs)
python -m act_steering.lorra --quick --out artifacts/lorra_triplets.jsonl

# full build from a contrast set, with a U-separation floor
python -m act_steering.lorra --contrast-set artifacts/contrast_set.json \
    --out artifacts/lorra_triplets.jsonl --min-delta-u 0.05 --seed 42

# dry-run: build the triplets, then print the exact upstream training command
python -m act_steering.lorra --train --out artifacts/lorra_triplets.jsonl

# GPU phase: run the composed lorra_finetune invocation
python -m act_steering.lorra --train --execute --model meta-llama/Llama-3.1-8B-Instruct
```

Key flags: `--max-triplets` (seeded subsample cap), `--min-delta-u`
(drop pairs below a U separation), `--target-layers` (hidden-state layer
indices for the alignment), `--lorra-alpha` / `--lorra-beta` (LoRRA loss
weights), `--epochs` / `--batch-size` / `--learning-rate` (HF training
config).

**Dataset hook.** The upstream `AlpacaSupervisedDataset` currently reads
the `tatsu-lab/alpaca` dataset. For an ACT run, point it at the triplets
JSONL written by this CLI (one-line integration in
`lorra_finetune/src/train_val_datasets.py`; the dry-run output prints the
exact command to run once the hook is in place). The fine-tune itself is
the only LoRRA step that needs the 4090.

## GPU runbook

See [GPU_RUNBOOK.md](GPU_RUNBOOK.md) for the exact sequence: directions
extraction, per-condition pipeline traces, reader validation, deflection
reduction, quality guardrails with a separate reference model, the
sycophancy dialogue harness + probe, and the dose-response sweep.

## Evaluation protocol

Six evaluators, each answering one question. All are CPU-runnable in mock mode
and take the real model-backed components in the GPU phase; all write a single
JSON report with run metadata (``model``, ``seed``, ``git`` hash, ``config``)
so every published number traces back to a log:

| Evaluator | Question | Key statistics |
|---|---|---|
| ``reader_validation`` | Does the EPA reader agree with human ratings? | Pearson ``r``, Spearman, MAE, Lin's CCC per dimension; cross-validated splits; ICC-style rater agreement |
| ``deflection_reduction`` | Does steering reduce per-turn deflection ``U``? | Paired per-turn differences vs. a named baseline; cluster-bootstrap CIs (clusters = scenarios); significance flag at the requested ``alpha`` |
| ``quality_guardrails`` | Does steering degrade the text? | N-gram repetition rate, response length, perplexity under a separate reference scorer |
| ``sycophancy_probe`` | Does steering change sycophantic behaviour or just tone? | Are-you-sure flip rate, paired bootstrap CIs, harshness counter-check with threshold |
| ``ablations`` | Which part of the design does the work? | Labeled condition matrix (unsteered, monitor-only, steer, random-direction, constant-vector, identity-only, situation-dependent, best-of-k, best-of-k+steer) through the full control loop |
| ``dose_response`` | How much steering strength is usable? | Alpha sweep of per-dimension reader shift vs. perplexity/repetition growth; ``degenerated_at`` and ``safe_alpha_max`` thresholds |

Interpretation rule: the reader built on the steered model is **diagnostic
only** (it shares the steered hidden states); headline claims must use
``reader_validation`` to pick the best-independently-validated metric.

## Data provenance

All data lives in [`data/act`](../../data/act) and is registered by
``DataPaths`` (no module hardcodes paths):

| File | Content | Source |
|---|---|---|
| ``MTurkInteract_Identities.csv`` | Human EPA ratings of social-role terms (``consultant``, ``student``, ...) | MTurk-Interact crowdsourced semantic-differential ratings |
| ``MTurkInteract_Behaviors.csv`` | Human EPA ratings of behaviour terms (``apologize``, ``praise``, ...) | MTurk-Interact crowdsourced semantic-differential ratings |
| ``MTurkInteract_Modifiers.csv`` | Human EPA ratings of modifiers (``warmly``, ``angrily``, ...) | MTurk-Interact crowdsourced semantic-differential ratings |
| ``2010impressionformation.csv`` | Impression-formation coefficient sets (the ACT equations) | Published 2010 impression-formation equation sets (Heise 2007 convention) |
| ``user_inputs.json`` | 34 user inputs (conversation starters) | Collected for the paper's generation pipeline |
| ``all_truncated_outputs.json`` | 582 truncated model outputs (continuation stems) | Collected for the paper's generation pipeline |

The two JSON files are registered in ``DataPaths`` (existence checked by the
scaffold test); they feed the GPU-phase reader calibration and contrast
continuations. Contrast sets and direction artifacts are derived, seeded,
and written to ``artifacts/`` by the ``stimuli`` / ``directions`` CLIs — they
are never hand-edited.

## Limitations

- **GPU phase partially run (as of 2026-10-31).** Directions, demo traces,
  the reader gate, deflection reduction, and quality guardrails have run on
  ``Llama-3.1-8B-Instruct`` (see ``results/`` and ``GPU_RUNBOOK.md``);
  the full sycophancy probe, dose-response sweep, and the H3 battery
  (runbook steps 6–8) are still to run. The first reader gate **failed**
  (E r = −0.213, P r = +0.055, A r = −0.016 vs. the r ≥ 0.5 rule) — see
  ``results/H3_ANALYSIS.md``; no EPA-target number is publishable until a
  re-gate passes (refit levers: ``--direction-method cluster_mean``,
  ``--epa-templates``). Mock-mode results demonstrate the machinery, not
  the effect size.
- **Reader circularity.** A reader fitted on the steered model's own hidden
  states is diagnostic only; ``reader_validation`` exists to promote an
  independently validated metric to the headline one.
- **ACT was calibrated on human-human interaction.** Whether its
  deflection-minimising targets are appropriate for an *AI agent* in a role is
  an open hypothesis, not an assumption; the ablation matrix and the
  target-validity analysis are designed to test it.
- **Normative content.** ACT encodes role-normative (often status-graded)
  expectations; the quality guardrails (harshness, repetition, perplexity) and
  the sycophancy counter-check are the guardrails, but deployment decisions
  must weigh whose norms are being enforced.
- **In-loop contrast mode is Llama/Mistral-only** (upstream RepE limitation)
  and disables the KV cache; the static per-persona-pair mode is the portable
  path.
- **LoRRA persistence is optional (P2).** The triplet builder and training
  CLI are implemented (see [LoRRA persistence mode](#lorra-persistence-mode-optional));
  the fine-tune itself runs in the GPU phase behind the one-line upstream
  dataset hook, and is not part of the default pipeline.
