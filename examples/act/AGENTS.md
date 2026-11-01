# AGENTS.md — `examples/act` (ACT-RepE revision implementation)

Reference document for agents resuming work in this directory. Read this first,
then `README.md` (module details) and `GPU_RUNBOOK.md` (GPU commands).

## What this is

The revision implementation of the AAAI-27 paper *Affective Intelligence:
Steering LLM Responses Through Affect Control Theory and Representation
Engineering* (submission #00, rating 3). It steers conversational LLM
generation to **minimize ACT deflection** (deflection-minimising, role-
normative responses) using the upstream `repe` package, with a multi-turn
control loop (monitor → steer → ACT best-of-k select → repair) and a
measurement stack designed to answer the reviewers' circular-metric objection
(W2) and the sycophancy-as-tone objection (W1).

The revision plan (workstreams WS0–WS7, hypotheses H1–H5, the decisive H3
experiment) lives at
`C:\Users\Kyra\Documents\Repos\affective-intelligence\AGENTS.md`.
Paper + reviews: wiki pages `entities/act-repe-aaai-paper` and
`syntheses/act-deflection-steering-via-repe`.

## ⚠️ GPU rules (read before running anything)

1. **The assistant agent runs as the local model on the RTX 4090**
   (llama-server). Its weights occupy the 4090 for the whole session.
2. **Never run GPU code while the agent session is active**, and never load
   a model on CPU (16 GB of weights will OOM the machine).
3. All GPU steps are executed **by the user**. Give the user copy-paste
   commands (from `GPU_RUNBOOK.md`), then wait for the output files to appear
   in `results/` / `artifacts/` before analyzing.
4. CPU-only work is always fine: `pytest`, `ruff`, mock-mode CLIs, reading and
   analyzing result JSONs.

## Environment

- Python: `C:/Users/Kyra/mambaforge-pypy3/envs/repeng/python.exe`
  (mambaforge env `repeng`; CPython 3.11, torch 2.7.x CUDA, transformers 4.52.x).
- Repo root: `C:\Users\Kyra\Documents\Repos\qwen\representation-engineering`
  (qwen fork of `andyzoujm/representation-engineering`, `pip install -e .`).
- From this directory: `python -m pytest` (374 tests, all CPU) and
  `python -m ruff check .` (line length 100, isort, bugbear, pyupgrade).
- Default model: `meta-llama/Llama-3.1-8B-Instruct`. Artifact filenames use a
  **lowercase slug** of the model id:
  `artifacts/directions_meta-llama-llama-3-1-8b-instruct.npz` (+ `.json`
  sidecar). Don't hardcode other casings.

## Current state (as of 2026-10-31)

**Done (CPU-verified, all tests passing):**

- **T1 — scaffold:** `act_core` (two-event ACT, closed-form optimal
  behaviour, bounded least squares), `dictionary` (MTurk-Interact EPA
  lookups), `stimuli` (1,300-pair distribution-matched deflection contrast
  set), `state` (conversation fundamentals/drift), `monitor` (per-token
  deflection trace + threshold trigger), `steerer` (RepE contrast steering),
  `selector` (ACT best-of-k over actual text), `pipeline` (multi-turn loop +
  JSON trace logs), `lorra` (deflection-triplet builder + training CLI), and
  six evaluators in `evaluation/`.
- **T2 — GPU model modes wired:** `readers.py` (pure projection core +
  model-bound EPA reader and separate reference-model scorer), pipeline real
  mode, model modes for `reader_validation` / `quality_guardrails` /
  `dose_response`, and `GPU_RUNBOOK.md` (steps 1–8).
- **T3 — real ablation matrix (H3):** 10 labeled conditions
  (B0 unsteered, B1 PE, B2 anti-sycophancy prompt, B3 CAA sycophancy vector,
  B5 random direction, B6 identity-only, B7 shuffled identity, B8 constant
  target, M1 ACT-RepE, M2 Hybrid) × 500 scenarios through the full loop;
  cluster-bootstrap CIs (clusters = scenarios) and paired per-condition
  contrasts; model-mode provider wired; mock battery verified end-to-end.
- **T4 — reader refit levers (WS1.8 phase 1):** `--direction-method
  {pca,cluster_mean}` (difference-of-means, seeded mixed pair orderings),
  `--epa-templates` multi-template fitting (per-template fit + signed unit
  direction averaging), bundles now save `epa.{dim}.raw` pre-orthonormalised
  directions with provenance meta (legacy bundles still load; reader and gate
  prefer `.raw`), and the gate gained `--stimulus-template` (default `fit`:
  wraps terms in the template the directions were fit with).

**GPU runs already done (user, Oct 30–31):** runbook step 1 (directions),
step 2 (2-turn demo traces, steered + unsteered), step 3 (reader gate —
**failed**, see below), step 4 (deflection reduction, n=2 turns), step 5
(quality guardrails), LoRRA triplet build.

**Critical finding — the reader gate FAILS**
(`results/reader_validation.json`, 2026-10-31, 5-fold CV vs the human EPA
ratings in the dictionary; gate rule: held-out Pearson `r >= 0.5`):

| Dimension | Pearson r | Verdict |
|---|---|---|
| E | **−0.213** (inverted) | FAIL |
| P | +0.055 | FAIL |
| A | −0.016 | FAIL |

Two compounded causes: (a) **protocol mismatch** — directions were fit on
templated text `"You respond by {term}."` while the first gate run scored
*bare* terms (fixed in T4: the gate now wraps by default); (b) **fit-method
mismatch** — PCA on extreme pairs doesn't read out human EPA across the full
range (the WS1.8 problem; levers 2–3 of T4 address it). Consequences:
- **No EPA-target number is publishable** (distance-to-target, tracking,
  model-mode `u_post`) until a re-gate passes.
- The **relative** ablation comparison (M1 vs B6/B7/B8, one reader for all)
  remains a valid *mechanistic* comparison.
- Full analysis + decision rule: `results/H3_ANALYSIS.md`.

**Pending, in order** (user runs the GPU parts; commands in `GPU_RUNBOOK.md`):

1. **Reader re-gate** (T4-S6.2): re-run step 3 as-is (protocol match, no
   refit); if still failing, `cluster_mean` refit (step 1 levers) + re-gate;
   if still failing, multi-template refit + re-gate.
2. **H3 battery** (step 8): 10 conditions × 500 scenarios (~8,000
   generations) → `results/ablations_battery.json`.
3. **Full sycophancy probe** (step 6): 4,071 are-you-sure items × 2 dialogue
   logs (data in `data/are_you_sure.jsonl`, `data/answer.jsonl`,
   `data/feedback.jsonl`).
4. **Dose-response sweep** (step 7).
5. **External resources** (decision point, `H3_ANALYSIS.md` §4): WS2.2 human
   EPA ratings (~400 responses × 5 raters), WS4 preregistered human study,
   M-J LLM judge — need rater compensation and/or paid frontier-model tokens,
   not available on the local 4090.

## Code map (details in README.md)

```
act_steering/
  config.py        # DataPaths, ExperimentConfig, seeds — torch-free
  dictionary.py    # MTurk-Interact identities/behaviours/modifiers EPA
  act_core.py      # impression formation, deflection U, closed-form b*
  stimuli.py       # contrast sets; RESPONSE_TEMPLATE = "You respond by {behaviour}."
  state.py         # ConversationState (fundamentals, drift, triggers)
  directions.py    # EPA + deflection direction extraction; DirectionsBundle
  monitor.py       # per-token deflection trace, threshold, repair trigger
  steerer.py       # per-turn contrast steering (RepE / CAA-style tail)
  selector.py      # ACT best-of-k over generated text
  pipeline.py      # multi-turn loop; --mock / --quick / real model mode
  readers.py       # EPA reader + reference-model scorer (pure core + model-bound)
  lorra.py         # deflection-triplet builder + LoRRA training CLI (optional P2)
evaluation/
  reader_validation.py   # reader-vs-human gate (M-H agreement per dimension)
  deflection_reduction.py# paired per-turn U differences, cluster bootstrap
  quality_guardrails.py  # repetition / length / perplexity (separate ref model)
  sycophancy_probe.py    # flip rate + harshness (are-you-sure)
  sycophancy_dialogue.py # dialogue log generation for the probe
  ablations.py           # 10-condition H3 matrix, mock + model mode
  dose_response.py       # alpha sweep, degeneration threshold
  stats.py               # shared JSON report + meta (model, seed, git hash)
```

## Invariants and gotchas

- **Layering:** model-free modules (`config`, `dictionary`, `act_core`,
  `stimuli`, `state`, `monitor`, `selector`, `readers` pure core, `stats`)
  must import and run without `torch`. Model-bound code imports torch at
  module level and is GPU-gated.
- **Every CLI** supports `--quick` (CPU smoke) and `--seed`; every report is
  JSON with run metadata (`model`, `seed`, `git` hash, `config`) so each
  number traces to a log.
- **Direction fit protocol:** EPA contrast items are fit on the *bare*
  template `"You respond by {behaviour}."` — no role context (unlike the
  deflection contrast set, which uses full trial context). The gate's
  `--stimulus-template fit` matches this; keep fit and score protocols
  aligned whenever you change either side.
- **`.raw` directions:** new bundles store `epa.{dim}.raw` (pre-Løwdin
  signed unit directions). The reader/gate prefer them for measurement;
  Løwdin-orthonormalised directions remain the steering set. Legacy bundles
  load fine and fall back to canonical.
- **Chat template:** all model-bound prompts must go through the model's
  chat template. A raw-prompt bug broke the first GPU traces (2026-10-31);
  the fix followed the `affective-intelligence` repo's Llama template
  implementation. Don't regress this.
- **Reader circularity rule:** a reader built on the steered model's hidden
  states is *diagnostic only*; headline claims use the gate-validated metric
  (and eventually M-X on a different model / human ratings).
- **Data:** human data lives at repo-level `data/act/` (via `DataPaths`):
  the MTurk-Interact CSVs are **human** EPA ratings of words (the paper's
  dictionary source), plus the 2010 impression-formation coefficient set.
  Sycophancy benchmark data lives in `examples/act/data/`. Never treat the
  machine-generated contrast set as human ratings.
- **No external code reuse:** nothing is copied from other forks; ACT
  algorithm details may reference the published literature (Heise 2007) only.
- **Style:** full type hints, Google-style docstrings, frozen dataclass
  configs, `pathlib`, no hardcoded paths (everything via `DataPaths` or CLI),
  seeds derived from `ExperimentConfig.seed`, `logging` not `print` in
  library code. See README.md "Style guide".

## Wiki (project knowledge base)

The project wiki is at `C:\Users\Kyra\Documents\llm-wiki`.

- **At session start:** `wiki_recall` for the task at hand.
- **After significant work:** `wiki_observe` (atomic observation) and
  `wiki_retro` (durable insight) — future sessions surface them automatically.
- **Key pages:**
  - `syntheses/act-deflection-steering-via-repe` — design synthesis
    (architecture, open questions, M4 best-of-k rationale).
  - `entities/act-repe-aaai-paper` — paper + reviews + verdicts.
  - `entities/act-repe-revision-codebase` — current codebase state
    (this AGENTS.md is the in-repo mirror; keep both in sync when state
    changes materially).
  - observations `obs-2026-10-31-llama-3-1-8b-reader-gate-fails-*` (gate
    failure), `obs-2026-10-31-epa-direction-fit-protocol-*` (fit protocol),
    `obs-2026-10-31-gpu-traces-broken-raw-prompt-*` (chat-template bug).
