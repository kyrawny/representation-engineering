## GPU runbook

The CPU package is complete: every entry point runs in mock mode on any
machine. This section is the exact sequence to run the real (GPU) phase on
the 4090. All commands run from this directory
(`examples/act`), in the `repeng` environment.

**VRAM:** `Llama-3.1-8B-Instruct` in fp16 needs ~16–17 GB plus
activation headroom on the 24 GB card. If the local assistant model
(llama.cpp `llama-server`) is resident on the same 4090, unload it
first — the two do not fit together, so the load must be swapped
between the assistant and these GPU steps.

**Prompt framing (read before any generation step).** An instruct model
fed the flat `You are X. Context: … User: …` transcript does not answer
the scenario — it *continues* the template (repeats the setup, echoes the
trailing `User:` line). Every model-bound generation step below now
renders its prompt through the tokenizer chat template by default
(`--prompt-style chat`, added 2026-10-29):

- `act_steering.pipeline` (steps 2, 3b): the identity instruction goes
  into a system role; `--system-variant role|act_descriptors|anti_sycophancy`
  selects the B0/B1/B2 system prompt (default `role`).
- `evaluation.sycophancy_dialogue` (step 6): the QA item is a user
  message, the model's own initial answer an assistant message, and the
  pushback a fresh user turn.
- `evaluation.dose_response` (step 7): each probe prompt is a single
  user message.
- `evaluation.ablations` (step 8): battery scenarios render as a chat
  turn with the same `--system-variant` options.

All four also accept `--prompt-style raw` to reproduce the AAAI-27 flat
continuation strings byte-for-byte (baseline reproduction only), and
responses are stripped of the prompt prefix, whitespace and trailing
`<|eot_id|>` / `<|end_of_text|>` tokens before scoring.

**Consequence:** traces and dialogue logs generated before this change
were produced on the flat template — the model answered the template
instead of the scenario — and must be regenerated from step 2 onward.
Do not mix old and new logs in any comparison.

**Prereqs (CPU, one time):**

```bash
# 0. contrast set (1,300 identity pairs; feeds direction extraction + LoRRA)
python -m act_steering.stimuli --out artifacts/contrast_set.json
# 0b. scenario battery (500 one-turn scenarios; feeds the H3 ablation matrix)
python -m act_steering.battery --out artifacts/scenario_battery.json
```

**Step 1 — directions (GPU; the only prerequisite artifact).**
Fits the EPA reader directions and the deflection-contrast direction on
`Llama-3.1-8B-Instruct`; writes `artifacts/directions_META-Llama-3-8B-Instruct.npz`
plus a JSON sidecar (model name, layers, raw cosines, seed):

```bash
python -m act_steering.directions --model meta-llama/Llama-3.1-8B-Instruct --lowdin
# smoke first if the 4090 is shared (20 pairs, 10/side, layers 0,16,31):
python -m act_steering.directions --quick
```

**Refit (re-run step 1, then everything downstream).** The directions are
fitted on the contrast set alone (machine-generated EPA labels). Once the
human EPA ratings (WS2) exist, or if the step-3 reader gate fails, refit:
regenerate the contrast set with a fresh seed and/or augment it, then
re-run step 1 — the artifact name is unchanged, so re-run steps 2–8 after
a refit. Useful levers: `--contrast-set`, `--n-epa-per-side`, `--seed`.

```bash
python -m act_steering.stimuli --out artifacts/contrast_set.json --seed 20271004
python -m act_steering.directions --model meta-llama/Llama-3.1-8B-Instruct --lowdin
```

**Reader-gate levers (WS1.8 phase 1, cheapest first).**

1. *Protocol match, no refit needed.* Re-run step 3 as-is. The gate now
   wraps the validation terms in the template the directions were fit
   with (read from the bundle metadata; `--stimulus-template fit` is the
   default; use `--stimulus-template bare` for the old bare-term protocol
   as a robustness check). The first real gate run scored bare terms
   against templated fits — part of its failure may be protocol mismatch,
   not only direction quality. If this alone passes the gate, stop here.
2. *Difference-of-means directions.* Refit with `cluster_mean` instead of
   PCA: a more robust estimator of the high−low difference that needs
   mixed pair orderings (auto-mixed from `--seed`; override with
   `--mix-seed`).

   ```bash
   python -m act_steering.directions --model meta-llama/Llama-3.1-8B-Instruct --lowdin --direction-method cluster_mean
   ```
3. *Multi-template fitting.* Fit on several surface templates and average
   the signed unit directions per layer (cancels template-specific noise):

   ```bash
   python -m act_steering.directions --model meta-llama/Llama-3.1-8B-Instruct --lowdin \
       --direction-method cluster_mean \
       --epa-templates "You respond by {behaviour}.,Your reply is: {behaviour}.,You act by {behaviour}."
   ```

   The gate then scores terms wrapped in the *first* template (override
   with a literal `--stimulus-template`).

Every refit also saves the pre-orthonormalisation signed unit directions
as `epa.{dim}.raw` in the bundle; the reader and the gate prefer those for
measurement (Løwdin-orthonormalised directions remain the steering set).
If the gate still fails after levers 1–3, proceed to WS1.8 phase 2
(ridge-regression reader, separate workstream) or M-X (reader on a
different model).

**Step 2 — pipeline traces (GPU).** One trace log per condition. The
unsteered condition is `--alpha 0.0` (the steerer wrapper stays on, but the
offsets are zero — a clean B0):

```bash
# steered (run after step 3b: add --monitor-threshold <th> to the --repair line)
python -m act_steering.pipeline --script scripts/demo.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --alpha 0.5 --k 4 --monitor-positions 4 --repair \
    --out results/trace_steered.json
python -m act_steering.pipeline --script scripts/demo.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --alpha 0.0 --k 4 \
    --out results/trace_unsteered.json
```

Script JSON: `{"user": "student", "agent": "teacher", "messages": ["..."]}`
with identity terms from the MTurk-Interact dictionary. Each trace log holds
the per-turn monitor traces, candidate scores and winner index.

`--token-pos` (default `end`) controls where the steering vector applies:
`end` hits the last prompt position plus every new token (CAA-style tail
steering, the default); `none` steers every position, which is what the
AAAI-27 implementation used — use it when reproducing that baseline.
`--prompt-style` (default `chat`) / `--system-variant` (default `role`,
B0) control the generation framing — see “Prompt framing” above.

**Step 3 — reader gate (GPU; runs before any headline number).**
Cross-validated agreement of the hidden-state reader against the human EPA
ratings in the dictionary (this is the metric-promotion step — the reader
built on the steered model is diagnostic only):

```bash
python -m evaluation.reader_validation --reader model \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n-items 60 --out results/reader_validation.json
```

The gate wraps each validation term in the fit template by default
(`--stimulus-template fit`) so the reader is scored in the same stimulus
format the directions were fit on; `--stimulus-template bare` reproduces
the legacy bare-term protocol for robustness checks.

**Gate rule:** the reader may score/steer a dimension only if its
held-out agreement with human EPA ratings on that dimension is adequate
(target: Pearson `r >= 0.5`; the AAAI-27 Activity reader at `r = 0.315`
was too weak to drive or score steering). If any dimension fails:
refit the directions (step 1, refit note) and/or rebuild the reader with
the human ratings as extra training data (WS1.8), and until then use
human ratings or the LLM judge (M-J) as the primary metric for that
dimension — do not publish reader-scored numbers for it.

**Step 3b — monitor recalibration (GPU, before any `--repair` run).**
The per-token deflection monitor's exceedance threshold defaults to 1.0
in score units, which is arbitrary. Calibrate it on an *unsteered* trace
so the baseline firing rate is ~5–10% (the monitor should rarely fire
without steering, and `--repair` should only rescue genuinely deflected
tail tokens):

```bash
# 1. unsteered trace with the monitor on (no repair)
python -m act_steering.pipeline --script scripts/demo.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --alpha 0.0 --monitor-positions 4 \
    --out results/trace_unsteered_monitor.json

# 2. pick the threshold from the per-token scores (negative = deflection;
#    a response fires when any token score < -threshold)
python - <<'EOF'
import json

d = json.load(open("results/trace_unsteered_monitor.json"))
turns = [t for t in d["turns"] if t.get("monitor")]
for th in (0.5, 1.0, 2.0, 3.0, 5.0):
    fires = sum(any(s < -th for s in t["monitor"]["scores"]) for t in turns)
    print(f"threshold={th}: response firing rate = {fires}/{len(turns)} "
          f"({100.0 * fires / len(turns):.1f}%)")
EOF

# 3. re-run the steered trace (step 2) with the chosen threshold:
#    ... --monitor-threshold <th> --repair
```

**Step 4 — deflection reduction (CPU, over the traces).** Paired per-turn
``U`` differences with cluster-bootstrap CIs:

```bash
python -m evaluation.deflection_reduction \
    --log unsteered=results/trace_unsteered.json \
    --log steered=results/trace_steered.json \
    --baseline unsteered --treatments steered \
    --out results/deflection_reduction.json
```

**Step 5 — quality guardrails (GPU for the reference scorer).**
Repetition / length / perplexity under a *separate* reference model —
per WS2, use a model the steered model is not:

```bash
python -m evaluation.quality_guardrails \
    --input unsteered=results/trace_unsteered.json \
    --input steered=results/trace_steered.json \
    --scorer model --reference-model Qwen/Qwen2.5-7B-Instruct \
    --out results/quality_guardrails.json
```

**Step 6 — sycophancy probe (GPU, two dialogue logs + CPU analysis).**
Generate the are-you-sure dialogue log per condition (steered condition
reuses the same alpha as step 2), then analyse:

```bash
python -m evaluation.sycophancy_dialogue --generate model \
    --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.0 \
    --items data/are_you_sure.jsonl --out results/dialogue_unsteered.json
python -m evaluation.sycophancy_dialogue --generate model \
    --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.5 \
    --items data/are_you_sure.jsonl --out results/dialogue_steered.json
python -m evaluation.sycophancy_probe \
    --log unsteered=results/dialogue_unsteered.json \
    --log steered=results/dialogue_steered.json \
    --baseline unsteered --treatments steered \
    --out results/sycophancy_probe.json
```

`--items` accepts either a JSON list of `{"id", "question", "correct"}` or
the sycophancy-eval `are_you_sure.jsonl` directly (meg-tong/sycophancy-eval,
Sharma et al. 2024): MCQ rows (aqua_mc / math_mc_cot / mmlu_mc_cot /
truthful_qa_mc) map to their `correct_letter`, trivia_qa rows to their
`correct_answer`, and the 817 open `truthful_qa` rows (no ground truth) are
skipped — **4,071 usable items** in the full file, so ~8.1k generations per
condition. Smoke-run with `--max-items 100` first; a mock pipeline test is
`--generate mock`.

The dialogue runs as a real chat transcript (see “Prompt framing”):
the model's own initial answer is an assistant message and the pushback
is a new user turn — with the old flat framing an instruct model
echoed the template instead of reconsidering. `--prompt-style raw`
reproduces the AAAI-27 strings.
The harshness field currently uses `--judge mock|none`; the validated LLM
judge (frontier-model tokens) plugs in as the judge source.

**Step 7 — dose-response sweep (GPU).** Sweeps the steering strength on a
fixed prompt set, measuring per-dimension reader shift vs. reference-model
perplexity and repetition; reports `degenerated_at` and `safe_alpha_max`:

```bash
python -m evaluation.dose_response --probe model \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reference-model Qwen/Qwen2.5-7B-Instruct \
    --prompts data/probe_prompts.json --alphas 0 0.25 0.5 1.0 1.5 2.0 \
    --out results/dose_response.json
```

`data/probe_prompts.json` is a JSON list of prompts (e.g. the conversation
starters from `data/act/user_inputs.json`). Each prompt is wrapped as a
single user message in the chat template by default (`--prompt-style
raw` passes it through flat, AAAI-27 reproduction).

**Step 8 — H3 ablation matrix (GPU; the decisive experiment).**
Runs every ablation condition over the scenario battery (500 one-turn
scenarios from `artifacts/scenario_battery.json`) and reports per-condition
post-response deflection with **cluster bootstrap CIs** (clusters =
scenarios) plus paired contrasts against `unsteered`. One steerer is built
per condition (one model wrap); `random_dir` / `shuffled` draws vary per
scenario via a seeded index. This is the H3 question: does the
situation-dependent target (`situation_dependent`, M1) beat the constant
(B8), identity-only (B6) and shuffled-identity (B7) targets?

```bash
# smoke first (~10 scenarios, 3 conditions):
python -m evaluation.ablations --battery artifacts/scenario_battery.json \
    --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.5 \
    --max-scenarios 10 \
    --conditions unsteered situation_dependent constant_vec \
    --out results/ablations_smoke.json
# full matrix: 10 conditions × 500 scenarios (≈ 8,000 generations,
# the k=4 conditions generate 4 candidates each; expect several hours):
python -m evaluation.ablations --battery artifacts/scenario_battery.json \
    --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.5 \
    --out results/ablations_battery.json
```

CPU wiring smoke (no GPU; deterministic mock components — the numbers
are a wiring check, not a signal):

```bash
python -m evaluation.ablations --battery artifacts/scenario_battery.json \
    --out results/ablations_battery_mock.json
```

The battery scenarios render through the chat template in model mode
(`--prompt-style chat` default) with the same `--system-variant`
choices as step 2, so the B0/B1/B2 contrast is available at the
ablation level as well.

**Not yet GPU-wired (mock only):** `act_steering.lorra --train --execute`
is wired and prints the exact upstream fine-tune command; it additionally
needs the one-line dataset hook in `lorra_finetune/src/train_val_datasets.py`
(see the LoRRA section).

**Ordering notes.** Step 1 gates everything (the directions artifact);
steps 3 (reader gate) and 3b (monitor threshold) gate the `--repair`
rerun of step 2; steps 4 and the probe analysis are CPU-only and can run
while step 5–8 generate. Steps 5–7 each load a second model (the reference) — run them
one at a time on the 4090, or pass `--device cpu` on the reference-heavy
steps if VRAM is tight. Step 8 loads a single model and is the longest
GPU step. Every report JSON embeds `meta` (model, seed, git hash, config)
so each published number traces back to a log.
