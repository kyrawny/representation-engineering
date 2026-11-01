# H3 analysis — M1 (situation-dependent) vs target-family ablations (B6/B7/B8)

Status: **decision pending GPU run.** The ablation machinery (T3) is complete
and CPU-verified; the decisive comparison itself requires the model-mode
battery run (GPU runbook step 8), which has not happened yet. This document
records what the current evidence does and does not support, and the
decision for the next phase.

## 1. Evidence inventory

| Evidence | State | Signal? |
|---|---|---|
| Mock battery report (500 scenarios × 10 conditions, `results/ablations_battery_mock.json`) | Done (T3-S4) | **No** — deterministic mock components; wiring check only |
| Model-mode battery (`results/ablations_battery.json`) | **Not run** (runbook step 8) | — |
| Reader gate (`results/reader_validation.json`, 5-fold CV vs human EPA in the dictionary) | Done, real | **FAIL — see below** |
| 2-turn demo deflection (`results/deflection_reduction.json`) | Done, real | No (n = 2 turns; CI crosses 0) |
| 12-item mock sycophancy probe (`results/sycophancy_probe.json`) | Done, mock | No (wiring only) |
| Full sycophancy dialogues (4,071 items × unsteered/steered) | Not run (runbook step 6) | — |

## 2. The reader gate fails (critical finding)

`reader_validation --n-items 60` on `Llama-3.1-8B-Instruct` (directions
fit from the 1,300-pair contrast set, Løwdin-orthonormalised):

| Dimension | held-out Pearson r | CCC | Verdict (gate: r ≥ 0.5) |
|---|---|---|---|
| E | **−0.213** | −0.078 | FAIL (inverted) |
| P | +0.055 | +0.050 | FAIL |
| A | −0.016 | −0.016 | FAIL |

The hidden-state directions derived from the machine-labelled contrast set
do **not** linearly read out human EPA semantics of the model's own
responses on this model. Consequences:

1. No EPA-targeted number (distance-to-target, tracking, and — in
   model mode — the battery's `u_post`, which is scored through the
   behaviour reader) is publishable as "human-EPA" evidence until the
   reader is rebuilt and re-gated. This is exactly the W2 risk the
   revision plan targets (circular, noisy automated metric).
2. The **relative** ablation comparison (M1 vs B6/B7/B8, all scored by
   the same reader) is still a valid *mechanistic* comparison — it tests
   whether the ACT target family changes model behaviour in the reader's
   own space — but it cannot be framed as tracking human ratings.
3. The AAAI-27 paper's own reader was weak in places (Activity r ≈ 0.315
   was already flagged); this fit is worse, and on a different model, so
   a directions/reader rebuild (WS1.8) is mandatory before headline
   EPA numbers.

## 3. What the H3 verdict will require (decision rule)

When the step-8 report (`results/ablations_battery.json`) exists, H3
("situation-dependent targets beat constant/identity-only/shuffled at
matched non-sycophancy") is supported **only if**:

1. `situation_dependent` (M1) `u_post` CI lies below `constant_vec` (B8),
   `identity_only` (B6) and `shuffled` (B7) — cluster-bootstrap contrast
   vs each, excluding 0. (M1 vs B8 is the *decisive* contrast: it
   separates "ACT helps" from "any good default helps".)
2. M1 does not increase the sycophancy flip rate vs `unsteered`
   (full 4,071-item probe, runbook step 6) — the "matched
   non-sycophancy" leg of H3.
3. After the reader rebuild (below), the EPA-tracking version of the
   comparison reproduces the ordering qualitatively on re-gated data.

If M1 does **not** beat B8: follow revision-plan §6 — restrict to the
scenario subset where targets vary most (the battery records
per-scenario targets in the trace logs for this), check whether ACT
helps there, and otherwise reframe to "interpretable affect
specification" and lower the claims.

## 4. Next-phase decision

In order:

1. **Run the GPU battery (runbook step 8)** — smoke (10 scenarios,
   3 conditions) then the full 10-condition × 500-scenario matrix
   (~8,000 generations; hours on the 4090). This is the mechanistic
   M1-vs-B6/B7/B8 comparison.
2. **Rebuild the reader, re-run the gate (runbook step 1 refit + step 3)**
   before publishing any EPA number: refit directions with a fresh
   contrast seed; if the gate still fails, apply WS1.8 (difference-in-means
   directions, multi-component readers, and — once available — human EPA
   ratings as reader training data with a held-out validation split).
3. **Run the full sycophancy probe (runbook step 6)** for the
   non-sycophancy leg of H3 (flip rate, ~8,100 generations per condition).
4. **External resources (decision point):** the metric-promotion and
   human-study workstreams (WS2.2 human EPA ratings, ~400 responses ×
   5 raters; WS4 preregistered study; M-J LLM judge) require paid
   frontier-model tokens and/or rater compensation — not available on
   the local 4090. Recommend budgeting these after step 1's result
   decides whether the mechanistic comparison is worth the spend.

## 5. Bottom line

The machinery question ("can we run the H3 matrix?") is answered: yes —
CPU-verified, one steerer wrap per condition, per-scenario seeded draws,
cluster-bootstrap CIs, and a copy-paste runbook. The science question
("does M1 beat B8?") is **not yet answerable**: the model-mode battery
has not been run, and the reader that would let us score EPA targets
against human ratings currently fails its validation gate, so even a
positive battery result is only a mechanistic signal until the reader
rebuild passes.
