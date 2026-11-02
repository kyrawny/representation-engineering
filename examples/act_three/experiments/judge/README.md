# LLM judge for the ACT revision: how to run it

This folder scores the revision's model outputs with an LLM judge. That covers plan task T8 and the judge-dependent parts of T9, T10, T4 and T6 in `../../REVIEW_RESPONSE_PLAN.md`. It is self-contained: everything it reads is in `inputs/`, so you need no GPU, no torch and no other part of the repository. Results go to `outputs/<judge-model>/`; send that folder back when you are done.

**Status (2026-11-03):** run once with `gemini-3.6-flash`; the outputs are in `outputs/gemini-3.6-flash/`. `inter-judge` has not been run.

## 1. Setup (5 minutes)

```bash
cd examples/act_three/experiments/judge
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the key in your shell, not in a file in the repository:

```bash
export GEMINI_API_KEY=...            # Windows PowerShell:  $env:GEMINI_API_KEY="..."
```

The default judge is `gemini-3.6-flash` at temperature 0 with JSON output. Use `--model` to pick another model. `--backend anthropic` (with `ANTHROPIC_API_KEY` and `pip install anthropic`) uses `claude-sonnet-5-5` instead.

## 2. Check the pipeline offline (free, no key)

```bash
python run_judge.py validation --backend mock --out /tmp/judge_mock
python run_judge.py bulk --backend mock --out /tmp/judge_mock --limit 10
```

The `mock` backend returns random scores, so its numbers are meaningless; this only checks that everything runs.

## 3. A cheap real test

```bash
python run_judge.py j1 --generator llama3 --limit 5
```

Open `outputs/gemini-3.6-flash/j1_llama3.jsonl` and `judge_cache.jsonl`. Check that every response got eight integer scores from 1 to 7, and that `judge_cache.jsonl` shows sensible replies.

## 4. Validate the judge first (required before bulk scoring)

```bash
python run_judge.py validation
```

This makes about 145 calls and runs two checks:

- **`validate-humans`**: scores the 56 responses that 48 people rated (14 vignettes × 4 conditions). It correlates the judge with the mean human rating for appropriateness (a), naturalness (b) and non-sycophancy (c).
  - An item is **validated** if r ≥ 0.4 and its 95% CI (bootstrap over vignettes) excludes 0.
  - Unvalidated items are still scored, but the paper can only report them as secondary.
  - This also prints whether the judge reproduces the human ordering of the four conditions.
- **`validate-calibration`**: has the judge rate the EPA of 130 calibration utterances with known dictionary values. It reports r, MAE and CCC per dimension next to the EPA readers' r. For each dimension it records whether the judge or the cross-model reader is the better independent EPA measure.

Paste the printed summary into your message back. If (a) or (c) fails validation, still run step 5, but say so.

## 5. Bulk scoring

```bash
python run_judge.py bulk                     # j1 + ays + multiturn + j2 (subset) + order-bias
python run_judge.py inter-judge --backend2 gemini --model2 gemini-3.8-flash
```

| Task | What | Calls |
|---|---|---|
| `j1` | Appropriateness, naturalness, non-sycophancy, plus five decomposition items (flatters, gives in, assertive, harsh, addresses concern). Each call shows all conditions of one trial in a fixed random order. | 2,600 (1,300 Llama + 1,300 Ministral trials) |
| `ays` | Grades both turns of the "Are you sure?" benchmark (correct / incorrect / no answer; admits mistake). | 1,200 |
| `multiturn` | `j1` items for every assistant turn of the three-turn dialogues, with the earlier turns as context. | 1,200 |
| `j2` | EPA rating of every text in the stratified subset (`--scope all` for all 1,300 trials, about 3× more). | about 8,800 |
| `order-bias` | 100 trials re-scored in reversed order. | 200 (100 cached) |
| `inter-judge` | A second judge on a random 20% of the Llama trials. A Gemini second judge is the same vendor, so prefer a different vendor if available. | about 520 |

The total is roughly 14,000 calls: about 6–7 million input tokens and 1–2 million output tokens, plus any thinking tokens the model adds. Check current Gemini pricing before starting. To keep it cheap you can run tasks one by one (`j1`, then `ays`, ...) and skip `j2`.

**Resuming.** Every successful call is cached in `outputs/<model>/judge_cache.jsonl` (keyed by prompt version, model and exact prompt). If a run stops, or some calls fail (they are retried with back-off and reported at the end), run the same command again; only the missing calls are made.

**Rate limits.** Lower `--threads` (default 8) if you get many 429 errors.

**Do not edit the prompts** in `llm_judge.py`. The items (a)–(c) copy the human survey wording, and any change invalidates comparability and the cache. If a change is unavoidable, bump the version string in `PROMPT_VERSION`.

## 5b. ELEPHANT AITA-YTA (added 2026-11-03)

```bash
python run_judge.py aita --backend mock --out /tmp/judge_mock --limit 10   # offline check
python run_judge.py aita
```

This scores about 1,050 responses (150 posts × 6 conditions + 150 Reddit top comments) with ELEPHANT's three judge prompts, so about 3,150 short calls. The metrics are validation, indirectness and framing (Cheng et al., ICLR 2026).
- **Prompts.** They are ELEPHANT's own, copied verbatim into `elephant_prompts.py` from the commit named there. Our only changes: their system message goes at the top of the prompt, and one line at the end asks for the 0/1 answer as JSON (our backends require JSON). ELEPHANT itself used GPT-4o as the judge; we use Gemini.
- **Scale.** In all three metrics, 1 is the sycophantic label.
- **Output.** `outputs/<model>/aita_scores.jsonl`. Its cache entries use the prompt version `J4-elephant-v1`, so the earlier tasks' cache is unaffected.

The `inputs/aita_records.jsonl` file exists only after the authors have generated the responses and re-exported the bundle.

## 6. Send the results back

Commit or zip the whole `outputs/<model>/` folder, including `judge_cache.jsonl` (it holds the raw replies), and include the printed validation summary.

## What is in `inputs/`

- **Llama trials** (`j1_trials_llama3.jsonl`): every trial with the first 500 characters of each condition's response. That is the 4 original conditions on all trials, plus constant/shuffled/identity-only targets, CAA and the anti-sycophancy prompt on the stratified half.
- **Ministral trials** (`j1_trials_ministral.jsonl`): the 4 original conditions.
- **Benchmark and dialogues** (`ays_records.jsonl`, `multiturn.jsonl`): benchmark questions with gold answers and both responses; the three-turn dialogues.
- **Human study** (`human_vignettes.jsonl`, `human_response_means.csv`): the 56 survey responses and their **mean** ratings over 48 participants. No participant-level data is included.
- **Calibration** (`calibration_utterances.jsonl`, `reader_validity.json`): the 130 calibration utterances with dictionary EPA, and the readers' validity.
- **Manifest** (`manifest.json`): counts and source hashes.

All texts are generated by Llama-3.1-8B or Ministral-8B (plus the trivia questions and fixed prompts), so a Gemini judge never scores Gemini's own outputs (plan rule). The texts are sent to the judge provider's API.

## After the results come back (authors)

```bash
python -m examples.act_three.experiments.analysis.judge_results --model gemini-3.6-flash
```

This writes:

- `results/analysis/judge_validation.json` and `results/analysis/judge_results.json`, which include the primary contrasts (ACT-RepE vs constant target, shuffled target, CAA and anti-sycophancy prompt on appropriateness and non-sycophancy, Holm-corrected).
- The judge EPA version of Table 3.
- Judge-graded flip rates, with agreement against the earlier graders and the blind hand labels.
- Multi-turn slopes.
- `figures/fig_pareto_appropriate_vs_nonsyc.pdf`.

Then `make_revision_tables` (which calls `make_judge_tables`) writes the judge macros and `paper/generated/tab_judge.tex`, and `make_response` regenerates the response letter.

To rebuild `inputs/` from the experiment outputs (on the original machine):

```bash
python -m examples.act_three.experiments.analysis.export_judge_inputs
```
