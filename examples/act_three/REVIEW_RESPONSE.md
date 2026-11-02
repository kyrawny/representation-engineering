# Response to reviewers

We thank the reviewers. This letter is generated from the same macro file as the revised paper (`paper/generated/macros.tex`), so every number below is the number in the paper. Section names refer to the revised paper (`paper/AnonymousSubmission2027.tex`); code and outputs are in `experiments/revision/`, `experiments/analysis/` and `experiments/results/{revision,analysis}/`.

**What we could not do, up front.** (1) We had no LLM judge, so the appropriateness, naturalness and non-sycophancy of the *new* baselines (constant/shuffled targets, CAA, anti-sycophancy prompt) were rated by no one; the human study was not rerun (a new preregistered study with these conditions is the proper test and we offer it). (2) Persona vectors were not run (cited only). (3) ELEPHANT social-sycophancy and judge-based multi-turn scoring were not run. (4) Claude Sonnet 5 generations are not in our repository and were not re-scored. See `BLOCKERS.md`.

**What changed our conclusions.** The independent-reader analysis does not support several original claims, and we have withdrawn or weakened them: the ranking Hybrid < RepE ~ PE < Unsteered on Llama (only a small RepE gain survives independent scoring); "PE is best for Evaluation and RepE for Potency/Activity"; the pooled p-values of the human study; "RepE achieves a more normatively appropriate balance"; the description of sycophancy reduction as an "emergent property"; and the frontier-model comparison ("not an emergent property of capability"). We also found that the shipped Llama configuration file disables the orthogonalised directions that the paper and the logged runs use; revision runs force them on (details in `BLOCKERS.md`).

---

## Reviewer 1 (rating 5)

**R1.1 "The experiments do not clearly show what ACT adds over existing methods. CAA and persona vectors would be useful baselines."**
*Done.* We added a CAA sycophancy vector (public CAA dataset; layer 13 of 32, multiplier −2.0, chosen automatically by held-out log-odds change and a perplexity constraint) injected through the same steering pipeline, and an explicit anti-sycophancy system prompt. Persona vectors were *not* run; they are cited and discussed (Related Work) and CAA is our representation-steering baseline. *Result:* on distance to the ACT target (independent reader) ACT-RepE is closer than CAA by +0.053 (95% CI [0.029, 0.075], Holm p <0.001) and not distinguishable from the anti-sycophancy prompt (−0.010, CI [−0.032, 0.014]). On the benchmark, see R3.3. We cannot say whether CAA or the prompt is perceived as more or less sycophantic than ACT-RepE (no ratings). *Where:* Experimental Setup "Conditions, Ablations and Baselines"; Results "What Does ACT Add".

**R1.2 "RepE-only still uses ACT targets; compare fixed EPA targets with ACT targets, steering unchanged."**
*Done, with three variants under identical steering code:* constant target (mean ACT target; all 1,300 trials), the target of another identity pair for the same scenario (also a misspecified-identity test), and identity-only target (no reading of the message). *Result (independent reader, distance to ACT target, condition minus ACT-RepE):* constant +0.047 (CI [0.033, 0.061], Holm p <0.001, MDE 0.021); shuffled +0.038 (CI [0.023, 0.054]); identity-only +0.023 (CI [0.002, 0.046]). Constant and shuffled targets do not track the target (Potency slope −0.01, 0.02) while ACT-RepE does, weakly (0.13). Role reversal: the Potency difference between high- and low-status agents is +0.25 (CI [0.19, 0.32]) for ACT-RepE versus +0.04 for a constant target, against an ACT-prescribed gap of 2.01. *Interpretation (our reading, not a claim of better quality):* ACT targets do move outputs differently from fixed targets, mostly through the identity-dependent part; the gain is small, is smaller and not significant with the Mistral-7B reader (+0.020, Holm p 0.108), and no ratings exist to say it is better. *Where:* Results "What Does ACT Add".

**R1.3 "Compare with existing anti-sycophancy methods in the human evaluation (appropriateness, naturalness, non-sycophancy together)."**
*Not done as asked.* We had no judge and did not rerun the human study, so the new methods have no human ratings and no Pareto plot of appropriateness vs non-sycophancy could be produced. What we did instead: the epistemic benchmark for all conditions (R3.3), the mixed-model reanalysis of the existing human data (R3.4), and text-feature analyses. This remains the main gap and is stated as such in Limitations.

**R1.4 "Related work such as *Sycophancy Detection in Large Language Models* deserves discussion."**
*Done.* Related Work now covers CAA, persona vectors, linear-probe approaches to sycophancy, SycophancyEval, SYCON-Bench, ELEPHANT and the survey of Malmqvist (arXiv 2411.15287). **Note:** the arXiv paper we were pointed to for this reference is titled "Sycophancy in Large Language Models: Causes and Mitigations", not "Sycophancy Detection in Large Language Models"; we cite it under its verified title and could not identify another paper with the reviewer's title. If the reviewer meant a different paper we would be glad to add it. All new references were checked against their arXiv pages.

## Reviewer 2 (rating 3)

**R2.W1 "Primarily integrates existing ACT and RepE methods without substantial technical innovation."**
We have not changed the method and do not claim more than the ablations support: the contribution is the pairing of ACT targets with RepE, and we now show quantitatively how small and how identity-driven the ACT-specific part is (R1.2).

**R2.W2 / Q3 "The same EPA reading framework informs steering and evaluates it."**
*Done.* Every text is re-read (first 500 characters, all conditions alike) by readers built on other models (Ministral for Llama texts, Llama for Ministral texts, Mistral-7B for E and A; its Potency reader is empty), and we report VADER and each reader's validity on the 130-utterance calibration set. *Result:* the own reader agrees only moderately with the independent reader on Llama texts (r = 0.68 E, 0.47 P, 0.45 A). The original Llama ranking is not robust: Hybrid improves on unsteered by −0.110 under the own reader but −0.035 (CI [−0.056, −0.015]) under the independent one and −0.008 on trials with no truncated text; PE is *farther* from the target than unsteered (+0.015); RepE keeps a small gain (−0.032). For Ministral the ordering survives (Hybrid −0.120, RepE −0.083). The robust effect is on Potency. We also found that readings of truncated (500-character-limited) texts differ strongly from readings of full texts, which inflated the original Hybrid result. We did not use an LLM judge. *Where:* Results "How Circular..." and "Automated Steering Under Independent Measures".

**R2.W3 / Q5 "Single-turn only, predefined identities, no multi-turn evidence; uncertain identities; cultural diversity."**
*Partly addressed.* (a) "Are you sure?" is two-turn with the generic identities assistant/person and the method applied unchanged at each turn (R3.3). (b) Three-turn scripted dialogues with praise-plus-favour and pushback follow-ups: in 100 scripted conversations the share of turn-3 replies that give the requested concession (a lexical marker with high precision but low recall) is 0.21 for unsteered text versus 0.04 (ACT-RepE), 0.08 (ACT-Hybrid) and 0.16 (constant target); the turn-by-condition slope differs from unsteered for ACT-RepE (−0.08 per turn, p <0.001) but not for the constant target (p 0.301). This is a tone-level result: the marker was only checked on a small hand-labelled sample, the follow-ups are scripted and no judge was available; it does not match the answer-flipping benchmark, where ACT steering gives no gain. (c) Misspecified identities: with the target of the wrong identity pair, the effect disappears (ΔP +0.07 vs +0.25), so correct identities matter; we did not test identity inference. (d) Culture: no new data; stated in Limitations (US 2010 dictionary only).

**R2.W4 / Q6 "Metric choices, scenarios and identities are insufficiently justified."**
*Partly addressed.* New appendix tables list the 7 scenario categories (the paper wrongly said 10), their mean read EPA and the 13 identity pairs (6 role-reversed twins, 1 symmetric), and the share of target variance explained by identity pair (37%/82%/46% for E/P/A) and scenario (33%/12%/42%). *Not resolved:* the repository does not record how the scenarios were produced; the manuscript carries an explicit placeholder asking the authors to state it. We could not validate metrics against an LLM judge.

**R2.Q1 "Substantive sycophancy or merely more assertive responses?"**
(i) Epistemic test: flip rate under "Are you sure?" (R3.3). (ii) Human data: adding assertiveness/harshness proxies (reader Potency, negated Evaluation, imperative and exclamation rates) to the non-sycophancy mixed model changes the RepE advantage from 0.59 to 0.50 (attenuation 16%), i.e. most of it does not run through these proxies; the proxies come from the same reader, so this is a weak test and we say so. A judge-based decomposition was not possible.

**R2.Q2 "Why is Hybrid best automatically but worst for humans?"**
Under independent readers Hybrid is *not* clearly best for Llama (above). Hybrid texts are longer (+20 words vs unsteered) and more exclamatory (+0.13 per sentence). Across rated responses, being closer to the Activity target does not predict appropriateness (+0.42, p <0.001; larger deviation was associated with higher ratings). Exploratory, 56 responses.

**R2.Q4 "What does ACT add over simpler prompting or conventional steering baselines?"** See R1.1 and R1.2.

## Reviewer 3 (rating 5)

**R3.1 "Automated evaluation is circular; the reader is noisy; gains are modest."**
See R2.W2. Every gain is now reported with its minimum detectable effect (e.g. Llama Hybrid under the independent reader −0.035, MDE 0.029) and tracking slopes; outputs follow their targets only weakly (Potency slope 0.13 for ACT-RepE), and the Llama Potency and Activity readers are weak (r = 0.515, 0.315).

**R3.2 "The human study does not clearly support the central claim."**
We agree in part. Mixed models show RepE is rated more non-sycophantic than unsteered (+0.85, CI [0.49, 1.22]) with unchanged naturalness (+0.02, CI [−0.24, 0.28]) but lower appropriateness (−0.56, CI [−0.92, −0.20]); Hybrid is least natural and appropriate. The claim of a better appropriateness/sycophancy balance is removed. The vignettes come from an earlier 5-pair run, which is now disclosed.

**R3.3 "Sycophancy is measured with one Likert item rather than standard benchmarks."**
*Done for epistemic sycophancy:* SycophancyEval "Are you sure?" (200 TriviaQA items, two turns, generic identities, all conditions). Graded by a local yes/no grader and, as a check, alias matching (the two graders agree on 0.59 of turn-2 labels; against 37 blind hand labels the agreement is 0.76 (string) and 0.78 (local)); no external judge. *Result:* flip rate (local grader) unsteered 0.77 (CI [0.70, 0.84]); ACT-RepE Δ +0.05 (CI [−0.02, 0.13], Holm p 0.997; string grader Δ +0.09); constant target Δ −0.20 (string +0.07); CAA Δ +0.06 (string +0.09); anti-sycophancy prompt Δ −0.25 (CI [−0.35, −0.15]; string −0.01). Turn-1 accuracy falls under every steered condition (string grader: ACT-RepE 0.69, CAA 0.72 vs unsteered 0.82). **ACT steering does not reduce answer flipping** (no ACT-RepE or ACT-Hybrid contrast vs unsteered is significant after Holm correction, and the point estimates are positive under both graders), so the perceived-sycophancy reduction in the human study does not extend to epistemic sycophancy in our data; the anti-sycophancy prompt does lower flips under the local grader and in a small blind hand check (flip rate 0.21 vs 0.57), not under string matching. The grading is imperfect (see above) and the paper says so. *Not done:* ELEPHANT (needs a judge). *Where:* Results "Epistemic Sycophancy".

**R3.4 "Statistics treat repeated ratings as independent; a mixed-effects analysis is needed."**
*Done.* CLMMs with crossed random effects for participant and vignette (R `ordinal`), likelihood-ratio tests (χ² = 101.3, 91.5, 161.2), Holm-adjusted contrasts, a participant-level Friedman/Wilcoxon check and Krippendorff's α (0.14, 0.12, 0.17). The original "p < 10⁻²⁷"-style statements are removed. Main conclusions survive with wider intervals.

---

## Items offered but not completed
New human study with the added baselines and separate sycophancy items; persona vectors; ELEPHANT; judge-based scoring; cross-cultural dictionaries; Ministral replication of the ablations/benchmarks (see `BLOCKERS.md`).
