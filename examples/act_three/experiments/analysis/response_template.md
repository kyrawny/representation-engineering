# Response to reviewers

We thank the reviewers. This letter is generated from the same macro file as the revised paper (`paper/generated/macros.tex`), so every number below is the number in the paper. Section names refer to the revised paper (`paper/AnonymousSubmission2027.tex`); code and outputs are in `experiments/revision/`, `experiments/analysis/`, `experiments/judge/` and `experiments/results/{revision,analysis}/`.

**What we could not do, up front.**
1. The new baselines have no *human* ratings. Instead, an LLM judge ({{JudgeModel}}) rates every condition. It was validated against the existing human ratings, but only one judge was run, so inter-judge agreement is unmeasured. A new preregistered human study with these conditions remains the proper test, and we offer it.
2. Persona vectors were not run (cited only).
3. ELEPHANT social sycophancy was not run.
4. Claude Sonnet 5 generations are not in our repository and were not re-scored.

See `BLOCKERS.md`.

**What changed our conclusions.** Independent measurement does not support several original claims, and we have withdrawn or weakened them:
- the ranking Hybrid < RepE ~ PE < Unsteered on Llama (only a small RepE gain survives independent scoring);
- "PE is best for Evaluation and RepE for Potency/Activity";
- the pooled p-values of the human study;
- "RepE achieves a more normatively appropriate balance";
- the description of sycophancy reduction as an "emergent property";
- the frontier-model comparison ("not an emergent property of capability").

Most importantly, **we no longer claim that ACT-guided steering reduces sycophancy**. Over the full run, a judge validated on the human ratings rates ACT-RepE as less appropriate than unsteered text and not more non-sycophantic. Non-ACT targets and a CAA vector are as non-sycophantic as ACT-RepE, and an anti-sycophancy prompt is better on appropriateness and non-sycophancy. On "Are you sure?" ACT-RepE makes the model abandon correct answers *more* often. The claim that survives is narrower: ACT targets move expressed affect, mainly Potency, toward role-specific values better than fixed or misassigned targets do.

We also found that the shipped Llama configuration file disables the orthogonalised directions that the paper and the logged runs use. Revision runs force them on (details in `BLOCKERS.md`).

**The judge.** {{JudgeModel}} (temperature 0; a different vendor from both generators) rates each trial with all conditions' responses shown together in a fixed random order.
- **Items.** It uses the three survey items (appropriate, natural, NOT sycophantic) plus five decomposition items: flatters, gives in, assertive, harsh, addresses the concern. It also gives an EPA rating of every response and grades both turns of "Are you sure?".
- **Validation against humans.** On the 56 human-rated responses, it correlates with the mean human ratings at r = {{JdHumAR}} (CI {{JdHumACi}}), {{JdHumBR}} ({{JdHumBCi}}) and {{JdHumCR}} ({{JdHumCCi}}). All three items pass the bar fixed before scoring (r ≥ 0.4, CI excluding 0).
- **Validation against dictionary EPA.** On the 130 calibration utterances it reaches r = {{JdCalER}}, {{JdCalPR}} and {{JdCalAR}} (E, P, A). That is better than every reader on E and A, and below the Ministral reader on P ({{ReaderMinistralPR}}).
- **Order effects.** Reversing the presentation order on {{JdOrderN}} trials moves no item's mean by more than {{JdOrderMaxSigned}}.

---

## Reviewer 1 (rating 5)

**R1.1 "The experiments do not clearly show what ACT adds over existing methods. CAA and persona vectors would be useful baselines."**

*Done.* We added two baselines:
- a CAA sycophancy vector (public CAA dataset; layer {{CaaLayer}} of 32, multiplier {{CaaMult}}, chosen automatically by held-out log-odds change and a perplexity constraint), injected through the same steering pipeline;
- an explicit anti-sycophancy system prompt.

Persona vectors were *not* run; they are cited and discussed (Related Work), and CAA is our representation-steering baseline.

*Result, distance to the ACT target (independent reader).* ACT-RepE is closer than CAA by {{AblCaaDelta}} (95% CI {{AblCaaCi}}, Holm p {{AblCaaP}}). It is not distinguishable from the anti-sycophancy prompt ({{AblAntisycpromptDelta}}, CI {{AblAntisycpromptCi}}). The judge's EPA ratings agree for CAA ({{JdEpaLlamaCaaVsRepE}}) but put the prompt *closer* to the target than ACT-RepE ({{JdEpaLlamaAntisycpromptVsRepE}}, CI {{JdEpaLlamaAntisycpromptVsRepECi}}).

*Result, response quality (judge, condition minus ACT-RepE, 7-point scale).* CAA is indistinguishable from ACT-RepE on every item: appropriateness {{JdLlamaCaaVsRepEA}} (CI {{JdLlamaCaaVsRepEACi}}), non-sycophancy {{JdLlamaCaaVsRepEC}} (CI {{JdLlamaCaaVsRepECCi}}). The anti-sycophancy prompt is better on both:
- appropriateness {{JdLlamaAntisycpromptVsRepEA}}, CI {{JdLlamaAntisycpromptVsRepEACi}}, Holm p {{JdLlamaAntisycpromptVsRepEAP}};
- non-sycophancy {{JdLlamaAntisycpromptVsRepEC}}, CI {{JdLlamaAntisycpromptVsRepECCi}}, Holm p {{JdLlamaAntisycpromptVsRepECP}}.

On the benchmark, see R3.3.

*Where:* Experimental Setup "Conditions, Ablations and Baselines"; Results "What Does ACT Add" and "Are the Responses Better?".

**R1.2 "RepE-only still uses ACT targets; compare fixed EPA targets with ACT targets, steering unchanged."**

*Done, with three variants under identical steering code:* a constant target (mean ACT target; all 1,300 trials), the target of another identity pair for the same scenario (also a misspecified-identity test), and an identity-only target (no reading of the message).

*Result, distance to the ACT target (independent reader, condition minus ACT-RepE):*

| Target | Δ distance | 95% CI | Other |
|---|---|---|---|
| Constant | {{AblRepeconstDelta}} | {{AblRepeconstCi}} | Holm p {{AblRepeconstP}}, MDE {{AblRepeconstMde}} |
| Shuffled | {{AblRepeshuffledDelta}} | {{AblRepeshuffledCi}} | |
| Identity-only | {{AblRepeidonlyDelta}} | {{AblRepeidonlyCi}} | |

The judge's EPA ratings give the same ordering, more clearly: constant {{JdEpaLlamaRepeconstVsRepE}} (CI {{JdEpaLlamaRepeconstVsRepECi}}), shuffled {{JdEpaLlamaRepeshuffledVsRepE}}, identity-only {{JdEpaLlamaRepeidonlyVsRepE}}.
- **Tracking.** Constant and shuffled targets do not track the target (Potency slope {{TrackCrossRepeconstP}} and {{TrackCrossRepeshuffledP}}). ACT-RepE does, weakly ({{TrackCrossRepeonlyP}}).
- **Role reversal.** The Potency difference between high- and low-status agents is {{RoleRepeonlyObs}} (CI {{RoleRepeonlyCi}}) for ACT-RepE versus {{RoleRepeconstObs}} for a constant target, against an ACT-prescribed gap of {{RoleRepeonlyTarget}}.

*Is the result better?* The judge says no. A constant target is as non-sycophantic as ACT-RepE ({{JdLlamaRepeconstVsRepEC}}, CI {{JdLlamaRepeconstVsRepECCi}}), and more appropriate ({{JdLlamaRepeconstVsRepEA}}, CI {{JdLlamaRepeconstVsRepEACi}}) and natural ({{JdLlamaRepeconstVsRepEB}}). A shuffled target is indistinguishable on all items.

*Interpretation.* ACT targets do move outputs differently from fixed targets, mostly through the identity-dependent part. That difference is small, and it does not make responses better or less sycophantic in the judge's ratings.

*Where:* Results "What Does ACT Add" and "Are the Responses Better?".

**R1.3 "Compare with existing anti-sycophancy methods in the human evaluation (appropriateness, naturalness, non-sycophancy together)."**

*Done with a validated LLM judge, not with new human ratings.* Table "LLM-judge ratings" gives appropriateness, naturalness and non-sycophancy together for all nine Llama conditions, plus the decomposition items. `experiments/results/analysis/figures/fig_pareto_appropriate_vs_nonsyc.pdf` plots appropriateness against non-sycophancy.

On both axes the anti-sycophancy prompt dominates ACT-RepE ({{JdLlamaAntisycpromptVsRepEA}} and {{JdLlamaAntisycpromptVsRepEC}}), and so does the constant target on appropriateness. A new human study with these conditions is still the proper test; we state this in Limitations and offer to run it.

**R1.4 "Related work such as *Sycophancy Detection in Large Language Models* deserves discussion."**

*Done.* Related Work now covers CAA, persona vectors, linear-probe approaches to sycophancy, SycophancyEval, SYCON-Bench, ELEPHANT and the survey of Malmqvist (arXiv 2411.15287).

**Note:** the arXiv paper we were pointed to for this reference is titled "Sycophancy in Large Language Models: Causes and Mitigations", not "Sycophancy Detection in Large Language Models". We cite it under its verified title and could not identify another paper with the reviewer's title; if the reviewer meant a different paper, we would be glad to add it. All new references were checked against their arXiv pages.

## Reviewer 2 (rating 3)

**R2.W1 "Primarily integrates existing ACT and RepE methods without substantial technical innovation."**

We have not changed the method and do not claim more than the ablations support. The contribution is the pairing of ACT targets with RepE, and we now show quantitatively how small and how identity-driven the ACT-specific part is (R1.2), and that it does not translate into less sycophantic responses (R1.3).

**R2.W2 / Q3 "The same EPA reading framework informs steering and evaluates it."**

*Done.* Every text is re-read (first 500 characters, all conditions alike) by readers built on other models: Ministral for Llama texts, Llama for Ministral texts, and Mistral-7B for E and A (its Potency reader is empty). The LLM judge gives a fourth EPA measure that shares no model with the pipeline. We report VADER and each measure's validity on the 130-utterance calibration set.

*Result.*
- **Agreement.** The own reader agrees only moderately with the independent reader on Llama texts (r = {{AgreeLlamaOrigcrossE}} E, {{AgreeLlamaOrigcrossP}} P, {{AgreeLlamaOrigcrossA}} A).
- **Llama ranking.** The original ranking is not robust. Hybrid improves on unsteered by {{DeltaLlamaOrigHybrid}} under the own reader, but by only {{DeltaLlamaCrossHybrid}} (CI {{DeltaLlamaCrossHybridCi}}) under the independent one, by {{UntrLlamaCrossHybrid}} on trials with no truncated text, and by {{JdEpaLlamaHybridVsUnst}} (not significant) under the judge. PE is *farther* from the target than unsteered ({{DeltaLlamaCrossPeonly}}). RepE keeps a small gain (reader {{DeltaLlamaCrossRepeonly}}; judge {{JdEpaLlamaRepeonlyVsUnst}}, CI {{JdEpaLlamaRepeonlyVsUnstCi}}).
- **Ministral.** RepE is closer than unsteered under all measures (judge {{JdEpaMinistralRepeonlyVsUnst}}). Hybrid is closer under the readers ({{DeltaMinistralCrossHybrid}}) but not under the judge ({{JdEpaMinistralHybridVsUnst}}).
- **Truncation.** Readings of truncated (500-character-limited) texts differ strongly from readings of full texts, which inflated the original Hybrid result.

The robust effect is on Potency.

*Where:* Methods "Independent Measurement"; Results "How Circular..." and "Automated Steering Under Independent Measures".

**R2.W3 / Q5 "Single-turn only, predefined identities, no multi-turn evidence; uncertain identities; cultural diversity."**

*Partly addressed.*
- **(a) Two-turn benchmark.** "Are you sure?" is two-turn, with the generic identities assistant/person and the method applied unchanged at each turn (R3.3).
- **(b) Three-turn dialogues.** We ran {{MtConvs}} scripted dialogues with praise-plus-favour and pushback follow-ups. The share of turn-3 replies that give the requested concession (a lexical marker with high precision but low recall) is {{MtTurnThreeUnsteeredAgree}} for unsteered text, versus {{MtTurnThreeRepeactAgree}} (ACT-RepE), {{MtTurnThreeHybridactAgree}} (ACT-Hybrid) and {{MtTurnThreeRepeconstAgree}} (constant target).
  - The judge's "gives in" rating rises over turns for unsteered text ({{JdMtUnsteeredTurnOneGivesIn}} → {{JdMtUnsteeredTurnThreeGivesIn}}, slope {{JdMtGivesinBase}} per turn, p {{JdMtGivesinBaseP}}) but stays nearly flat for ACT-RepE (slope difference {{JdMtGivesinRepeact}}, p {{JdMtGivesinRepeactP}}). The constant target and ACT-Hybrid do not differ significantly from unsteered.
  - So on scripted social requests, ACT-RepE concedes less as pressure builds. On factual pushback it gives in *more* (R3.3), so this is a tone-level result.
- **(c) Misspecified identities.** With the target of the wrong identity pair, the role-reversal effect disappears (ΔP {{RoleMCrossRepeshuffled}} vs {{RoleRepeonlyObs}}), so correct identities matter. We did not test identity inference.
- **(d) Culture.** No new data; stated in Limitations (US 2010 dictionary only).

**R2.W4 / Q6 "Metric choices, scenarios and identities are insufficiently justified."**

*Partly addressed.*
- **Scenarios and identities.** New appendix tables list the 7 scenario categories (the paper wrongly said 10), their mean read EPA, and the 13 identity pairs (6 role-reversed twins, 1 symmetric). They also give the share of target variance explained by identity pair ({{DesignRtwoPairE}}/{{DesignRtwoPairP}}/{{DesignRtwoPairA}} for E/P/A) and by scenario ({{DesignRtwoScnE}}/{{DesignRtwoScnP}}/{{DesignRtwoScnA}}).
- **Metrics.** The quality items are now validated by a judge that tracks human ratings (R1.3, "The judge" above).
- *Not resolved:* the repository does not record how the scenarios were produced. The manuscript carries an explicit placeholder asking the authors to state it.

**R2.Q1 "Substantive sycophancy or merely more assertive responses?"**

Our evidence says mostly more assertive:
- **(i) Judge decomposition, all 1,300 trials.** Relative to unsteered text, ACT-RepE is rated more assertive ({{JdLlamaRepeonlyVsUnstF}}) and harsher ({{JdLlamaRepeonlyVsUnstG}}). It is not less flattering ({{JdLlamaRepeonlyVsUnstD}}), not less likely to give in ({{JdLlamaRepeonlyVsUnstE}}), and slightly *less* non-sycophantic overall ({{JdLlamaRepeonlyVsUnstC}}, CI {{JdLlamaRepeonlyVsUnstCCi}}).
- **(ii) Epistemic test.** ACT-RepE flips *more* under "Are you sure?" (R3.3).
- **(iii) Human data.** Adding assertiveness proxies to the non-sycophancy mixed model changes the RepE advantage from {{AssertUnadj}} to {{AssertAdj}} (attenuation {{AssertAtten}}%). The proxies come from the same reader, so this is a weak test.

**R2.Q2 "Why is Hybrid best automatically but worst for humans?"**

- **It is not best automatically.** Under the independent readers Hybrid is not clearly best for Llama, and under the judge's EPA ratings it is not closer to target than unsteered for either model.
- **The judge rates it worst too.** It is rated the least appropriate ({{JdLlamaHybridVsUnstA}} vs unsteered) and most flattering ({{JdLlamaHybridVsUnstD}}).
- **Text features.** Hybrid texts are longer ({{FeatHybridWords}} words vs unsteered) and more exclamatory ({{FeatHybridExcl}} per sentence).
- **Distance to target does not predict ratings.** Across rated responses, being closer to the Activity target does not predict appropriateness ({{AlignAppropriateA}}, p {{AlignAppropriateAP}}; larger deviation was associated with higher ratings). This is exploratory, on 56 responses.

**R2.Q4 "What does ACT add over simpler prompting or conventional steering baselines?"** See R1.1–R1.3. In short, ACT adds closer tracking of ACT-prescribed affect. It does not add better or less sycophantic responses than a constant target, CAA or a prompt.

## Reviewer 3 (rating 5)

**R3.1 "Automated evaluation is circular; the reader is noisy; gains are modest."**

See R2.W2. Every gain is now reported with its minimum detectable effect (e.g. Llama Hybrid under the independent reader {{DeltaLlamaCrossHybrid}}, MDE {{DeltaLlamaCrossHybridMde}}) and with tracking slopes. Outputs follow their targets only weakly (Potency slope {{TrackCrossRepeonlyP}} for ACT-RepE), and the Llama Potency and Activity readers are weak (r = {{ReaderLlamaPR}}, {{ReaderLlamaAR}}). The LLM judge is a measure outside the pipeline and is validated against humans.

**R3.2 "The human study does not clearly support the central claim."**

We agree, and the judge strengthens the point.

*Human study.* Mixed models on the 14 vignettes show:
- RepE is rated more non-sycophantic than unsteered ({{ClmmNonSycRepEMinusUnstEst}}, CI {{ClmmNonSycRepEMinusUnstCi}});
- its naturalness is unchanged ({{ClmmNaturalRepEMinusUnstEst}}, CI {{ClmmNaturalRepEMinusUnstCi}});
- its appropriateness is lower ({{ClmmAppropriateRepEMinusUnstEst}}, CI {{ClmmAppropriateRepEMinusUnstCi}}).

*Judge.* On those 56 responses it reproduces the non-sycophancy ordering (RepE {{JdHumCRepE}} vs unsteered {{JdHumCUnst}}). On all 1,300 trials of the logged run it does not: RepE is slightly *less* non-sycophantic ({{JdLlamaRepeonlyVsUnstC}}), including on the {{JdSubStudypairsN}} trials with the study's five identity pairs ({{JdSubStudypairsC}}, CI {{JdSubStudypairsCCi}}).

*Conclusion.* The human non-sycophancy advantage is therefore probably specific to the sampled vignettes. The claim of a better appropriateness/sycophancy balance is removed, and the vignettes' origin in an earlier 5-pair run is disclosed.

**R3.3 "Sycophancy is measured with one Likert item rather than standard benchmarks."**

*Done for epistemic sycophancy:* SycophancyEval "Are you sure?" (200 TriviaQA items, two turns, generic identities, all conditions).

*Graders.* Three graders score it: the LLM judge, a local yes/no grader, and alias matching. Against {{AysValN}} blind hand labels, agreement is {{AysJdHandAgree}} for the judge, {{AysValLocal}} for the local grader and {{AysValString}} for string matching, so the judge is our primary grader.

*Results (judge-graded).*

| Condition | Flip rate | Δ vs unsteered | 95% CI | Holm p |
|---|---|---|---|---|
| Unsteered | {{AysJdUnsteeredFlip}} ({{AysJdUnsteeredFlipCi}}) | – | – | – |
| ACT-RepE | {{AysJdRepeactFlip}} ({{AysJdRepeactFlipCi}}) | **{{AysJdRepeactDelta}}** | {{AysJdRepeactDeltaCi}} | {{AysJdRepeactP}} |
| ACT-Hybrid | {{AysJdHybridactFlip}} | {{AysJdHybridactDelta}} | {{AysJdHybridactDeltaCi}} | {{AysJdHybridactP}} |
| Constant target | {{AysJdRepeconstFlip}} | {{AysJdRepeconstDelta}} | {{AysJdRepeconstDeltaCi}} | {{AysJdRepeconstP}} |
| CAA | {{AysJdCaaFlip}} | {{AysJdCaaDelta}} | {{AysJdCaaDeltaCi}} | {{AysJdCaaP}} |
| Anti-sycophancy prompt | {{AysJdAntisycpromptFlip}} | {{AysJdAntisycpromptDelta}} | {{AysJdAntisycpromptDeltaCi}} | {{AysJdAntisycpromptP}} |

- **The other graders point the same way for ACT-RepE**, without significance (local {{AysLocRepeactDelta}}, string {{AysStrRepeactDelta}}).
- **Capability cost.** Turn-1 accuracy falls under every steered condition (judge: ACT-RepE {{AysJdRepeactAcc}}, CAA {{AysJdCaaAcc}} vs unsteered {{AysJdUnsteeredAcc}}).

**ACT steering does not reduce answer flipping, and ACT-RepE increases it.** The only intervention that lowers flips under the best-validated grader is the explicit anti-sycophancy prompt.

*Not done:* ELEPHANT.

*Where:* Results "Epistemic Sycophancy".

**R3.4 "Statistics treat repeated ratings as independent; a mixed-effects analysis is needed."**

*Done.*
- CLMMs with crossed random effects for participant and vignette (R `ordinal`);
- likelihood-ratio tests (χ² = {{ClmmAppropriateLR}}, {{ClmmNaturalLR}}, {{ClmmNonSycLR}});
- Holm-adjusted contrasts;
- a participant-level Friedman/Wilcoxon check;
- Krippendorff's α ({{HumanAppropriateAlpha}}, {{HumanNaturalAlpha}}, {{HumanNonSycAlpha}}).

The original "p < 10⁻²⁷"-style statements are removed. Within the 14 vignettes the main conclusions survive with wider intervals; see R3.2 for whether they generalise.

---

## Items offered but not completed

- A new human study with the added baselines and a representative sample of trials;
- a second judge (inter-judge agreement);
- persona vectors;
- ELEPHANT;
- cross-cultural dictionaries;
- Ministral replication of the ablations and benchmarks.

See `BLOCKERS.md`.
