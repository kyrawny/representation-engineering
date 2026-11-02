"""
Write the short human-readable notes files required by the plan from the JSON outputs
(so the numbers in the notes cannot drift from the analyses).

    python -m examples.act_three.experiments.analysis.write_notes
"""
import json

from .common import OUT_DIR, load_json

ITEMS = {"appropriate": "appropriateness", "natural": "naturalness", "non_sycophantic": "non-sycophancy"}


def human_notes() -> str:
    clmm = load_json(OUT_DIR / "human_clmm.json")
    mixed = load_json(OUT_DIR / "human_mixed.json")
    L = ["# Human-study reanalysis (T7)", "",
         "Data: `human_long.csv` (48 participants x 14 vignettes x 4 conditions x 3 items). Vignettes come from the July 500-trial, "
         "5-pair run, not the 1,300-trial run in Table 3. Models: CLMM `rating ~ cond + (1|pid) + (1|vig)` (and a random participant "
         "condition-slope variant; the AIC rule picks which one the contrasts come from), LMM with crossed random effects, "
         "participant-level Friedman/Wilcoxon, Krippendorff alpha (ordinal; units = vignette x condition, raters = participants).", ""]
    for it, lab in ITEMS.items():
        r, m, pl = clmm[it], mixed["lmm"][it], mixed["participant_level"][it]
        L.append(f"## {lab}")
        L.append(f"- LR test of condition: chi2 = {r['LR_chisq']:.1f}, p = {r['LR_p']:.2g}; contrasts from: {r['contrast_model']}.")
        for c, ci in zip(r["contrasts_holm"], r["contrasts_ci"]):
            L.append(f"  - {c['contrast']}: log-odds {c['estimate']:+.2f} [{ci['asymp.LCL']:+.2f}, {ci['asymp.UCL']:+.2f}], Holm p = {c['p.value']:.2g}")
        L.append(f"- Participant-level Friedman p = {pl['friedman_p']:.2g}; Krippendorff alpha = {mixed['reliability'][it]['krippendorff_alpha_ordinal']:.2f}; "
                 f"ICC participant = {m['icc_pid']:.2f}, vignette = {m['icc_vig']:.2f}.")
        L.append(f"- Original pooled Kruskal-Wallis: H = {mixed['original_pooled'][it]['H']:.1f}, p = {mixed['original_pooled'][it]['p']:.2g}.")
        L.append("")
    L += ["## Claims vs the original paper", "",
          "- Condition effects remain significant for all three items; the pooled p-values of order 1e-15 to 1e-27 were artefacts of treating 672 ratings per condition as independent, and are replaced by CLMM estimates.",
          "- RepE > unsteered on non-sycophancy and Hybrid worst on naturalness/appropriateness are retained (see contrasts above); 'unsteered most appropriate' is retained against RepE and PE only to the extent the CLMM contrasts say so.",
          "- Rater agreement is low (alpha above), so single ratings are noisy and vignette-level conclusions are fragile with 14 vignettes."]
    return "\n".join(L) + "\n"


def gap_notes() -> str:
    h = load_json(OUT_DIR / "hybrid_human_gap.json")
    tf = h["text_features"]["llama3"]
    m, pv = tf["means"], tf["paired_vs_unsteered"]
    ov = h["oversteer"]["llama3"].get("cross", {})
    al = h["human"]["alignment_models"]
    asr = h["human"]["assertiveness_models"]
    L = ["# Why does Hybrid disagree with humans? (T10, judge-free)", "",
         f"1. Hybrid responses are longer than unsteered ones ({m['hybrid']['n_words']:.0f} vs {m['unsteered']['n_words']:.0f} words on average; paired difference "
         f"{pv['hybrid']['n_words']['delta']:+.1f}, CI [{pv['hybrid']['n_words']['lo']:+.1f}, {pv['hybrid']['n_words']['hi']:+.1f}]) and use more exclamation marks "
         f"({pv['hybrid']['exclaim_per_sent']['delta']:+.2f} per sentence), consistent with the PE wording 'energetic and animated' acting together with the activation shift.",
         f"2. Over-steering: on the independent reader Hybrid moves Potency from below the target to slightly above it (mean signed deviation {ov['hybrid']['P']['mean_signed_dev']:+.2f}, unsteered {ov['unsteered']['P']['mean_signed_dev']:+.2f}) "
         f"but is more active than targeted ({ov['hybrid']['A']['mean_signed_dev']:+.2f} on Activity, unsteered {ov['unsteered']['A']['mean_signed_dev']:+.2f}); over-steering is therefore slight on Potency and clearer only on Activity.",
         f"3. Alignment vs ratings (56 responses, exploratory): within conditions, |Activity deviation| is associated with appropriateness at {al['appropriate']['within_condition']['abs_dev_A']['est']:+.2f} "
         f"(p = {al['appropriate']['within_condition']['abs_dev_A']['p']:.3g}), i.e. in the opposite direction to the hypothesis that closer-to-target is more appropriate.",
         f"4. Assertiveness: adding reader-based assertiveness/harshness proxies and lexical features to the non-sycophancy model changes the RepE-minus-unsteered effect from "
         f"{asr['repe_only_minus_unsteered_unadjusted']['est']:.2f} to {asr['repe_only_minus_unsteered_adjusted']['est']:.2f} ({asr['attenuation_pct']:.0f}% attenuation). Caveat: the proxies come from the same reader.",
         "5. No LLM judge was available, so naturalness and the (d)-(h) rubric items were not scored; the explanation above rests on text features and the reader only."]
    return "\n".join(L) + "\n"


def run():
    (OUT_DIR / "human_reanalysis_notes.md").write_text(human_notes(), encoding="utf-8")
    if (OUT_DIR / "hybrid_human_gap.json").exists():
        (OUT_DIR / "hybrid_human_gap_notes.md").write_text(gap_notes(), encoding="utf-8")


if __name__ == "__main__":
    run()
    print("notes written")
