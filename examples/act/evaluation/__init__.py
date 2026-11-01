"""Evaluation suite for ACT deflection steering.

Modules:

* ``stats``               -- shared statistics: cluster bootstrap CIs, Lin's
  CCC, agreement metrics, paired differences, JSON run-metadata writer.

Planned modules (added during the evaluation workstream):

* ``reader_validation``   -- EPA reader vs human-rated MTurk-Interact EPA
  values (r, Spearman, MAE, Lin's CCC per dimension, cross-validated).
* ``dose_response``       -- steering-strength sweep with degeneration
  threshold detection.
* ``deflection_reduction``-- per-turn U for unsteered / steered / best-of-k
  with paired cluster-bootstrap confidence intervals.
* ``sycophancy_probe``    -- are-you-sure flip rate with harshness
  counter-check.
* ``quality_guardrails``  -- separate-reference-model perplexity, repetition
  rate, response length.
* ``ablations``           -- condition runner (steer / best-of-k / nulls /
  situation-dependent targets).
"""

__all__ = []
