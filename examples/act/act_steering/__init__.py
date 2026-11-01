"""ACT-faithful deflection steering on top of the RepE toolkit.

This package steers conversational LLM generation so that responses minimize
*deflection* as defined by Affect Control Theory (ACT): the squared distance
between post-event impressions and the sentiments (EPA fundamentals) in play.

Architecture (three composable intervention modes plus a multi-turn loop):

* ``monitor``  -- reads a per-token deflection trace from a PCA reader fitted
  on low- vs high-deflection contrast continuations.
* ``steerer``  -- actuates generation with a deflection-contrast direction
  (``c = alpha * (h_low - h_high)``) whose per-turn gain is proportional to
  the current deflection.
* ``selector`` -- ACT best-of-k: generate k candidates, score the actual text
  by post-event deflection, select the argmin.
* ``pipeline`` -- the per-turn control loop tying state, steering, selection,
  monitoring, and re-identification repair together.

The pure-logic modules (``config``, ``dictionary``, ``act_core``, ``stimuli``,
``state``, ``monitor``, ``selector``) are torch-free and CPU-testable; the
model-bound modules (``directions``, ``steerer``, ``pipeline``) require a GPU.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
