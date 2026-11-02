"""
scale70b: test whether ACT x RepE steering still works at 70B parameters.

See README.md in this folder.  Stages:
    s00_preflight             environment, multi-GPU steering, reader equivalence
    run_cli (stage 1)         the paper's direction extraction and reader tuning, unchanged
    s02_targets               read user messages, compute ACT targets (1,300 trials)
    s03_norm_probe            relative size of the steering perturbation
    s04_strength_calibration  pre-specified rule for the steering strength
    s05_main_steering         unsteered / RepE / PE / hybrid (+ optional ablations), sharded
    s06_coherence             perplexity and text metrics
    s07_cross_read            score outputs with a reader built on another model
    s08_merge_analyze         statistics, verdict, REPORT.md (CPU only)
"""
