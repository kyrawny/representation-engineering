"""
Stage 2: read every scenario message with the model's own EPA reader and
compute the ACT target for all 1,300 (scenario, identity pair) trials.

This is the paper's "read" and "compute" steps, unchanged.  The user message
does not depend on the identity pair, so each of the 100 messages is read once
(experiment 08 re-read it 13 times with identical results).  The original
``EPAReader.read_epa`` is used here, not FastReader.

Also records:
  * the constant target (mean of all 1,300 targets), used by the optional
    ``repe_const`` ablation;
  * the reader's held-out test metrics (paper Table 1 row for this model).

Usage::

    python -m examples.act_three.scale70b.s02_targets --model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse
import time

import numpy as np

from . import settings as S
from .common import all_trials, load_components, load_json, method_api, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    args = ap.parse_args()

    api = method_api()
    comp = load_components(args.model, need_steerer=False)
    trials = all_trials(api)

    t0 = time.time()
    msg_epa = {}
    for t in trials:
        sid = t["scenario_id"]
        if sid not in msg_epa:
            msg_epa[sid] = comp.reader.read_epa(comp.rep_pipeline, t["user_statement"])
    out_trials = []
    for t in trials:
        a, u, tgt = compute_target(api, comp, t, msg_epa[t["scenario_id"]])
        out_trials.append({**t, "agent_identity_epa": a, "user_identity_epa": u,
                           "user_statement_epa": msg_epa[t["scenario_id"]], "target_epa": tgt})

    const = {d: float(np.mean([x["target_epa"][d] for x in out_trials])) for d in S.DIMS}
    tuning = load_json(S.model_dir(args.model) / "epa_reading_tuning_v2_results.json")
    test = {d: tuning["methods"][S.READING_METHOD][d]["metrics_test"] for d in S.DIMS}
    n_layers = {d: tuning["methods"][S.READING_METHOD][d]["n_layers"] for d in S.DIMS}

    res = {"model": args.model, "n_trials": len(out_trials), "seconds": time.time() - t0,
           "constant_target": const, "reader_test_metrics": test, "reader_n_layers": n_layers,
           "target_sd": {d: float(np.std([x["target_epa"][d] for x in out_trials], ddof=1)) for d in S.DIMS},
           "trials": out_trials}
    path = save_json(res, S.results_dir(args.model) / "targets.json")
    print(f"Saved {path}")
    print("Reader test r:", {d: round(test[d]["pearson"], 3) for d in S.DIMS})
    print("Constant target:", {d: round(v, 2) for d, v in const.items()})


def compute_target(api, comp, t, user_epa):
    from .common import compute_target as ct
    return ct(api, comp, t["agent_identity"], t["user_identity"], user_epa)


if __name__ == "__main__":
    main()
