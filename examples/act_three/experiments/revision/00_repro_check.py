"""
T0.4 / Q0: reproducibility check.

Regenerates ``unsteered`` and ``repe_only`` for the first N trials of the logged
experiment-08 run with the logged target and compares with the logged text.

    python -m examples.act_three.experiments.revision.00_repro_check --model llama3 --n 5
"""
import argparse
import json

from ..setup import make_system_prompt
from . import common as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ortho", choices=["config", "on", "off"], default="config")
    args = ap.parse_args()

    raw = R.load_08(args.model)
    trials = raw["trials"][: args.n]
    comp = R.load_components_ortho(args.model, {"config": None, "on": True, "off": False}[args.ortho])
    out = []
    for t in trials:
        sp = make_system_prompt(t["agent_identity"], t["user_identity"])
        um = t["user_statement"]
        un = R.gen_plain(comp, sp, um)
        rp = R.gen_epa(comp, sp, um, t["target_epa"])
        for cond, new in (("unsteered", un), ("repe_only", rp)):
            logged = t["conditions"][cond]["text"]
            out.append({"scenario": t["scenario_id"], "pair": t["pair"], "cond": cond,
                        "identical": new[:500] == logged, "new": new[:500], "logged": logged})
            print(cond, t["scenario_id"], t["pair"], "IDENTICAL" if new[:500] == logged else "DIFFERENT")
    n_same = sum(o["identical"] for o in out)
    res = {"ortho_flag": args.ortho, "model": args.model, "n_trials": len(trials), "n_texts": len(out), "n_identical": n_same,
           "all_identical": n_same == len(out), "records": out}
    path = R.rev_dir(args.model) / ("00_repro_check.json" if args.ortho == "config" else f"00_repro_check_ortho_{args.ortho}.json")
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{n_same}/{len(out)} identical -> {path}")


if __name__ == "__main__":
    main()
