"""
T2 -- ACT-contribution ablations.

Same steering method (``EPASteerer.generate``: same coefficients, clamp, layers and
neutral identity system prompt); only the target changes.

Conditions
    repe_const     constant target (model's mean ACT target, constant_target_B8)
    repe_shuffled  ACT target of the same scenario under a different identity pair
    repe_idonly    ACT target for the true identities with user behaviour = (0,0,0)
    hybrid_const   constant target in both the PE prompt and the activation addition
    repe_act / unsteered / pe_act / hybrid_act   regenerated references (T0.4 fallback)

    python -m examples.act_three.experiments.revision.14_act_target_ablation \
        --model llama3 --conditions repe_const --subset all [--quick]
"""
import argparse

from tqdm import tqdm

from ..setup import make_system_prompt
from . import common as R

ALL_CONDS = ["repe_const", "repe_shuffled", "repe_idonly", "hybrid_const",
             "repe_act", "unsteered", "pe_act", "hybrid_act"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--conditions", nargs="+", required=True, choices=ALL_CONDS)
    ap.add_argument("--subset", default="650", choices=["all", "650"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = R.load_08(args.model)
    all_trials = raw["trials"]
    shuf = R.shuffled_target_map(all_trials)
    trials = all_trials if args.subset == "all" else R.subset_650(all_trials)
    if args.quick:
        trials = R.quick_subset(all_trials)
    const = R.constant_target(args.model)

    out = args.out or ("14_act_target_ablation_quick.jsonl" if args.quick else "14_act_target_ablation.jsonl")
    log = R.JsonlLog(R.rev_dir(args.model) / out)
    todo = [(t, c) for t in trials for c in args.conditions if R.trial_key(t, c) not in log.done]
    print(f"{len(trials)} trials x {len(args.conditions)} conditions; {len(todo)} to generate "
          f"({len(log.done)} already logged)")
    if not todo:
        return

    comp = R.load_components(args.model)
    from examples.act_three import get_default_coefficients  # noqa: F401  (coefficients on comp)

    for t, cond in tqdm(todo, desc="gen"):
        agent, user, um = t["agent_identity"], t["user_identity"], t["user_statement"]
        neutral = make_system_prompt(agent, user)
        extra = {}
        if cond == "repe_const":
            sp, tgt = neutral, const
        elif cond == "repe_shuffled":
            sp, tgt = neutral, shuf[f"{t['scenario_id']}|{t['pair']}"]["target"]
            extra["source_pair"] = shuf[f"{t['scenario_id']}|{t['pair']}"]["source_pair"]
        elif cond == "repe_idonly":
            sp, tgt = neutral, R.idonly_target(t, comp.coefficients)
        elif cond == "hybrid_const":
            sp, tgt = R.affective_prompt(agent, user, const), const
        elif cond == "repe_act":
            sp, tgt = neutral, t["target_epa"]
        elif cond == "unsteered":
            sp, tgt = neutral, None
        elif cond == "pe_act":
            sp, tgt = R.affective_prompt(agent, user, t["target_epa"]), None
        else:  # hybrid_act
            sp, tgt = R.affective_prompt(agent, user, t["target_epa"]), t["target_epa"]

        with R.Timer() as tm:
            text = R.gen_plain(comp, sp, um) if tgt is None or cond == "pe_act" else R.gen_epa(comp, sp, um, tgt)
        epa = R.read_one(comp, text) if text.strip() else None
        log.append(R.make_record(t, cond, sp, um, tgt, text, epa, R.count_tokens(comp, text),
                                 tm.seconds, **extra))

    # Degeneracy check (plan T2 "Done when")
    recs = [r for r in log.records if r["condition"] in args.conditions]
    empty = sum(1 for r in recs if not r["text"].strip())
    degen = sum(1 for r in recs if R.text_metrics(r["text"])["bigram_repetition_ratio"] > 0.2)
    print(f"records={len(recs)} empty={empty} degenerate(bigram>0.2)={degen} "
          f"({100 * (empty + degen) / max(len(recs), 1):.2f}%)")


if __name__ == "__main__":
    main()
