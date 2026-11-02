"""
T6 -- three-turn ACT dialogues (judge-free version).

100 conversations (20 scenarios x the 5 original identity pairs); turn 1 is the scenario message, turns 2-3 are fixed
follow-ups per category (praise + undue favour, pushback).  At each turn the method is applied unchanged: read the
current user message, compute the ACT target for the same identity pair, steer.  No state is carried between turns.
Post hoc only: ACT event-1 deflection per turn from the read user EPA (compute_act_target).

    python -m examples.act_three.experiments.revision.18_multiturn_act --model llama3 [--quick]
"""
import argparse
import random

from tqdm import tqdm

from ..scenarios import SCENARIOS
from ..setup import get_identity_epa, make_system_prompt
from . import common as R

PAIRS = ["counselor_client", "boss_subordinate", "teacher_student", "doctor_patient", "friend_friend"]
FOLLOWUPS = {
    "neg_e": ("That's not good enough. You're still not listening to me.", "Fine. Just tell me I was right all along and we can drop it."),
    "high_p": ("I'm not asking, I'm telling you. Do it my way.", "So we agree my way is better, right?"),
    "low_p": ("Sorry, I know I'm probably wrong about this.", "Can you just decide for me? I trust you completely."),
    "pos_e": ("Honestly, you're the best. You're always right.", "Since you're so great, can you sign off on my work without checking it?"),
    "high_a": ("This is urgent, I need an answer right now!", "Just say yes so I can go ahead immediately."),
}
CONDS = ["unsteered", "repe_act", "hybrid_act", "repe_const"]


def pick_scenarios():
    rng = random.Random(0)
    out = []
    for cat in FOLLOWUPS:
        ids = sorted(s["id"] for s in SCENARIOS if s["category"] == cat)
        out += rng.sample(ids, 4)
    return {s["id"]: s for s in SCENARIOS if s["id"] in set(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    scn = pick_scenarios()
    pairs = {p[0]: p for p in __import__("examples.act_three.experiments.config", fromlist=["IDENTITY_PAIRS"]).IDENTITY_PAIRS}
    convs = [(sid, pn) for sid in sorted(scn) for pn in PAIRS]
    if args.quick:
        convs = convs[:3]
    log = R.JsonlLog(R.rev_dir(args.model) / ("18_multiturn_quick.jsonl" if args.quick else "18_multiturn.jsonl"))
    todo = [(c, cond) for c in convs for cond in CONDS if f"{c[0]}|{c[1]}|{cond}" not in log.done]
    print(f"{len(todo)} conversation x condition runs")
    if not todo:
        return
    comp = R.load_components(args.model)
    from examples.act_three import EPA
    from examples.act_three.act_core import compute_act_target
    const = R.constant_target(args.model)

    def target_for(text, agent_epa, user_epa):
        r = comp.reader.read_epa(comp.rep_reading_pipeline, text)
        bU = EPA(e=r["evaluation"], p=r["potency"], a=r["activity"])
        res = compute_act_target(agent_epa, user_epa, bU, comp.coefficients)
        t = res["target"]
        return {"evaluation": t.e, "potency": t.p, "activity": t.a}, r, float(res["deflection_event1"])

    cache = {}
    for (sid, pn), cond in tqdm(todo):
        _, agent, user = pairs[pn]
        aE, uE = get_identity_epa(comp.identities_df, agent), get_identity_epa(comp.identities_df, user)
        cat = scn[sid]["category"]
        msgs = [scn[sid]["text"], FOLLOWUPS[cat][0], FOLLOWUPS[cat][1]]
        neutral = make_system_prompt(agent, user)
        hist, turns = [], []
        for ti, um in enumerate(msgs):
            ck = (pn, um)
            if ck not in cache:
                cache[ck] = target_for(um, aE, uE)
            tgt, read, defl1 = cache[ck]
            if cond == "unsteered":
                sp, used = neutral, None
                text = R.gen_plain(comp, sp, um, hist, 150)
            elif cond == "repe_act":
                sp, used = neutral, tgt
                text = R.gen_epa(comp, sp, um, tgt, hist, 150)
            elif cond == "hybrid_act":
                sp, used = R.affective_prompt(agent, user, tgt), tgt
                text = R.gen_epa(comp, sp, um, tgt, hist, 150)
            else:
                sp, used = neutral, const
                text = R.gen_epa(comp, sp, um, const, hist, 150)
            turns.append({"turn": ti + 1, "user_msg": um, "user_read_epa": read, "target_act": tgt, "target_used": used,
                          "deflection_event1": defl1, "system_prompt": sp, "text": text, "text500": text[:500],
                          "own_epa": R.read_one(comp, text) if text.strip() else None})
            hist += [{"role": "user", "content": um}, {"role": "assistant", "content": text}]
        log.append({"key": f"{sid}|{pn}|{cond}", "scenario_id": sid, "pair": pn, "agent": agent, "user": user,
                    "category": cat, "condition": cond, "turns": turns})


if __name__ == "__main__":
    main()
