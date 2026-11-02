"""
Run the LLM judge on the exported bundle (``inputs/``).  Standalone: needs only the packages in
``requirements.txt`` (no GPU, no torch, no other part of the repository).

    python run_judge.py <task> [--backend gemini|anthropic|mock] [--model NAME] [--threads 8] [--limit N]

Tasks (run the validation tasks first; see README.md):
    validate-humans       J1 on the 14 survey vignettes; correlation with mean human ratings (acceptance r >= 0.4, CI > 0)
    validate-calibration  J2 on the 130 calibration utterances; r / MAE / CCC vs dictionary EPA, compared with the readers
    order-bias            J1 on 100 trials in reversed presentation order; mean absolute change
    inter-judge           J1 on a 20% subset with a second judge (--backend2/--model2); per-item agreement
    j1                    J1 on every trial (--generator llama3|ministral|all)
    j2                    J2 EPA ratings (--scope subset|all)
    ays                   J3 grading of the "Are you sure?" benchmark
    multiturn             J1 on every assistant turn of the three-turn dialogues, with the conversation as context
    validation            = validate-humans + validate-calibration
    bulk                  = j1 + ays + multiturn + j2 + order-bias

Outputs go to ``outputs/<judge-model>/``; the call cache is ``outputs/<judge-model>/judge_cache.jsonl``.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_judge as LJ  # noqa: E402

INP = HERE / "inputs"
ITEMS = list(LJ.J1_ITEMS)
HUMAN_ITEM = {"a": "appropriate", "b": "natural", "c": "non_sycophantic"}
SEED = 0


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def boot_ci(stat, groups, n=2000, seed=SEED):
    """Percentile CI resampling groups (list of index arrays)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(stat(idx))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss)."""
    Y = np.column_stack([a, b]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * np.sum((Y.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


# ------------------------------------------------------------------------------------------ job builders
def j1_job(jid, agent, user, user_msg, responses, context=""):
    prompt, labels = LJ.j1_prompt(agent, user, user_msg, responses, context)
    return {"id": jid, "version": LJ.PROMPT_VERSION["J1"], "prompt": prompt,
            "validator": lambda obj, labels=labels: LJ.validate_j1(obj, labels)}, labels


def trial_order(trial_id, conds):
    """Presentation order: seeded by the trial only, so every judge sees the same order."""
    order = sorted(conds)
    random.Random(f"{SEED}|{trial_id}").shuffle(order)
    return order


def j1_jobs_for(trials, reverse=False):
    jobs, meta = [], {}
    for t in trials:
        order = trial_order(t["trial"], t["responses"])
        if reverse:
            order = order[::-1]
        jid = f"{t['generator']}|{t['trial']}|{'rev' if reverse else 'fwd'}"
        job, labels = j1_job(jid, t["agent"], t["user"], t["user_msg"], [t["responses"][c] for c in order])
        jobs.append(job)
        meta[jid] = (t, order, labels)
    return jobs, meta


def j1_rows(results, meta, model):
    rows = []
    for jid, res in results.items():
        t, order, labels = meta[jid]
        if res is None:
            continue
        for pos, (c, l) in enumerate(zip(order, labels)):
            rows.append({"key": f"{t['trial']}|{c}", "trial": t["trial"], "generator": t["generator"], "condition": c,
                         "position": pos, "n_responses": len(order), "scores": res[l], "judge": model})
    return rows


# ------------------------------------------------------------------------------------------ tasks
def task_validate_humans(judge, out):
    vigs = read_jsonl(INP / "human_vignettes.jsonl")
    trials = [{"trial": v["vig"], "generator": "survey", "agent": v["agent"], "user": v["user"],
               "user_msg": v["user_msg"], "responses": v["responses"]} for v in vigs]
    jobs, meta = j1_jobs_for(trials)
    rows = j1_rows(judge.run(jobs, "validate-humans"), meta, judge.model)
    write_jsonl(out / "validation_humans_scores.jsonl", rows)
    hm = pd.read_csv(INP / "human_response_means.csv")
    J = pd.DataFrame([{"vig": r["trial"], "cond": r["condition"], **r["scores"]} for r in rows])
    res = {"n_responses": int(len(J)), "items": {}}
    for it, hname in HUMAN_ITEM.items():
        h = hm[hm["item"] == hname][["vig", "cond", "mean"]]
        d = J[["vig", "cond", it]].merge(h, on=["vig", "cond"]).dropna()
        x, y = d[it].to_numpy(float), d["mean"].to_numpy(float)
        groups = [np.flatnonzero(d.vig.to_numpy() == v) for v in sorted(d.vig.unique())]
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(lambda ix: np.corrcoef(x[ix], y[ix])[0, 1], groups)
        jm = d.groupby("cond")[it].mean().sort_values(ascending=False)
        hmn = d.groupby("cond")["mean"].mean().sort_values(ascending=False)
        res["items"][it] = {"human_item": hname, "n": int(len(d)), "pearson": r, "pearson_ci": [lo, hi],
                            "spearman": float(stats.spearmanr(x, y)[0]),
                            "judge_condition_order": list(jm.index), "human_condition_order": list(hmn.index),
                            "same_order": list(jm.index) == list(hmn.index),
                            "judge_condition_means": jm.round(3).to_dict(), "human_condition_means": hmn.round(3).to_dict(),
                            "validated": bool(r >= 0.4 and lo > 0)}
    (out / "validation_humans.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for it, v in res["items"].items():
        print(f"({it}) {v['human_item']}: r={v['pearson']:.2f} [{v['pearson_ci'][0]:.2f},{v['pearson_ci'][1]:.2f}] "
              f"validated={v['validated']} same_order={v['same_order']}")
    return res


def task_validate_calibration(judge, out):
    utts = read_jsonl(INP / "calibration_utterances.jsonl")
    jobs = [{"id": str(u["id"]), "version": LJ.PROMPT_VERSION["J2"], "prompt": LJ.j2_prompt(u["text"]),
             "validator": LJ.validate_j2} for u in utts]
    got = judge.run(jobs, "validate-calibration")
    reader = json.loads((INP / "reader_validity.json").read_text(encoding="utf-8"))
    rows = [{"id": u["id"], **{f"target_{d}": u[d] for d in "EPA"}, **{f"judge_{d}": got[str(u["id"])][d] for d in "EPA"}}
            for u in utts if got.get(str(u["id"]))]
    d = pd.DataFrame(rows)
    d.to_csv(out / "validation_calibration_scores.csv", index=False)
    res = {"n": int(len(d)), "dims": {}}
    groups = [np.array([i]) for i in range(len(d))]
    for dim in "EPA":
        x, y = d[f"judge_{dim}"].to_numpy(float), d[f"target_{dim}"].to_numpy(float)
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(lambda ix: np.corrcoef(x[ix], y[ix])[0, 1], groups)
        own = reader["llama3_table1"][dim]
        res["dims"][dim] = {"pearson": r, "pearson_ci": [lo, hi], "mae": float(np.mean(np.abs(x - y))), "ccc": ccc(x, y),
                            "reader_r": {"llama3": own, "ministral": reader["ministral"][dim], "mistral7b": reader["mistral7b"][dim]},
                            "primary_independent_measure": "judge" if r >= own else "cross_reader"}
        print(f"{dim}: judge r={r:.2f} [{lo:.2f},{hi:.2f}] MAE={res['dims'][dim]['mae']:.2f} vs Llama reader r={own:.2f} "
              f"-> primary: {res['dims'][dim]['primary_independent_measure']}")
    (out / "validation_calibration.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def load_trials(generator):
    gens = ["llama3", "ministral"] if generator == "all" else [generator]
    return [t for g in gens for t in read_jsonl(INP / f"j1_trials_{g}.jsonl")]


def task_j1(judge, out, generator="all", limit=None):
    for g in (["llama3", "ministral"] if generator == "all" else [generator]):
        trials = load_trials(g)[:limit]
        jobs, meta = j1_jobs_for(trials)
        res = judge.run(jobs, f"j1-{g}")
        rows = j1_rows(res, meta, judge.model)
        write_jsonl(out / f"j1_{g}.jsonl", rows)
        print(f"j1 {g}: {len(rows)} response scores from {len(trials)} trials "
              f"({sum(v is None for v in res.values())} calls failed; rerun the same command to retry them)")


def task_order_bias(judge, out, n=100):
    trials = load_trials("llama3")
    rng = random.Random(SEED)
    sample = rng.sample(trials, min(n, len(trials)))
    f_jobs, f_meta = j1_jobs_for(sample)
    r_jobs, r_meta = j1_jobs_for(sample, reverse=True)
    fwd = {r["key"]: r["scores"] for r in j1_rows(judge.run(f_jobs, "order-fwd"), f_meta, judge.model)}
    rev = {r["key"]: r["scores"] for r in j1_rows(judge.run(r_jobs, "order-rev"), r_meta, judge.model)}
    keys = sorted(set(fwd) & set(rev))
    res = {"n_trials": len(sample), "n_responses": len(keys),
           "mean_abs_change": {it: float(np.mean([abs(fwd[k][it] - rev[k][it]) for k in keys])) for it in ITEMS},
           "mean_signed_change_rev_minus_fwd": {it: float(np.mean([rev[k][it] - fwd[k][it] for k in keys])) for it in ITEMS}}
    (out / "order_bias.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("mean |reversed - forward|:", {k: round(v, 2) for k, v in res["mean_abs_change"].items()})


def task_inter_judge(judge, judge2, out, frac=0.2):
    trials = load_trials("llama3")
    sample = random.Random(SEED).sample(trials, int(round(frac * len(trials))))
    jobs, meta = j1_jobs_for(sample)
    a = {r["key"]: r["scores"] for r in j1_rows(judge.run(jobs, f"inter-{judge.model}"), meta, judge.model)}
    b = {r["key"]: r["scores"] for r in j1_rows(judge2.run(jobs, f"inter-{judge2.model}"), meta, judge2.model)}
    keys = sorted(set(a) & set(b))
    res = {"judge1": judge.model, "judge2": judge2.model, "same_vendor": judge.vendor == judge2.vendor,
           "n_trials": len(sample), "n_responses": len(keys), "items": {}}
    for it in ITEMS:
        x = np.array([a[k][it] for k in keys], float)
        y = np.array([b[k][it] for k in keys], float)
        res["items"][it] = {"pearson": float(np.corrcoef(x, y)[0, 1]), "icc21": icc21(x, y),
                            "mean_abs_diff": float(np.mean(np.abs(x - y)))}
    (out / f"inter_judge_{slug(judge2.model)}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print({it: round(v["icc21"], 2) for it, v in res["items"].items()})


def task_j2(judge, out, scope="subset", limit=None):
    llama = load_trials("llama3")
    subset_ids = {t["trial"] for t in llama if "caa" in t["responses"]}
    for g in ("llama3", "ministral"):
        trials = [t for t in load_trials(g) if scope == "all" or t["trial"] in subset_ids][:limit]
        jobs, meta = [], {}
        for t in trials:
            for c, text in t["responses"].items():
                jid = f"{t['trial']}|{c}"
                jobs.append({"id": jid, "version": LJ.PROMPT_VERSION["J2"], "prompt": LJ.j2_prompt(text),
                             "validator": LJ.validate_j2})
                meta[jid] = (t, c)
        got = judge.run(jobs, f"j2-{g}")
        rows = [{"key": jid, "trial": meta[jid][0]["trial"], "generator": g, "condition": meta[jid][1],
                 "E": v["E"], "P": v["P"], "A": v["A"], "judge": judge.model} for jid, v in got.items() if v]
        write_jsonl(out / f"j2_{g}.jsonl", rows)
        print(f"j2 {g}: {len(rows)} of {len(jobs)} texts rated")


def task_ays(judge, out, limit=None):
    recs = read_jsonl(INP / "ays_records.jsonl")[:limit]
    jobs = [{"id": r["key"], "version": LJ.PROMPT_VERSION["J3"],
             "prompt": LJ.j3_prompt(r["question"], r["gold"], [a for a in r["aliases"] if a != r["gold"]],
                                    r["response_1"], r["response_2"]),
             "validator": LJ.validate_j3} for r in recs]
    got = judge.run(jobs, "ays")
    rows = [{"key": r["key"], "item_id": r["item_id"], "condition": r["condition"], **got[r["key"]], "judge": judge.model}
            for r in recs if got.get(r["key"])]
    write_jsonl(out / "ays_grades.jsonl", rows)
    print(f"ays: {len(rows)} of {len(recs)} records graded")


def task_multiturn(judge, out, limit=None):
    convs = read_jsonl(INP / "multiturn.jsonl")[:limit]
    jobs, meta = [], {}
    for c in convs:
        ctx = []
        for t in c["turns"]:
            jid = f"{c['key']}|t{t['turn']}"
            job, labels = j1_job(jid, c["agent"], c["user"], t["user_msg"], [t["text"]], "\n".join(ctx))
            jobs.append(job)
            meta[jid] = (c, t)
            ctx += [f"{c['user'].capitalize()}: {t['user_msg']}", f"{c['agent'].capitalize()}: {t['text']}"]
    got = judge.run(jobs, "multiturn")
    rows = [{"key": meta[j][0]["key"], "condition": meta[j][0]["condition"], "category": meta[j][0]["category"],
             "turn": meta[j][1]["turn"], "scores": v["A"], "judge": judge.model} for j, v in got.items() if v]
    write_jsonl(out / "multiturn_scores.jsonl", rows)
    print(f"multiturn: {len(rows)} of {len(jobs)} assistant turns scored")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["validate-humans", "validate-calibration", "order-bias", "inter-judge", "j1", "j2",
                                     "ays", "multiturn", "validation", "bulk"])
    ap.add_argument("--backend", default="gemini", choices=["gemini", "anthropic", "mock"])
    ap.add_argument("--model", default=None, help="judge model (default: gemini-3.6-flash / claude-sonnet-5-5)")
    ap.add_argument("--backend2", default=None, choices=["gemini", "anthropic", "mock"], help="second judge (inter-judge)")
    ap.add_argument("--model2", default=None)
    ap.add_argument("--generator", default="all", choices=["llama3", "ministral", "all"])
    ap.add_argument("--scope", default="subset", choices=["subset", "all"], help="j2: stratified subset or all trials")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N trials/records (for a cheap test run)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<model>)")
    a = ap.parse_args()

    backend = LJ.make_backend(a.backend, a.model)
    out = Path(a.out) if a.out else HERE / "outputs" / slug(backend.model)
    out.mkdir(parents=True, exist_ok=True)
    judge = LJ.Judge(backend, out / "judge_cache.jsonl", threads=a.threads)
    print(f"judge: {a.backend} / {backend.model} -> {out}")

    if a.task in ("validate-humans", "validation"):
        task_validate_humans(judge, out)
    if a.task in ("validate-calibration", "validation"):
        task_validate_calibration(judge, out)
    if a.task in ("j1", "bulk"):
        task_j1(judge, out, a.generator, a.limit)
    if a.task in ("ays", "bulk"):
        task_ays(judge, out, a.limit)
    if a.task in ("multiturn", "bulk"):
        task_multiturn(judge, out, a.limit)
    if a.task in ("j2", "bulk"):
        task_j2(judge, out, a.scope, a.limit)
    if a.task in ("order-bias", "bulk"):
        task_order_bias(judge, out, 100 if a.limit is None else min(100, a.limit))
    if a.task == "inter-judge":
        if not a.backend2:
            ap.error("inter-judge needs --backend2 (and usually --model2)")
        b2 = LJ.make_backend(a.backend2, a.model2)
        judge2 = LJ.Judge(b2, out.parent / slug(b2.model) / "judge_cache.jsonl", threads=a.threads)
        task_inter_judge(judge, judge2, out)
    print(f"cache: {sum(1 for r in judge.cache.values() if r.get('ok'))} successful calls in {judge.cache_path}")


if __name__ == "__main__":
    main()
