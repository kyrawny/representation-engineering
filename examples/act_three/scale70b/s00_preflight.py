"""
Stage 0: preflight.  Run this first; it takes about 10 minutes on 70B.

Checks:
  1. Environment (GPUs, versions, git commit) -> env_report.json
  2. The model loads under the ACT70B_* settings and produces finite logits.
  3. One unsteered and one steered generation work with the model split across
     GPUs (the steering hooks must move vectors to each layer's device).
     The steered text must differ from the unsteered text.
  4. Throughput -> projected wall-clock time for the main run.
  5. (After stage 1) FastReader agrees with the original EPAReader.

Usage::

    python -m examples.act_three.scale70b.s00_preflight --model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse
import time

import torch

from . import settings as S
from .common import (env_report, load_model, load_components, method_api, save_json,
                     verify_fast_reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--skip-reader-check", action="store_true",
                    help="Use before stage 1 has produced directions and a reader.")
    args = ap.parse_args()

    out_dir = S.results_dir(args.model)
    report = {"model": args.model, "env": env_report(), "checks": {}}
    print(report["env"])
    if report["env"].get("version_mismatches"):
        print("WARNING: package versions differ from requirements-lock.txt:",
              report["env"]["version_mismatches"])

    model, tok = load_model(args.model)
    api = method_api()
    report["checks"]["n_layers"] = model.config.num_hidden_layers
    report["checks"]["device_map_devices"] = sorted(
        {str(v) for v in getattr(model, "hf_device_map", {"": "single"}).values()})

    # 2. finite logits
    prompt = api["format_chat_prompt"](tok, api["make_system_prompt"]("counselor", "client"),
                                       "Thank you so much for helping me with this.")
    with torch.no_grad():
        ids = tok(prompt, return_tensors="pt").to(model.device)
        logits = model(**ids).logits
    report["checks"]["finite_logits"] = bool(torch.isfinite(logits).all().item())
    if not report["checks"]["finite_logits"]:
        print("FAIL: non-finite logits. Try ACT70B_DTYPE=bfloat16 for ALL stages.")

    # 4. throughput (unsteered)
    t0 = time.time()
    text = api["clean_response"](api["generate_unsteered"](model, tok, prompt))
    dt = time.time() - t0
    n_tok = len(tok(text)["input_ids"])
    report["checks"]["unsteered_example"] = text
    report["checks"]["seconds_per_generation"] = dt
    report["checks"]["tokens_per_second"] = n_tok / dt if dt > 0 else None

    if not args.skip_reader_check:
        # 3. Steered generation across devices, at a deliberately strong setting
        #    (strength 16) so that a working hook must change the output.  This
        #    tests the mechanism only; the strength used later comes from stage 4.
        comp = load_components(args.model, need_steerer=True, strength=16.0,
                               model_and_tokenizer=(model, tok))
        target = {"evaluation": 2.0, "potency": 2.5, "activity": 1.0}
        t0 = time.time()
        steered = api["clean_response"](comp.steerer.generate(
            prompt=prompt, target_epa=target, **api["GENERATION_DEFAULTS"]))
        report["checks"]["steered_seconds"] = time.time() - t0
        report["checks"]["steered_example"] = steered
        report["checks"]["steering_changes_output"] = steered != text
        # 5. reader equivalence
        texts = [text, steered,
                 "I can't believe how incompetent you are.",
                 "Could we start the meeting? I think everyone is here now.",
                 "Thank you so much, I really appreciate everything you've done.",
                 "Stop. I am in charge here and you will do as I say.",
                 "I'm so sorry, I didn't mean to cause any trouble.",
                 "This is amazing!! Let's go right now, I can't wait!"]
        report["checks"]["fast_reader"] = verify_fast_reader(comp, texts)

    # Projection for the main run (4 core conditions, 1,300 trials)
    spg = report["checks"]["seconds_per_generation"]
    spg_steer = report["checks"].get("steered_seconds", spg * 1.25)
    est = 1300 * (2 * spg + 2 * spg_steer)
    report["projection_hours_main_run_single_shard"] = round(est / 3600, 1)

    path = save_json(report, out_dir / "preflight.json")
    print("\n=== Preflight summary ===")
    for k, v in report["checks"].items():
        if k not in ("unsteered_example", "steered_example"):
            print(f"  {k}: {v}")
    print(f"  projected main run (one shard): {report['projection_hours_main_run_single_shard']} h")
    print(f"Saved {path}")
    ok = report["checks"]["finite_logits"] and report["checks"].get("steering_changes_output", True)
    fr = report["checks"].get("fast_reader")
    if fr and not fr["ok"]:
        print("NOTE: batched FastReader exceeded tolerance; later stages will use --read-batch-size 1."
              if fr.get("ok_batch1") else "FAIL: FastReader disagrees with EPAReader even at batch 1.")
        ok = ok and fr.get("ok_batch1", False)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
