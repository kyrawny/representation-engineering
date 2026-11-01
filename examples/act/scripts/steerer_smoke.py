"""GPU smoke: steered versus unsteered generation with monitor traces (S8.2.2).

Generates one response to a scripted ACT event (user identity, agent
identity, message behaviour), once unsteered and once under the static
deflection steerer, scores both responses with the per-token deflection
monitor, and writes a JSON log containing both generation traces.

Prerequisites: a directions artifact (GPU runbook step 1) and a free
GPU. Run from ``examples/act``:

    python scripts/steerer_smoke.py --out results/steerer_smoke.json

The event's event-1 deflection ``U_pre`` is computed from the
MTurk-Interact dictionaries (same convention as ``stimuli``) and drives
the steerer's proportional gain for the steered generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

#: ``examples/act`` on ``sys.path`` so the plain-script invocation resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_steering.act_core import ACTCoefficients  # noqa: E402
from act_steering.config import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DataPaths,
    MonitorConfig,
    SteeringConfig,
)
from act_steering.dictionary import ACTDictionary, EPATerm  # noqa: E402
from act_steering.directions import (  # noqa: E402
    DirectionsBundle,
    load_model_and_pipeline,
)
from act_steering.monitor import (  # noqa: E402
    DeflectionMonitor,
    make_hidden_provider,
    resolve_reader_layers,
)
from act_steering.steerer import DeflectionSteerer  # noqa: E402
from act_steering.stimuli import CONTEXT_TEMPLATE  # noqa: E402
from evaluation.stats import build_meta, write_results  # noqa: E402

_LOGGER = logging.getLogger(__name__)

#: Default scripted event (all terms exist in the MTurk-Interact dictionaries).
DEFAULT_USER = "student"
DEFAULT_AGENT = "teacher"
DEFAULT_MESSAGE = "criticize"


def _lookup(terms: list[EPATerm], term: str, kind: str) -> EPATerm:
    """Find a dictionary term by name or exit with a clear error."""
    for entry in terms:
        if entry.term == term:
            return entry
    valid = ", ".join(t.term for t in terms[:8])
    raise SystemExit(
        f"{kind} term {term!r} not in the dictionary (e.g. {valid}, ...)"
    )


def _event1_deflection(
    coeffs: ACTCoefficients, user: EPATerm, agent: EPATerm, message: EPATerm
) -> float:
    """Deflection U of event 1 (user acts on agent with the message behaviour)."""
    s = np.array(user.as_vector(), dtype=np.float64)
    b = np.array(message.as_vector(), dtype=np.float64)
    o = np.array(agent.as_vector(), dtype=np.float64)
    w = coeffs.impressions(s, b, o)
    return float(np.sum((w - np.concatenate([s, b, o])) ** 2))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(
        description="Steered vs unsteered GPU smoke with monitor traces."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument(
        "--directions", type=Path, default=None,
        help="Directions .npz (default: artifacts/directions_<model>.npz)",
    )
    parser.add_argument("--user", default=DEFAULT_USER, help="User identity term")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Agent identity term")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Message behaviour term")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=0.5, help="Steering base strength")
    parser.add_argument("--threshold", type=float, default=1.0, help="Monitor trigger threshold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Recorded in the log")
    parser.add_argument(
        "--out", type=Path, default=Path("results/steerer_smoke.json"),
        help="Output JSON log path",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # 1. ACT event: look up terms and compute U_pre.
    paths = DataPaths()
    dictionary = ACTDictionary.load(paths.data_dir)
    coeffs = ACTCoefficients.load(paths.coefficients_file)
    user = _lookup(dictionary.identities, args.user, "user")
    agent = _lookup(dictionary.identities, args.agent, "agent")
    message = _lookup(dictionary.behaviors, args.message, "message")
    u_pre = _event1_deflection(coeffs, user, agent, message)
    prompt = CONTEXT_TEMPLATE.format(
        user=user.term, agent=agent.term, message=message.term
    )
    _LOGGER.info("event: user=%s agent=%s message=%s U_pre=%.4f",
                 user.term, agent.term, message.term, u_pre)

    # 2. Directions artifact.
    npz = args.directions
    if npz is None:
        slug = "".join(
            c if c.isalnum() else "-" for c in args.model.lower()
        ).strip("-")
        npz = Path("artifacts") / f"directions_{slug}.npz"
    if not npz.exists():
        raise SystemExit(
            f"directions artifact {npz} not found; run the directions CLI first "
            "(GPU runbook step 1) or pass --directions"
        )
    bundle = DirectionsBundle.load(npz, npz.with_suffix(".json"))

    # 3. Model + steerer + monitor.
    import torch

    model, tokenizer, _ = load_model_and_pipeline(args.model, args.device, args.dtype)
    steerer = DeflectionSteerer(
        model, tokenizer, bundle,
        SteeringConfig(alpha=args.alpha, mode="static"),
    )
    monitor_config = MonitorConfig(threshold=args.threshold)
    provider = make_hidden_provider(
        model, tokenizer, resolve_reader_layers(bundle, monitor_config)
    )
    monitor = DeflectionMonitor(provider, bundle, monitor_config)

    def n_response_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # 4. Unsteered generation.
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    unsteered = tokenizer.decode(
        ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )
    unsteered_trace = monitor.trace(
        prompt + unsteered, n_response_tokens(unsteered)
    )
    _LOGGER.info("unsteered response (%d tokens): %s",
                 n_response_tokens(unsteered), unsteered[:120])

    # 5. Steered generation (proportional gain from U_pre).
    steered_full = steerer.generate(
        prompt,
        u_pre=u_pre,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_text=True,
    )
    steered_text = steered_full[0] if isinstance(steered_full, (list, tuple)) else steered_full
    steered = steered_text[len(prompt):] if steered_text.startswith(prompt) else steered_text
    steered_trace = monitor.trace(
        prompt + steered, n_response_tokens(steered)
    )
    _LOGGER.info("steered response (%d tokens): %s",
                 n_response_tokens(steered), steered[:120])

    # 6. Log.
    payload = {
        "prompt": prompt,
        "event": {
            "user": user.term,
            "agent": agent.term,
            "message": message.term,
            "u_pre": u_pre,
        },
        "unsteered": {
            "response": unsteered,
            "monitor": unsteered_trace.to_dict(),
        },
        "steered": {
            "response": steered,
            "alpha": steerer.config.alpha,
            "monitor": steered_trace.to_dict(),
        },
        "trace_diff": {
            "mean": steered_trace.mean - unsteered_trace.mean,
            "n_above": {
                "unsteered": unsteered_trace.n_above,
                "steered": steered_trace.n_above,
            },
        },
    }
    meta = build_meta(
        model=args.model,
        seed=args.seed,
        config={
            "max_new_tokens": args.max_new_tokens,
            "alpha": args.alpha,
            "threshold": args.threshold,
            "directions": str(npz),
        },
    )
    write_results(args.out, payload, meta)
    print(f"wrote {args.out}")
    print(
        "monitor mean: unsteered "
        f"{unsteered_trace.mean:.4f} -> steered {steered_trace.mean:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
