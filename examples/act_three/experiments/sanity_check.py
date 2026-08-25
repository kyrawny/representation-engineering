"""
Sanity Check — Interactive single-scenario tester.

Lets you quickly test the full ACT×RepE pipeline on a single scenario
with configurable model, identities, and user message.  Prints the
ACT target, unsteered response, steered response, and their read EPAs
side-by-side for easy comparison.

Usage::

    python -m examples.act_three.experiments.sanity_check

    # Custom identities and message
    python -m examples.act_three.experiments.sanity_check \
        --agent teacher --user student \
        --message "I don't understand this at all, can you explain it again?"

    # Use a different model
    python -m examples.act_three.experiments.sanity_check \
        --model mistralai/Ministral-8B-Instruct-2410

    # Override EPA target directly (skip ACT computation)
    python -m examples.act_three.experiments.sanity_check \
        --target-epa 2.0 0.5 -0.5
"""

import argparse
import json
import textwrap
from datetime import datetime

from .config import DIMENSION_NAMES, GENERATION_DEFAULTS
from .setup import (
    load_experiment_components,
    add_model_arg,
    get_identity_epa,
    make_system_prompt,
    generate_unsteered,
    clean_response,
)


def _format_epa(epa_dict: dict, indent: int = 4) -> str:
    """Pretty-print an EPA dict."""
    pad = " " * indent
    return (
        f"{pad}E (evaluation): {epa_dict['evaluation']:+.4f}\n"
        f"{pad}P (potency):    {epa_dict['potency']:+.4f}\n"
        f"{pad}A (activity):   {epa_dict['activity']:+.4f}"
    )


def _wrap_text(text: str, width: int = 80, indent: int = 4) -> str:
    """Wrap text with indentation."""
    pad = " " * indent
    lines = textwrap.wrap(text, width=width - indent)
    return "\n".join(pad + line for line in lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sanity Check: test steering on a single scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent", default="boss",
        help="Agent identity term (default: counselor)")
    parser.add_argument(
        "--user", default="subordinate",
        help="User identity term (default: client)")
    parser.add_argument(
        "--message", "-m", default=None,
        help="User message text. If not provided, uses a default scenario.")
    parser.add_argument(
        "--target-epa", nargs=3, type=float, default=None,
        metavar=("E", "P", "A"),
        help="Override ACT target EPA (e.g. --target-epa 2.0 0.5 -0.5)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Override max new tokens for generation")
    parser.add_argument(
        "--output", default=None,
        help="Save results to this JSON file (optional)")
    add_model_arg(parser)
    args = parser.parse_args()

    # ---- Load components ----
    print("Loading model and components...")
    comp = load_experiment_components(load_steerer=True, model_name=args.model)

    from examples.act_three import (
        EPA,
        get_response_epa_for_deflection_minimization,
    )
    from examples.act_three.model_registry import format_chat_prompt

    # ---- Resolve identities ----
    agent_term = args.agent
    user_term = args.user

    try:
        agent_epa = get_identity_epa(comp.identities_df, agent_term)
        user_epa_id = get_identity_epa(comp.identities_df, user_term)
    except (KeyError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Available identity terms:")
        terms = sorted(comp.identities_df["term"].unique())
        for i in range(0, len(terms), 5):
            print("  " + ", ".join(terms[i:i+5]))
        return

    # ---- User message ----
    user_message = args.message or (
        "I've been feeling really overwhelmed lately and I don't know "
        "what to do. Everything just seems to be falling apart."
    )

    # ---- Read user message EPA ----
    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    print(f"\n  Model:  {comp.model_name}")
    print(f"  Agent:  {agent_term} (E={agent_epa.e:+.2f}, "
          f"P={agent_epa.p:+.2f}, A={agent_epa.a:+.2f})")
    print(f"  User:   {user_term} (E={user_epa_id.e:+.2f}, "
          f"P={user_epa_id.p:+.2f}, A={user_epa_id.a:+.2f})")
    print(f"\n  User message:\n{_wrap_text(user_message)}")

    user_msg_epa = comp.reader.read_epa(
        comp.rep_reading_pipeline, user_message)
    print(f"\n  Read user message EPA:")
    print(_format_epa(user_msg_epa))

    # ---- Compute ACT target ----
    if args.target_epa:
        target_dict = {
            "evaluation": args.target_epa[0],
            "potency": args.target_epa[1],
            "activity": args.target_epa[2],
        }
        print(f"\n  Target EPA (manual override):")
    else:
        user_behavior = EPA(
            e=user_msg_epa["evaluation"],
            p=user_msg_epa["potency"],
            a=user_msg_epa["activity"],
        )
        target_epa = get_response_epa_for_deflection_minimization(
            agent_identity=agent_epa,
            user_identity=user_epa_id,
            user_behavior_epa=user_behavior,
            coefficients=comp.coefficients,
        )
        target_dict = {
            "evaluation": target_epa.e,
            "potency": target_epa.p,
            "activity": target_epa.a,
        }
        print(f"\n  Target EPA (ACT-computed):")
    print(_format_epa(target_dict))

    # ---- Build prompts ----
    sys_prompt = make_system_prompt(agent_term, user_term)
    prompt = format_chat_prompt(comp.tokenizer, sys_prompt, user_message)

    gen_kwargs = dict(GENERATION_DEFAULTS)
    if args.max_new_tokens:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    # ---- Generate unsteered ----
    print("\n" + "-" * 70)
    print("UNSTEERED RESPONSE")
    print("-" * 70)
    unsteered_text = generate_unsteered(
        comp.model, comp.tokenizer, prompt, **gen_kwargs)
    unsteered_text = clean_response(unsteered_text)
    print(f"\n{_wrap_text(unsteered_text)}")

    unsteered_epa = comp.reader.read_epa(
        comp.rep_reading_pipeline, unsteered_text)
    print(f"\n  Read EPA:")
    print(_format_epa(unsteered_epa))
    unsteered_dist = {
        dim: abs(unsteered_epa[dim] - target_dict[dim])
        for dim in DIMENSION_NAMES
    }
    print(f"\n  Distance to target:")
    for dim in DIMENSION_NAMES:
        print(f"    {dim}: {unsteered_dist[dim]:.4f}")

    # ---- Generate steered ----
    print("\n" + "-" * 70)
    print("STEERED RESPONSE (RepE)")
    print("-" * 70)
    steered_text = comp.steerer.generate(
        prompt=prompt,
        target_epa=target_dict,
        **gen_kwargs,
    )
    steered_text = clean_response(steered_text)
    print(f"\n{_wrap_text(steered_text)}")

    steered_epa = comp.reader.read_epa(
        comp.rep_reading_pipeline, steered_text)
    print(f"\n  Read EPA:")
    print(_format_epa(steered_epa))
    steered_dist = {
        dim: abs(steered_epa[dim] - target_dict[dim])
        for dim in DIMENSION_NAMES
    }
    print(f"\n  Distance to target:")
    for dim in DIMENSION_NAMES:
        print(f"    {dim}: {steered_dist[dim]:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Dimension':<15} {'Target':>8} {'Unsteered':>10} "
          f"{'Steered':>10} {'Δ Unst.':>8} {'Δ Steer.':>8}")
    print(f"  {'-'*13:<15} {'-'*8:>8} {'-'*10:>10} "
          f"{'-'*10:>10} {'-'*8:>8} {'-'*8:>8}")
    for dim in DIMENSION_NAMES:
        t = target_dict[dim]
        u = unsteered_epa[dim]
        s = steered_epa[dim]
        print(f"  {dim:<15} {t:>+8.3f} {u:>+10.3f} "
              f"{s:>+10.3f} {abs(u-t):>8.3f} {abs(s-t):>8.3f}")

    # ---- Optional JSON output ----
    if args.output:
        result = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": comp.model_name,
                "agent_identity": agent_term,
                "user_identity": user_term,
                "agent_identity_epa": {
                    "e": agent_epa.e, "p": agent_epa.p, "a": agent_epa.a,
                },
                "user_identity_epa": {
                    "e": user_epa_id.e, "p": user_epa_id.p, "a": user_epa_id.a,
                },
            },
            "user_message": user_message,
            "user_message_epa": user_msg_epa,
            "target_epa": target_dict,
            "target_source": "manual" if args.target_epa else "act_computed",
            "prompt": prompt,
            "unsteered": {
                "text": unsteered_text,
                "epa": unsteered_epa,
                "distances": unsteered_dist,
            },
            "steered": {
                "text": steered_text,
                "epa": steered_epa,
                "distances": steered_dist,
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    print()


if __name__ == "__main__":
    main()
