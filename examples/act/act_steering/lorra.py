"""LoRRA persistence mode: deflection-contrast triplets (S13, optional).

LoRRA (Representation Learning via Representation Alignment) fine-tunes a
model so that a *positive control* statement raises the probability of a
desired response and a *negative control* statement lowers it, by aligning
the model's internal representation towards/away from a control direction.
The repository ships the upstream trainer under ``lorra_finetune/``; this
module is the ACT adapter for it: it turns the distribution-matched
low/high-deflection contrast set (from :mod:`act_steering.stimuli`) into
LoRRA triplets in the exact template format the upstream dataset expects.

Triplet scheme (one triplet per contrast pair, upstream-faithful):

* ``instruction``: the shared trial context (user/agent/message);
* ``response``: the **low-deflection** continuation — desired under the
  positive control, undesired under the negative control (upstream uses one
  shared response per triplet);
* ``orig``:  ``{user_tag} {instruction} {assistant_tag} {response}``
* ``pos``:   ``{user_tag} {instruction} {pos_control} {assistant_tag} {response}``
* ``neg``:   ``{user_tag} {instruction} {neg_control} {assistant_tag} {response}``

where the controls are ``Give a low-deflection response.`` /
``Give a high-deflection response.`` by default.

This module is torch-free (layering rule: it only reshapes text data).
The training CLI wrapping ``lorra_finetune`` is added in the same module
(S13.2); the actual fine-tune runs in the GPU phase.

CLI
---
::

    python -m act_steering.lorra --quick --out artifacts/lorra_triplets.jsonl
    python -m act_steering.lorra --contrast-set artifacts/contrast_set.json \
        --out artifacts/lorra_triplets.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .stimuli import (
    RESPONSE_TEMPLATE,
    ContrastPair,
    dicts_to_pairs,
)

__all__ = [
    "DEFAULT_POS_TYPE",
    "DEFAULT_NEG_TYPE",
    "DEFAULT_CONTROL_TEMPLATE",
    "DEFAULT_REPO_ROOT",
    "DEFAULT_CONTRAST_SET",
    "DEFAULT_TRIPLETS_OUT",
    "DATASET_HOOK_NOTE",
    "LorraArgs",
    "build_triplet",
    "build_triplets",
    "build_train_command",
    "load_contrast_set",
    "write_triplets_jsonl",
    "main",
]

_LOGGER = logging.getLogger(__name__)

#: Upstream triplet templates (lorra_finetune/src/train_val_datasets.py).
ORIG_TEMPLATE = "{user_tag} {instruction} {assistant_tag} {response}"
POS_TEMPLATE = "{user_tag} {instruction} {type} {assistant_tag} {response}"
NEG_TEMPLATE = "{user_tag} {instruction} {type} {assistant_tag} {response}"

#: Default ACT control statements (fill ``{type}`` with pos/neg type).
DEFAULT_POS_TYPE = "a low-deflection"
DEFAULT_NEG_TYPE = "a high-deflection"
DEFAULT_CONTROL_TEMPLATE = "Give a {type} response."


@dataclass(frozen=True)
class LorraArgs:
    """Data-relevant LoRRA arguments (mirrors ``lorra_finetune/src/args.py``).

    Attributes:
        user_tag: User turn tag for chat models (e.g., ``[INST]``).
        assistant_tag: Assistant turn tag (e.g., ``[/INST]``).
        pos_type: Positive control type phrase (e.g., ``a low-deflection``).
        neg_type: Negative control type phrase (e.g., ``a high-deflection``).
        control_template: Control statement template with a ``{type}`` slot.
        max_res_len: Truncated response length used by the upstream trainer.
        lorra_alpha: LoRRA alignment strength (upstream default 5).
        lorra_beta: LoRRA secondary loss weight (upstream default 0).
    """

    user_tag: str = "[INST]"
    assistant_tag: str = "[/INST]"
    pos_type: str = DEFAULT_POS_TYPE
    neg_type: str = DEFAULT_NEG_TYPE
    control_template: str = DEFAULT_CONTROL_TEMPLATE
    max_res_len: int = 64
    lorra_alpha: float = 5.0
    lorra_beta: float = 0.0

    def pos_control(self) -> str:
        """The positive control statement for this configuration."""
        return self.control_template.format(type=self.pos_type)

    def neg_control(self) -> str:
        """The negative control statement for this configuration."""
        return self.control_template.format(type=self.neg_type)


def build_triplet(pair: ContrastPair, args: LorraArgs) -> dict[str, Any]:
    """Build one LoRRA triplet for one contrast pair.

    The shared response is the low-deflection continuation (desired under
    the positive control, undesired under the negative control), matching
    the upstream single-response triplet scheme.

    Args:
        pair: One low/high deflection contrast pair (shared context).
        args: The LoRRA arguments (tags, control statements).

    Returns:
        A dict with ``instruction``, ``response``, ``orig``, ``pos``,
        ``neg`` (the three template-rendered strings) plus provenance
        fields (``user_term``, ``agent_term``, ``low_behaviour``,
        ``high_behaviour``, ``delta_u``).
    """
    instruction = pair.low.context
    response = RESPONSE_TEMPLATE.format(behaviour=pair.low.behaviour_term)
    common = {
        "user_tag": args.user_tag,
        "instruction": instruction,
        "assistant_tag": args.assistant_tag,
        "response": response,
    }
    return {
        "instruction": instruction,
        "response": response,
        "orig": ORIG_TEMPLATE.format(**common),
        "pos": POS_TEMPLATE.format(**common, type=args.pos_control()),
        "neg": NEG_TEMPLATE.format(**common, type=args.neg_control()),
        "user_term": pair.low.user_term,
        "agent_term": pair.low.agent_term,
        "low_behaviour": pair.low.behaviour_term,
        "high_behaviour": pair.high.behaviour_term,
        "delta_u": float(pair.delta_u),
    }


def build_triplets(
    pairs: list[ContrastPair],
    args: LorraArgs,
    *,
    max_triplets: int | None = None,
    min_delta_u: float = 0.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build LoRRA triplets for a list of contrast pairs.

    Args:
        pairs: The contrast pairs (shared-context low/high deflection).
        args: The LoRRA arguments.
        max_triplets: Optional cap; when the kept pairs exceed it, a seeded
            without-replacement subsample is taken.
        min_delta_u: Pairs with a smaller ``delta_u`` are dropped.
        seed: Seed for the subsampling RNG.

    Returns:
        The list of triplet dicts (one per kept pair, original order).

    Raises:
        ValueError: If ``max_triplets`` is not positive or ``min_delta_u``
            is negative.
    """
    if max_triplets is not None and max_triplets <= 0:
        raise ValueError(f"max_triplets must be positive, got {max_triplets}")
    if min_delta_u < 0.0:
        raise ValueError(f"min_delta_u must be >= 0, got {min_delta_u}")
    kept = [pair for pair in pairs if pair.delta_u >= min_delta_u]
    if max_triplets is not None and len(kept) > max_triplets:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(kept), size=max_triplets, replace=False))
        kept = [kept[int(i)] for i in indices]
    return [build_triplet(pair, args) for pair in kept]


def load_contrast_set(path: Path) -> list[ContrastPair]:
    """Load a contrast-set JSON written by the ``stimuli`` CLI.

    Args:
        path: Path to the contrast-set JSON file.

    Returns:
        The contrast pairs.

    Raises:
        ValueError: If the JSON has no ``pairs`` list.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "pairs" not in data or not isinstance(data["pairs"], list):
        raise ValueError(f"{path} has no 'pairs' list")
    return dicts_to_pairs(data["pairs"])


def write_triplets_jsonl(triplets: list[dict[str, Any]], path: Path) -> Path:
    """Write triplets to a JSONL file (one JSON object per line).

    Args:
        triplets: The triplet dicts.
        path: Destination path (parent directories are created).

    Returns:
        The destination path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for triplet in triplets:
            fh.write(json.dumps(triplet, ensure_ascii=False) + "\n")
    _LOGGER.info("wrote %d LoRRA triplets to %s", len(triplets), out)
    return out


#: Repository root (``examples/act/act_steering/lorra.py`` -> 3 levels up).
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Default contrast-set artifact (written by the ``stimuli`` CLI).
DEFAULT_CONTRAST_SET = Path(__file__).resolve().parents[1] / "artifacts" / "contrast_set.json"
#: Default triplet output artifact.
DEFAULT_TRIPLETS_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "lorra_triplets.jsonl"

#: Integration note printed with the dry-run training command.
DATASET_HOOK_NOTE = (
    "NOTE: the upstream AlpacaSupervisedDataset "
    "(lorra_finetune/src/train_val_datasets.py) currently reads the "
    "tatsu-lab/alpaca dataset. For an ACT run, point it at the triplets "
    "JSONL written by this CLI (one-line dataset hook, see the README "
    "'LoRRA persistence mode' section)."
)


def build_train_command(
    *,
    model: str,
    repo_root: Path,
    lorra_args: LorraArgs,
    output_dir: Path,
    target_layers: str = "20,22,24",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    python: str | None = None,
) -> list[str]:
    """Compose the upstream ``lorra_finetune`` training invocation.

    The command passes the ACT LoRRA arguments (control statements, tags,
    alpha/beta, truncated response length) plus a minimal HuggingFace
    training configuration. The upstream entry point parses standard
    ``HfArgumentParser`` flags.

    Args:
        model: Model name or path for ``--model_name_or_path``.
        repo_root: Repository root containing ``lorra_finetune/``.
        lorra_args: The LoRRA arguments to forward.
        output_dir: HF ``--output_dir`` for the run.
        target_layers: Comma-separated hidden-state layer indices.
        epochs: ``--num_train_epochs``.
        batch_size: ``--per_device_train_batch_size``.
        learning_rate: ``--learning_rate``.
        python: Python interpreter (defaults to the current one).

    Returns:
        The argv list (safe to log or pass to ``subprocess.run``).
    """
    entry = Path(repo_root) / "lorra_finetune" / "src" / "llama2_lorra.py"
    return [
        python or sys.executable,
        str(entry),
        "--model_name_or_path", model,
        "--user_tag", lorra_args.user_tag,
        "--assistant_tag", lorra_args.assistant_tag,
        "--pos_type", lorra_args.pos_type,
        "--neg_type", lorra_args.neg_type,
        "--control_template", lorra_args.control_template,
        "--target_layers", target_layers,
        "--lorra_alpha", str(lorra_args.lorra_alpha),
        "--lorra_beta", str(lorra_args.lorra_beta),
        "--max_res_len", str(lorra_args.max_res_len),
        "--output_dir", str(output_dir),
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--overwrite_output_dir",
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring).

    Default mode builds the triplets JSONL from the contrast set. With
    ``--train`` the triplets are built first and the upstream training
    command is printed (dry-run); with ``--train --execute`` it is run
    via subprocess (GPU phase).
    """
    default_args = LorraArgs()
    parser = argparse.ArgumentParser(
        description="Build LoRRA deflection-contrast triplets (optionally wrap training)."
    )
    parser.add_argument(
        "--contrast-set", type=Path, default=DEFAULT_CONTRAST_SET,
        help="Contrast-set JSON from the stimuli CLI (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_TRIPLETS_OUT,
        help="Triplet JSONL output (default: %(default)s)",
    )
    parser.add_argument("--max-triplets", type=int, default=None)
    parser.add_argument("--min-delta-u", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Smoke: cap at 20 triplets")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--train", action="store_true",
        help="Also compose the lorra_finetune training command (dry-run)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Run the composed training command (GPU)",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--target-layers", default="20,22,24")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--user-tag", default=default_args.user_tag)
    parser.add_argument("--assistant-tag", default=default_args.assistant_tag)
    parser.add_argument("--pos-type", default=default_args.pos_type)
    parser.add_argument("--neg-type", default=default_args.neg_type)
    parser.add_argument("--control-template", default=default_args.control_template)
    parser.add_argument("--max-res-len", type=int, default=default_args.max_res_len)
    parser.add_argument("--lorra-alpha", type=float, default=default_args.lorra_alpha)
    parser.add_argument("--lorra-beta", type=float, default=default_args.lorra_beta)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.execute and not args.train:
        parser.error("--execute requires --train")
    max_triplets = 20 if args.quick else args.max_triplets
    lorra_args = LorraArgs(
        user_tag=args.user_tag,
        assistant_tag=args.assistant_tag,
        pos_type=args.pos_type,
        neg_type=args.neg_type,
        control_template=args.control_template,
        max_res_len=args.max_res_len,
        lorra_alpha=args.lorra_alpha,
        lorra_beta=args.lorra_beta,
    )
    pairs = load_contrast_set(args.contrast_set)
    triplets = build_triplets(
        pairs, lorra_args,
        max_triplets=max_triplets,
        min_delta_u=args.min_delta_u,
        seed=args.seed,
    )
    write_triplets_jsonl(triplets, args.out)
    if not args.train:
        return 0
    output_dir = args.output_dir or args.out.parent / "lorra_run"
    command = build_train_command(
        model=args.model,
        repo_root=args.repo_root,
        lorra_args=lorra_args,
        output_dir=output_dir,
        target_layers=args.target_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print("# LoRRA training command (dry-run):")
    print(" ".join(shlex.quote(part) for part in command))
    print(DATASET_HOOK_NOTE)
    if args.execute:
        return int(subprocess.run(command).returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
