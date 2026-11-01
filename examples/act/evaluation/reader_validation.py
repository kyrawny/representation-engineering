"""EPA reader validation against human-rated EPA values (S11).

Question answered: **does the RepE EPA reader track human ratings?**

The MTurk-Interact dictionary EPA values *are* human ratings (aggregates of
semantic-differential judgments), so they serve as offline ground truth:
for a set of stimulus texts grounded in dictionary terms, we compare the
reader's predicted EPA against the term's human EPA, per dimension (E, P, A)
and pooled. Agreement is reported with cross-validated folds so the estimate
does not rest on a single arbitrary split.

The reader is an injected callable ``text -> EPA`` (the GPU phase plugs in
the RepE reading pipeline); a deterministic mock reader powers the CPU
``--quick`` smoke mode.

CLI
---
::

    python -m evaluation.reader_validation --quick --out results/reader_validation.json
    python -m evaluation.reader_validation --reader mock --n-items 100 --seed 42 --out ...
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from act_steering.config import DEFAULT_MODEL, DEFAULT_SEED, DataPaths
from act_steering.dictionary import ACTDictionary
from act_steering.stimuli import RESPONSE_TEMPLATE
from evaluation import stats

__all__ = [
    "DIMENSIONS",
    "Reader",
    "ValidationItem",
    "build_mock_reader",
    "build_model_reader",
    "resolve_stimulus_template",
    "wrap_items",
    "main",
    "sample_dictionary_items",
    "validate_reader",
]

#: The three EPA dimension names.
DIMENSIONS: tuple[str, str, str] = ("E", "P", "A")

#: A reader maps a stimulus text to its predicted EPA (or raises).
Reader = Callable[[str], Any]


@dataclass(frozen=True)
class ValidationItem:
    """One validation stimulus with its human-rated reference EPA.

    Attributes:
        text: The stimulus text the reader scores.
        reference_epa: The human-rated EPA (E, P, A) from the dictionary.
        source: Where the reference came from (default ``"dictionary"``).
    """

    text: str
    reference_epa: tuple[float, float, float]
    source: str = "dictionary"


def sample_dictionary_items(
    dictionary: ACTDictionary,
    *,
    n_behaviours: int,
    n_identities: int,
    seed: int = DEFAULT_SEED,
) -> list[ValidationItem]:
    """Build validation items from MTurk-Interact terms (reproducible).

    The term's dictionary EPA is the human rating; the term itself is the
    stimulus text (the GPU reader may wrap it in a sentence template).

    Args:
        dictionary: The loaded ACT dictionary.
        n_behaviours: Number of behaviour terms to draw.
        n_identities: Number of identity terms to draw.
        seed: RNG seed for the draw.

    Returns:
        The validation items (behaviours first, then identities).
    """
    items: list[ValidationItem] = []
    for term in dictionary.sample(n_behaviours, seed, collection="behaviors"):
        items.append(ValidationItem(term.term, term.as_vector()))
    for term in dictionary.sample(n_identities, seed + 1, collection="identities"):
        items.append(ValidationItem(term.term, term.as_vector()))
    return items


def resolve_stimulus_template(arg: str, fit_template: str) -> str | None:
    """Resolve the ``--stimulus-template`` argument to a concrete template.

    ``"fit"`` returns the template the directions were fit with; ``"bare"``
    returns ``None`` (no wrapping); anything else is used as a literal
    template (and must contain a ``{behaviour}`` placeholder).

    Args:
        arg: The raw CLI value.
        fit_template: The fit template (from bundle metadata).

    Returns:
        The template string, or ``None`` for bare terms.

    Raises:
        ValueError: If a literal template lacks ``{behaviour}``.
    """
    if arg == "fit":
        return fit_template
    if arg == "bare":
        return None
    if "{behaviour}" not in arg:
        raise ValueError(f"stimulus template must contain {{behaviour}}: {arg!r}")
    return arg


def wrap_items(items: Sequence[ValidationItem], template: str | None) -> list[ValidationItem]:
    """Wrap each item's term in *template* (``None`` leaves items bare).

    The reader must be scored on the same stimulus format the directions
    were fit on, or the gate measures protocol mismatch, not validity.

    Args:
        items: The validation items (bare terms).
        template: A template with a ``{behaviour}`` placeholder, or ``None``.

    Returns:
        A new list of items (the input is not mutated).
    """
    if template is None:
        return list(items)
    return [replace(item, text=template.format(behaviour=item.text)) for item in items]


def build_mock_reader() -> Reader:
    """A deterministic hash-based mock reader (CPU smoke mode)."""

    def reader(text: str) -> tuple[float, float, float]:
        digest = hashlib.sha256(f"reader:{text}".encode()).digest()
        return tuple((digest[i] / 255.0 - 0.5) * 2.0 for i in range(3))  # type: ignore[return-value]

    return reader


def build_model_reader(model: Any, tokenizer: Any, bundle: Any) -> Reader:
    """Build the RepE EPA reader over a directions bundle (GPU phase).

    Args:
        model: HuggingFace causal LM with ``output_hidden_states``.
        tokenizer: Its tokenizer.
        bundle: A :class:`act_steering.directions.DirectionsBundle`.

    Returns:
        A :data:`Reader` callable mapping text to a predicted EPA tuple.
    """
    from act_steering.readers import build_epa_reader

    return build_epa_reader(model, tokenizer, bundle)


def _coerce_epa(name: str, value: Any) -> np.ndarray:
    """Coerce a reader output to a float (3,) array; raises on malformed.

    Accepts an :class:`act_steering.act_core.EPA`, a 3-tuple, or a (3,) array.
    """
    from act_steering.act_core import EPA

    if isinstance(value, EPA):
        return value.as_array()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name}: EPA must have shape (3,), got {arr.shape}")
    return arr


def validate_reader(
    reader: Reader,
    items: Sequence[ValidationItem],
    *,
    n_folds: int = 5,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare reader predictions against human reference EPAs.

    Args:
        reader: The reader callable (``text -> EPA``); failures are counted
            and the item is excluded from agreement.
        items: The validation items.
        n_folds: Number of cross-validation folds (capped by item count).
        seed: RNG seed for the fold split.

    Returns:
        A JSON-safe report with ``n_items``, ``n_scorable``, ``n_failed``,
        ``per_dimension`` (Pearson / Spearman / MAE / CCC for E, P, A),
        ``overall`` (pooled metrics plus mean 3-D distance), and ``folds``
        (per-fold per-dimension agreement).

    Raises:
        ValueError: If no items are given.
    """
    if not items:
        raise ValueError("validate_reader: no validation items")

    references: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    n_failed = 0
    for item in items:
        try:
            pred = _coerce_epa("reader output", reader(item.text))
        except Exception:  # noqa: BLE001 - reader failures are counted, not raised
            n_failed += 1
            continue
        references.append(np.asarray(item.reference_epa, dtype=np.float64))
        predictions.append(pred)

    per_dimension: dict[str, dict[str, float]] = {}
    for d, dim in enumerate(DIMENSIONS):
        ref_col = np.array([r[d] for r in references], dtype=np.float64)
        pred_col = np.array([p[d] for p in predictions], dtype=np.float64)
        if ref_col.size < 2:
            per_dimension[dim] = {"n": 0.0}
            continue
        per_dimension[dim] = {"n": float(ref_col.size), **stats.agreement_stats(ref_col, pred_col)}

    pooled_ref = np.concatenate([r for r in references]) if references else np.zeros(0)
    pooled_pred = np.concatenate([p for p in predictions]) if predictions else np.zeros(0)
    overall: dict[str, float]
    if pooled_ref.size >= 2:
        overall = {"n": float(pooled_ref.size), **stats.agreement_stats(pooled_ref, pooled_pred)}
        distances = np.linalg.norm(np.stack(predictions) - np.stack(references), axis=1)
        overall["mean_3d_distance"] = float(distances.mean())
    else:
        overall = {"n": 0.0}

    # cross-validated folds: deterministic split of the scorable items
    n_scorable = len(references)
    folds: list[dict[str, Any]] = []
    if n_scorable >= 2:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_scorable)
        effective_folds = min(n_folds, n_scorable)
        fold_slices = np.array_split(order, effective_folds)
        for i, sl in enumerate(fold_slices):
            fold_ref = np.array([references[j] for j in sl], dtype=np.float64)
            fold_pred = np.array([predictions[j] for j in sl], dtype=np.float64)
            if fold_ref.shape[0] < 2:
                continue
            fold_per_dim: dict[str, Any] = {}
            for d, dim in enumerate(DIMENSIONS):
                fold_per_dim[dim] = stats.agreement_stats(fold_ref[:, d], fold_pred[:, d])
            folds.append({"fold": i, "n": float(fold_ref.shape[0]), "per_dimension": fold_per_dim})

    return {
        "n_items": float(len(items)),
        "n_scorable": float(n_scorable),
        "n_failed": float(n_failed),
        "per_dimension": per_dimension,
        "overall": overall,
        "folds": folds,
    }


# --- CLI -----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reader validation and write a JSON report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        SystemExit: When the directions artifact is missing in model mode.
    """
    parser = argparse.ArgumentParser(description="Validate the EPA reader against human-rated EPA.")
    parser.add_argument("--out", required=True, type=Path, help="Output report JSON path")
    parser.add_argument("--reader", choices=("mock", "model"), default="mock")
    parser.add_argument("--quick", action="store_true", help="Smoke mode: mock reader, 40 items")
    parser.add_argument("--n-items", type=int, default=60, help="Total validation items")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (model reader)")
    parser.add_argument(
        "--directions", type=Path, default=None,
        help="Directions .npz (default: artifacts/directions_<model>.npz)",
    )
    parser.add_argument("--device", default="cuda", help="torch device (model reader)")
    parser.add_argument("--dtype", default="float16", help="torch dtype (model reader)")
    parser.add_argument(
        "--stimulus-template",
        default="fit",
        help=(
            "template to wrap validation terms in before reading: 'fit' "
            "(default; the template the directions were fit with, from the "
            "bundle metadata), 'bare' (no wrapping — robustness check), or a "
            "literal template with a {behaviour} placeholder"
        ),
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.reader = "mock"
        args.n_items = min(args.n_items, 40)

    if args.reader == "model":
        from act_steering.directions import (
            DirectionsBundle,
            load_model_and_pipeline,
            resolve_directions_paths,
        )

        npz_path, json_path = resolve_directions_paths(args.directions, args.model)
        if not npz_path.exists():
            raise SystemExit(
                f"directions artifact {npz_path} not found; run "
                f"`python -m act_steering.directions --model {args.model}` first"
            )
        model, tokenizer, _ = load_model_and_pipeline(args.model, args.device, args.dtype)
        bundle = DirectionsBundle.load(npz_path, json_path)
        reader = build_model_reader(model, tokenizer, bundle)
        fit_template = (bundle.epa_templates or [RESPONSE_TEMPLATE])[0]
    else:
        reader = build_mock_reader()
        fit_template = RESPONSE_TEMPLATE

    stimulus_template = resolve_stimulus_template(args.stimulus_template, fit_template)

    n_behaviours = int(np.ceil(args.n_items * 2 / 3))
    n_identities = args.n_items - n_behaviours
    data = DataPaths()
    dictionary = ACTDictionary.load(data.data_dir)
    items = sample_dictionary_items(
        dictionary, n_behaviours=n_behaviours, n_identities=n_identities, seed=args.seed
    )
    items = wrap_items(items, stimulus_template)
    report = validate_reader(reader, items, n_folds=args.n_folds, seed=args.seed)
    meta = stats.build_meta(
        model=args.model if args.reader == "model" else None,
        seed=args.seed,
        config={
            "reader": args.reader,
            "n_items": len(items),
            "n_folds": args.n_folds,
            "stimulus_template": stimulus_template or "bare",
        },
    )
    stats.write_results(args.out, report, meta)
    overall = report["overall"]
    print(
        f"wrote reader validation for {int(report['n_scorable'])}/{int(report['n_items'])} items "
        f"(pearson_r={overall.get('pearson_r', float('nan')):.3f}) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
