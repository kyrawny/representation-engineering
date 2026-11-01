"""Direction extraction for deflection steering (S6).

Extracts two families of directions from the generation model using the
upstream RepE ``rep-reading`` pipeline:

* **Deflection-contrast direction** (primary actuation axis): one direction
  per layer, fit by PCA on the hidden-state differences between the
  low- and high-deflection continuations of the seeded contrast set
  (``act_steering.stimuli``). The signed direction points toward
  *low* deflection, so adding ``alpha * d`` to an activation moves it in
  the direction that reduces deflection.
* **EPA dimension directions** (ablation + interpretability axis): one
  direction per EPA dimension, fit on extreme-behaviour contrast sets built
  from the MTurk-Interact dictionary (the most and least extreme behaviour
  terms on that dimension). Signed to point toward *high* values of the
  dimension. With ``lowdin=True`` the three directions are orthonormalised
  per layer by symmetric (Løwdin) orthogonalisation, which treats all
  dimensions symmetrically (no ordering bias, unlike Gram-Schmidt).

Saved vectors are already sign-corrected and unit-normalised. Project raw
hidden states directly onto them; do *not* re-apply ``direction_signs``
(``RepReader.transform`` never applies signs, so a second sign flip would
invert the direction).

The pure-logic half of this module (text formatting, contrast-set
construction, normalisation, orthogonalisation, bundle save/load) is
torch-free and CPU-testable. Model-bound functions import ``torch`` /
``transformers`` / ``repe`` lazily.

CLI (smoke mode with ``--quick``; requires a GPU)::

    python -m act_steering.directions --quick
    python -m act_steering.directions --layers 0,8,16,24,31 --lowdin
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_MODEL, DEFAULT_SEED, DataPaths
from .dictionary import ACTDictionary
from .stimuli import RESPONSE_TEMPLATE, ContrastPair, dicts_to_pairs

__all__ = [
    "EPA_DIMENSIONS",
    "EPA_DIM_ATTR",
    "DEFLECTION_DIRECTION",
    "BUNDLE_FORMAT",
    "DEFAULT_MAX_LENGTH",
    "item_text",
    "contrast_pair_to_texts",
    "pairs_to_training_inputs",
    "build_epa_contrast_items",
    "mix_pair_order",
    "average_signed_directions",
    "l2_normalize",
    "lowdin_orthonormalize",
    "cosine_matrix",
    "cosine_stats",
    "reader_directions_to_vectors",
    "decoder_layers_to_hf_indices",
    "parse_layers",
    "DirectionsBundle",
    "default_directions_paths",
    "resolve_directions_paths",
    "load_model_and_pipeline",
    "extract_deflection_direction",
    "extract_epa_directions",
    "extract_all",
    "main",
]

_LOGGER = logging.getLogger(__name__)

#: EPA dimension names, in the canonical E, P, A order.
EPA_DIMENSIONS: tuple[str, ...] = ("evaluation", "potency", "activity")

#: Dimension name -> EPATerm attribute holding that coordinate.
EPA_DIM_ATTR: dict[str, str] = {"evaluation": "e", "potency": "p", "activity": "a"}

#: Bundle key of the primary deflection-contrast direction.
DEFLECTION_DIRECTION = "deflection_contrast"

#: Artifact format tag stored in the JSON metadata.
BUNDLE_FORMAT = "act-directions-v1"

#: Last input token is the reading/steering position (matches the AAAI setup).
DEFAULT_REP_TOKEN = -1

#: Tokenisation cap for extraction forward passes.
DEFAULT_MAX_LENGTH = 512

#: Default number of extreme behaviour terms per side per EPA dimension.
DEFAULT_EPA_ITEMS_PER_SIDE = 60


# ---------------------------------------------------------------------------
# Contrast text formatting (pure)
# ---------------------------------------------------------------------------


def item_text(context: str, behaviour: str) -> str:
    """Full utterance text for one contrast item (context + response)."""
    return f"{context} {RESPONSE_TEMPLATE.format(behaviour=behaviour)}"


def contrast_pair_to_texts(pair: ContrastPair) -> tuple[str, str]:
    """Return ``(low_text, high_text)`` for one contrast pair.

    Both sides share the trial context; only the response behaviour
    differs (context invariance of the stimulus design).
    """
    return (
        item_text(pair.low.context, pair.low.behaviour_term),
        item_text(pair.high.context, pair.high.behaviour_term),
    )


def pairs_to_training_inputs(
    pairs: list[ContrastPair],
) -> tuple[list[str], list[list[int]]]:
    """Flatten contrast pairs into ``rep-reading`` training inputs.

    The upstream ``get_directions`` computes ``hidden[::2] -
    hidden[1::2]``, i.e. first minus second of each pair. Pairs are
    ordered ``(low, high)`` and labelled ``[1, 0]`` so label 1 marks the
    low-deflection side; the pipeline's sign determination then orients
    the direction so low-deflection continuations score higher.
    """
    inputs: list[str] = []
    labels: list[list[int]] = []
    for pair in pairs:
        low, high = contrast_pair_to_texts(pair)
        inputs.extend((low, high))
        labels.append([1, 0])
    return inputs, labels


def mix_pair_order(
    pairs: Sequence[tuple[str, str]], seed: int
) -> list[tuple[str, str]]:
    """Deterministically flip the order of ~half the pairs (seeded).

    Upstream ``cluster_mean`` direction fitting computes
    ``mean(differences of label-1 pairs) - mean(differences of label-0
    pairs)``, which only separates the high/low signal when both pair
    orderings occur (a single ordering leaves one cluster empty). Seeded
    flips keep the contrast set reproducible.

    Args:
        pairs: ``((first, second), ...)`` side-term pairs.
        seed: RNG seed; the same seed reproduces the same flips.

    Returns:
        The pairs with a seeded subset flipped to ``(second, first)``.
    """
    rng = np.random.default_rng(seed)
    flip = rng.random(len(pairs)) < 0.5
    return [(b, a) if f else (a, b) for (a, b), f in zip(pairs, flip)]


def build_epa_contrast_items(
    dictionary: ACTDictionary,
    n_per_side: int,
    template: str = RESPONSE_TEMPLATE,
    templates: Sequence[str] | None = None,
    label_format: str = "nested",
    mix_seed: int | None = None,
) -> dict[str, tuple[list[str], list[list[int]] | list[int]]]:
    """Build extreme-behaviour contrast items for each EPA dimension.

    Behaviour terms are ranked by their human-rated value on the
    dimension; the ``n_per_side`` most and least extreme terms form
    rank-matched pairs — i-th highest with i-th lowest — so every
    contrast is a maximally separated pair.

    Label semantics (both mark the high-value side):
      * ``"nested"`` (for direction method ``"pca"``): one ``[1, 0]`` /
        ``[0, 1]`` per pair — the 1 marks the high side of that pair.
      * ``"flat"`` (for direction method ``"cluster_mean"``): one int
        per pair — 1 if the *first* emitted side is the high side.

    Args:
        dictionary: Loaded MTurk-Interact dictionaries.
        n_per_side: Extreme terms per side per dimension.
        template: Text template with a ``{behaviour}`` placeholder
            (used when ``templates`` is None).
        templates: Optional list of templates; items are built per
            template and concatenated (multi-template fitting lever).
        label_format: ``"nested"`` or ``"flat"`` (see above).
        mix_seed: If given, pair orderings are seeded-flipped via
            :func:`mix_pair_order` so both orderings occur (required
            for a non-degenerate ``cluster_mean`` fit).

    Returns:
        ``{dimension: (inputs, labels)}`` ready for ``get_directions``.

    Raises:
        ValueError: If ``n_per_side`` exceeds half the behaviour pool,
            the label format is unknown, or a template lacks
            ``{behaviour}``.
    """
    if templates is None:
        templates = (template,)
    for tpl in templates:
        if "{behaviour}" not in tpl:
            raise ValueError(f"template must contain {{behaviour}}: {tpl!r}")
    if label_format not in ("nested", "flat"):
        raise ValueError(
            f"label_format must be 'nested' or 'flat', got {label_format!r}"
        )
    max_side = len(dictionary.behaviors) // 2
    if not 0 < n_per_side <= max_side:
        raise ValueError(f"n_per_side must be in (0, {max_side}], got {n_per_side}")
    items: dict[str, tuple[list[str], list[list[int]] | list[int]]] = {}
    for dim in EPA_DIMENSIONS:
        attr = EPA_DIM_ATTR[dim]
        ranked = sorted(dictionary.behaviors, key=lambda t: (getattr(t, attr), t.term))
        low = ranked[:n_per_side]
        high = ranked[-n_per_side:]
        base_pairs = [(hi.term, lo.term) for lo, hi in zip(low, reversed(high))]
        inputs: list[str] = []
        labels: list[list[int]] | list[int] = []
        for tpl in templates:
            mixed = (
                mix_pair_order(base_pairs, mix_seed) if mix_seed is not None else base_pairs
            )
            for (hi_t, _lo_t), (first, second) in zip(base_pairs, mixed):
                inputs.append(tpl.format(behaviour=first))
                inputs.append(tpl.format(behaviour=second))
                high_first = first == hi_t
                if label_format == "nested":
                    labels.append([1, 0] if high_first else [0, 1])
                else:
                    labels.append(1 if high_first else 0)
        items[dim] = (inputs, labels)
    return items


# ---------------------------------------------------------------------------
# Linear algebra (pure numpy)
# ---------------------------------------------------------------------------


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-norm float64 copy; zero/non-finite input maps to zeros."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not np.isfinite(norm):
        _LOGGER.warning("l2_normalize received a zero or non-finite vector")
        return np.zeros_like(v)
    return v / norm


def average_signed_directions(
    per_template: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Average per-template signed unit directions into one set.

    Multi-template fitting lever: each template's directions are already
    signed toward high values of the dimension and L2-normalised. The
    per-layer average of the signed unit vectors keeps the shared
    high-side signal while cancelling template-specific noise; the result
    is L2-renormalised.

    Args:
        per_template: ``{template: {layer: (hidden_size,) unit vector}}``.

    Returns:
        ``{layer: (hidden_size,) unit vector}`` over the common layers.

    Raises:
        ValueError: If the input is empty, the layer sets are disjoint, or
            a per-layer average is degenerate (templates point in exactly
            cancelling directions).
    """
    if not per_template:
        raise ValueError("per_template is empty")
    common = sorted(set.intersection(*(set(v) for v in per_template.values())))
    if not common:
        raise ValueError("no common layers across template direction sets")
    out: dict[int, np.ndarray] = {}
    for layer in common:
        stack = np.stack(
            [np.asarray(v[layer], dtype=np.float64) for v in per_template.values()]
        )
        acc = stack.mean(axis=0)
        norm = float(np.linalg.norm(acc))
        if norm <= 1e-12 or not np.isfinite(norm):
            raise ValueError(
                f"degenerate template direction average at layer {layer} "
                "(zero/non-finite norm — templates point in cancelling directions)"
            )
        out[layer] = acc / norm
    return out


def lowdin_orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Symmetric (Løwdin) orthonormalisation of a direction matrix.

    For a matrix ``D`` whose columns are directions, returns
    ``D @ (D.T @ D)^(-1/2)`` via the eigendecomposition of the Gram
    matrix. All columns are treated symmetrically, so (unlike
    Gram-Schmidt) the result does not depend on column order. Eigenvalues
    below a numerical floor are zeroed, so rank-deficient input does not
    raise: duplicated input columns collapse to identical output columns
    (the null-space combination of the output vanishes) while the
    independent directions stay orthonormal.

    Args:
        matrix: ``(hidden_size, n_directions)`` array.

    Returns:
        Orthonormalised ``(hidden_size, n_directions)`` array.

    Raises:
        ValueError: If the input is not 2-D.
    """
    D = np.asarray(matrix, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError("lowdin_orthonormalize expects a 2-D matrix")
    gram = D.T @ D
    eigvals, eigvecs = np.linalg.eigh(gram)
    top = float(np.max(eigvals))
    floor = np.finfo(np.float64).eps * gram.shape[0] * top
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(eigvals > floor, 1.0 / np.sqrt(eigvals), 0.0)
    gram_inv_sqrt = (eigvecs * inv_sqrt) @ eigvecs.T
    return D @ gram_inv_sqrt


def cosine_matrix(directions: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix of the columns of ``directions``."""
    D = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(D, axis=0)
    safe = np.where(norms > 0.0, norms, 1.0)
    N = D / safe
    return N.T @ N


def cosine_stats(directions: dict[str, dict[int, np.ndarray]]) -> dict[str, list]:
    """Per-layer 3x3 cosine matrices of the EPA directions (E, P, A order).

    Used as provenance statistics (raw interference before any
    orthogonalisation, and the residual after Løwdin).
    """
    out: dict[str, list] = {}
    for layer in sorted(directions["epa.evaluation"]):
        mat = np.stack(
            [directions[f"epa.{dim}"][layer] for dim in EPA_DIMENSIONS], axis=1
        )
        cos = cosine_matrix(mat)
        out[str(layer)] = [[float(x) for x in row] for row in cos]
    return out


def reader_directions_to_vectors(
    directions: dict[int, np.ndarray],
    signs: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    """Convert a RepReader's per-layer arrays into signed unit vectors.

    ``directions[layer]`` has shape ``(n_components, hidden_size)`` and
    ``signs[layer]`` shape ``(n_components,)``; only component 0 is used
    (every extraction call here fits a single component). The sign from
    the reader's label-based sign determination is applied *here*, so the
    returned vectors are final (see the module docstring on why they must
    not be re-signed downstream).

    Returns:
        ``{layer: (hidden_size,) float64 unit vector}``.
    """
    out: dict[int, np.ndarray] = {}
    for layer, arr in directions.items():
        vec = np.asarray(arr[0], dtype=np.float64)
        if signs is None:
            out[layer] = l2_normalize(vec)
            continue
        sign = float(np.asarray(signs[layer]).ravel()[0])
        out[layer] = l2_normalize(vec * sign)
    return out


def decoder_layers_to_hf_indices(layers: list[int], num_hidden_layers: int) -> list[int]:
    """Map 0-indexed decoder layer indices to HF hidden-state indices.

    HuggingFace ``output_hidden_states`` puts the embedding output at
    index 0, so decoder layer ``i`` lives at index ``i + 1`` (negative
    index ``-1`` is the last decoder layer).
    """
    out: list[int] = []
    for layer in layers:
        if not 0 <= layer < num_hidden_layers:
            raise ValueError(f"layer {layer} out of range [0, {num_hidden_layers})")
        out.append(layer + 1)
    return out


def parse_layers(spec: str) -> list[int] | None:
    """Parse a comma-separated layer spec; ``"all"`` maps to ``None``."""
    if spec.strip().lower() == "all":
        return None
    values = sorted({int(x) for x in re.split(r"[,\s]+", spec) if x.strip()})
    if not values:
        raise ValueError("empty layer spec")
    return values


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------


@dataclass
class DirectionsBundle:
    """A saved set of extracted directions with provenance metadata.

    Attributes:
        model_name: HuggingFace model identifier the directions came from.
        hidden_size: Embedding dimension of every direction.
        hidden_layers: 0-indexed decoder layer indices carried by the bundle.
        directions: ``{name: {layer: (hidden_size,) float64 unit vector}}``
            with names ``"deflection_contrast"`` and
            ``"epa.evaluation" | "epa.potency" | "epa.activity"``.
        created: UTC creation timestamp (ISO 8601).
        rep_token: Reading position used during extraction.
        seed: Master seed recorded for provenance.
        lowdin: Whether the EPA directions were Løwdin-orthonormalised.
        contrast_set_path: Source contrast-set JSON (deflection direction).
        n_contrast_pairs: Pairs used for the deflection-contrast direction.
        epa_items_per_side: Extreme terms per side per EPA dimension.
        stats: JSON-safe provenance statistics (cosine matrices etc.).
    """

    model_name: str
    hidden_size: int
    hidden_layers: list[int]
    directions: dict[str, dict[int, np.ndarray]]
    created: str = ""
    rep_token: int = DEFAULT_REP_TOKEN
    seed: int = DEFAULT_SEED
    lowdin: bool = False
    contrast_set_path: str = ""
    n_contrast_pairs: int = 0
    epa_items_per_side: int = 0
    stats: dict = field(default_factory=dict)
    epa_direction_method: str = "pca"
    epa_templates: list[str] = field(default_factory=list)
    epa_mix_seed: int | None = None
    epa_raw_saved: bool = False

    @property
    def direction_names(self) -> list[str]:
        """All direction names the bundle must carry.

        The ``epa.{dim}.raw`` entries (pre-orthonormalisation signed unit
        directions, used by the reader) are optional extras, not part of
        the canonical contract.
        """
        return [DEFLECTION_DIRECTION, *(f"epa.{dim}" for dim in EPA_DIMENSIONS)]

    def direction(self, name: str, layer: int) -> np.ndarray:
        """Return one signed unit direction vector."""
        try:
            return self.directions[name][layer]
        except KeyError as exc:
            raise KeyError(f"no direction {name!r} for layer {layer}") from exc

    def epa_matrix(self, layer: int) -> np.ndarray:
        """Return the ``(hidden_size, 3)`` E/P/A direction matrix for a layer."""
        return np.stack(
            [self.direction(f"epa.{dim}", layer) for dim in EPA_DIMENSIONS], axis=1
        )

    def validate(self) -> None:
        """Check that every (name, layer) carries a finite unit vector.

        Raises:
            ValueError: On a missing, misshaped, non-finite, or zero vector.
        """
        names = list(self.direction_names)
        if self.epa_raw_saved:
            names.extend(f"epa.{dim}.raw" for dim in EPA_DIMENSIONS)
        for name in names:
            for layer in self.hidden_layers:
                vec = self.direction(name, layer)
                if vec.shape != (self.hidden_size,):
                    raise ValueError(
                        f"{name} @ layer {layer}: shape {vec.shape} != "
                        f"({self.hidden_size},)"
                    )
                if not np.all(np.isfinite(vec)):
                    raise ValueError(f"{name} @ layer {layer}: non-finite entries")
                if float(np.linalg.norm(vec)) == 0.0:
                    raise ValueError(f"{name} @ layer {layer}: zero vector")

    def meta_dict(self) -> dict:
        """Return the JSON-safe metadata (direction arrays excluded)."""
        return {
            "format": BUNDLE_FORMAT,
            "created": self.created,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "hidden_layers": list(self.hidden_layers),
            "rep_token": self.rep_token,
            "seed": self.seed,
            "lowdin": self.lowdin,
            "contrast_set_path": self.contrast_set_path,
            "n_contrast_pairs": self.n_contrast_pairs,
            "epa_items_per_side": self.epa_items_per_side,
            "stats": self.stats,
            "epa_direction_method": self.epa_direction_method,
            "epa_templates": list(self.epa_templates),
            "epa_mix_seed": self.epa_mix_seed,
            "epa_raw_saved": self.epa_raw_saved,
        }

    def save(self, npz_path: Path, json_path: Path) -> None:
        """Write the directions (``.npz``) and metadata (``.json``)."""
        arrays = {
            f"{name}|{layer}": np.asarray(vec, dtype=np.float32)
            for name, layers in self.directions.items()
            for layer, vec in layers.items()
        }
        np.savez_compressed(Path(npz_path), **arrays)
        Path(json_path).write_text(
            json.dumps(self.meta_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, npz_path: Path, json_path: Path) -> DirectionsBundle:
        """Load and validate a bundle previously written by :meth:`save`."""
        meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if meta.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported directions format {meta.get('format')!r}")
        npz = np.load(npz_path)
        directions: dict[str, dict[int, np.ndarray]] = {}
        for key in npz.files:
            name, _, layer = key.rpartition("|")
            directions.setdefault(name, {})[int(layer)] = npz[key]
        bundle = cls(
            model_name=meta["model_name"],
            hidden_size=int(meta["hidden_size"]),
            hidden_layers=[int(layer) for layer in meta["hidden_layers"]],
            directions=directions,
            created=meta.get("created", ""),
            rep_token=int(meta.get("rep_token", DEFAULT_REP_TOKEN)),
            seed=int(meta.get("seed", DEFAULT_SEED)),
            lowdin=bool(meta.get("lowdin", False)),
            contrast_set_path=meta.get("contrast_set_path", ""),
            n_contrast_pairs=int(meta.get("n_contrast_pairs", 0)),
            epa_items_per_side=int(meta.get("epa_items_per_side", 0)),
            stats=meta.get("stats", {}),
            epa_direction_method=str(meta.get("epa_direction_method", "pca")),
            epa_templates=[str(t) for t in meta.get("epa_templates", [])],
            epa_mix_seed=(
                int(meta["epa_mix_seed"])
                if meta.get("epa_mix_seed") is not None
                else None
            ),
            epa_raw_saved=bool(meta.get("epa_raw_saved", False)),
        )
        bundle.validate()
        return bundle


# ---------------------------------------------------------------------------
# Model-bound extraction (lazy imports; GPU)
# ---------------------------------------------------------------------------


def ensure_pad_token(tokenizer: Any, model: Any = None) -> Any:
    """Ensure the tokenizer (and model config) have a pad token.

    Llama tokenizers ship without a pad token, and RepE batches its
    contrast inputs with ``padding=True`` — the HF pipeline refuses to
    batch a tokenizer whose ``pad_token`` is ``None``. Falls back to the
    EOS token (standard practice for decoder-only LMs).

    Args:
        tokenizer: The HF tokenizer (mutated in place when needed).
        model: Optional HF model; its ``config.pad_token_id`` is set
            from the tokenizer when missing.

    Returns:
        The (possibly mutated) tokenizer.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model_and_pipeline(model_name: str, device: str, dtype: str):
    """Load the generation model plus a ``rep-reading`` pipeline."""
    import torch
    from repe import repe_pipeline_registry
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers import (
        pipeline as hf_pipeline,
    )

    repe_pipeline_registry()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    ensure_pad_token(tokenizer)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )
    ensure_pad_token(tokenizer, model)
    return model, tokenizer, hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)


def extract_deflection_direction(
    rep_pipeline,
    pairs: list[ContrastPair],
    hf_layers: list[int],
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
):
    """Fit the deflection-contrast direction on the contrast pairs."""
    inputs, labels = pairs_to_training_inputs(pairs)
    return rep_pipeline.get_directions(
        inputs,
        rep_token=DEFAULT_REP_TOKEN,
        hidden_layers=hf_layers,
        n_difference=1,
        train_labels=labels,
        direction_method="pca",
        direction_finder_kwargs={"n_components": 1},
        batch_size=batch_size,
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def extract_epa_directions(
    rep_pipeline,
    epa_items: dict[str, tuple[list[str], list[list[int]] | list[int]]],
    hf_layers: list[int],
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    direction_method: str = "pca",
) -> dict[str, object]:
    """Fit one EPA direction per dimension on the extreme-behaviour sets.

    Args:
        rep_pipeline: Upstream RepE ``RepReaderPipeline``.
        epa_items: ``{dimension: (inputs, labels)}`` from
            :func:`build_epa_contrast_items` (label format must match
            ``direction_method``: nested for ``"pca"``, flat for
            ``"cluster_mean"``).
        hf_layers: 1-indexed HF hidden-state indices.
        batch_size: Forward-pass batch size.
        max_length: Tokenisation cap.
        direction_method: ``"pca"`` (principal axis of the difference
            cloud) or ``"cluster_mean"`` (difference of per-ordering mean
            differences — more robust, needs mixed pair orderings).

    Raises:
        ValueError: For an unknown direction method.
    """
    if direction_method not in ("pca", "cluster_mean"):
        raise ValueError(
            f"direction_method must be 'pca' or 'cluster_mean', "
            f"got {direction_method!r}"
        )
    # ClusterMeanRepReader takes no constructor kwargs; PCA wants n_components.
    finder_kwargs = {"n_components": 1} if direction_method == "pca" else {}
    readers: dict[str, object] = {}
    for dim in EPA_DIMENSIONS:
        inputs, labels = epa_items[dim]
        readers[dim] = rep_pipeline.get_directions(
            inputs,
            rep_token=DEFAULT_REP_TOKEN,
            hidden_layers=hf_layers,
            n_difference=1,
            train_labels=labels,
            direction_method=direction_method,
            direction_finder_kwargs=finder_kwargs,
            batch_size=batch_size,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    return readers


def _remap_to_decoder_keys(vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Rekey HF hidden-state indices (i+1) back to 0-indexed decoder layers."""
    return {hf_idx - 1: vec for hf_idx, vec in vectors.items()}


def default_directions_paths(
    model_name: str, out_dir: Path | str = "artifacts"
) -> tuple[Path, Path]:
    """The canonical directions artifact paths for a model name.

    Args:
        model_name: The HF model identifier.
        out_dir: The artifact directory (default ``artifacts``).

    Returns:
        The ``(npz_path, json_path)`` pair.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return (Path(out_dir) / f"directions_{slug}.npz", Path(out_dir) / f"directions_{slug}.json")


def resolve_directions_paths(
    directions: Path | None, model_name: str
) -> tuple[Path, Path]:
    """Resolve the directions artifact paths for a CLI invocation.

    CLIs check the returned ``npz_path`` for existence *before* loading
    the model (fail fast: no point loading an 8B model to discover the
    directions artifact is missing).

    Args:
        directions: An explicit .npz path (CLI ``--directions``), or
            ``None`` for the model's canonical default.
        model_name: The HF model identifier (used for the default).

    Returns:
        The ``(npz_path, json_path)`` pair.
    """
    if directions is not None:
        return directions, directions.with_suffix(".json")
    return default_directions_paths(model_name)


def extract_all(
    model_name: str,
    contrast_set_path: Path | str,
    layers: list[int] | None,
    out_dir: Path | str,
    device: str = "cuda",
    dtype: str = "float16",
    n_epa_per_side: int = DEFAULT_EPA_ITEMS_PER_SIDE,
    seed: int = DEFAULT_SEED,
    lowdin: bool = False,
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_pairs: int | None = None,
    direction_method: str = "pca",
    epa_templates: Sequence[str] | None = None,
    mix_seed: int | None = None,
) -> tuple[Path, Path]:
    """Run the full extraction and save the directions artifact.

    Args:
        model_name: HuggingFace model identifier.
        contrast_set_path: Contrast-set JSON written by the stimuli CLI.
        layers: 0-indexed decoder layer indices; ``None`` for all layers.
        out_dir: Directory for the ``directions_<model>.npz/.json`` pair.
        device: Torch device string.
        dtype: Torch dtype name.
        n_epa_per_side: Extreme behaviour terms per side per dimension.
        seed: Master seed (recorded in the metadata).
        lowdin: Løwdin-orthonormalise the three EPA directions per layer.
        batch_size: Forward-pass batch size.
        max_length: Tokenisation cap.
        max_pairs: Use at most this many contrast pairs (``--quick``).
        direction_method: ``"pca"`` (default) or ``"cluster_mean"`` for
            EPA direction fitting.
        epa_templates: Optional list of text templates (each with a
            ``{behaviour}`` placeholder). With more than one template the
            EPA fit is run per template and the signed unit directions
            are averaged per layer (multi-template lever).
        mix_seed: Seed for mixing contrast pair orderings (required for
            a non-degenerate ``cluster_mean`` fit; auto-derived from
            ``seed`` when ``direction_method == "cluster_mean"``).

    Returns:
        ``(npz_path, json_path)`` of the saved bundle.
    """
    paths = DataPaths()
    contrast = json.loads(Path(contrast_set_path).read_text(encoding="utf-8"))
    pairs = dicts_to_pairs(contrast["pairs"])
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    dictionary = ACTDictionary.load(paths.data_dir)
    label_format = "flat" if direction_method == "cluster_mean" else "nested"
    effective_mix_seed = (
        mix_seed
        if mix_seed is not None
        else (seed if direction_method == "cluster_mean" else None)
    )

    model, _, rep_pipeline = load_model_and_pipeline(model_name, device, dtype)
    num_layers = int(model.config.num_hidden_layers)
    decoder_layers = (
        list(range(num_layers)) if layers is None else sorted(set(layers))
    )
    hf_layers = decoder_layers_to_hf_indices(decoder_layers, num_layers)

    defl_reader = extract_deflection_direction(
        rep_pipeline, pairs, hf_layers, batch_size, max_length
    )

    def _epa_vectors(readers: dict[str, object]) -> dict[str, dict[int, np.ndarray]]:
        return {
            dim: _remap_to_decoder_keys(
                reader_directions_to_vectors(
                    readers[dim].directions, readers[dim].direction_signs
                )
            )
            for dim in EPA_DIMENSIONS
        }

    templates = list(epa_templates) if epa_templates else None
    if templates and len(templates) > 1:
        per_template: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for t_idx, tpl in enumerate(templates):
            items_t = build_epa_contrast_items(
                dictionary,
                n_epa_per_side,
                template=tpl,
                label_format=label_format,
                mix_seed=(
                    effective_mix_seed + t_idx
                    if effective_mix_seed is not None
                    else None
                ),
            )
            readers_t = extract_epa_directions(
                rep_pipeline,
                items_t,
                hf_layers,
                batch_size,
                max_length,
                direction_method,
            )
            per_template[tpl] = _epa_vectors(readers_t)
        epa_vectors: dict[str, dict[int, np.ndarray]] = {
            dim: average_signed_directions(
                {tpl: per_template[tpl][dim] for tpl in templates}
            )
            for dim in EPA_DIMENSIONS
        }
    else:
        tpl = templates[0] if templates else RESPONSE_TEMPLATE
        epa_items = build_epa_contrast_items(
            dictionary,
            n_epa_per_side,
            template=tpl,
            label_format=label_format,
            mix_seed=effective_mix_seed,
        )
        epa_readers = extract_epa_directions(
            rep_pipeline, epa_items, hf_layers, batch_size, max_length,
            direction_method,
        )
        epa_vectors = _epa_vectors(epa_readers)

    directions: dict[str, dict[int, np.ndarray]] = {
        DEFLECTION_DIRECTION: _remap_to_decoder_keys(
            reader_directions_to_vectors(defl_reader.directions, defl_reader.direction_signs)
        )
    }
    for dim in EPA_DIMENSIONS:
        directions[f"epa.{dim}"] = epa_vectors[dim]
        # Pre-orthonormalisation signed unit directions for the reader
        # (orthonormalisation is a steering-side requirement; raw unit
        # directions are less distorted for measurement).
        directions[f"epa.{dim}.raw"] = {
            layer: vec.copy() for layer, vec in epa_vectors[dim].items()
        }

    stats = {"epa_cosines_before": cosine_stats(directions)}
    if lowdin:
        for layer in sorted(directions["epa.evaluation"]):
            mat = np.stack(
                [directions[f"epa.{dim}"][layer] for dim in EPA_DIMENSIONS], axis=1
            )
            orth = lowdin_orthonormalize(mat)
            for j, dim in enumerate(EPA_DIMENSIONS):
                directions[f"epa.{dim}"][layer] = orth[:, j]
        stats["epa_cosines_after"] = cosine_stats(directions)

    bundle = DirectionsBundle(
        model_name=model_name,
        hidden_size=int(model.config.hidden_size),
        hidden_layers=decoder_layers,
        directions=directions,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        lowdin=lowdin,
        contrast_set_path=str(contrast_set_path),
        n_contrast_pairs=len(pairs),
        epa_items_per_side=n_epa_per_side,
        stats=stats,
        epa_direction_method=direction_method,
        epa_templates=templates if templates else [RESPONSE_TEMPLATE],
        epa_mix_seed=effective_mix_seed,
        epa_raw_saved=True,
    )
    bundle.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = default_directions_paths(model_name, out_dir)
    bundle.save(npz_path, json_path)
    return npz_path, json_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: extract the directions and save the artifact."""
    parser = argparse.ArgumentParser(
        description="Extract ACT steering/reading directions from a model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--dtype", default="float16", help="torch dtype name")
    parser.add_argument(
        "--layers",
        default="all",
        help="comma-separated 0-indexed decoder layers, or 'all'",
    )
    parser.add_argument(
        "--contrast-set",
        default=None,
        help="contrast-set JSON (default: artifacts/contrast_set.json)",
    )
    parser.add_argument(
        "--n-epa-per-side",
        type=int,
        default=DEFAULT_EPA_ITEMS_PER_SIDE,
        help="extreme behaviour terms per side per EPA dimension",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument(
        "--lowdin", action="store_true", help="Løwdin-orthonormalise the EPA set"
    )
    parser.add_argument(
        "--direction-method",
        choices=("pca", "cluster_mean"),
        default="pca",
        help="EPA direction fitting method (default: pca)",
    )
    parser.add_argument(
        "--epa-templates",
        default=None,
        help=(
            "comma-separated text templates with a {behaviour} placeholder "
            "for multi-template EPA fitting (default: single built-in template)"
        ),
    )
    parser.add_argument(
        "--mix-seed",
        type=int,
        default=None,
        help=(
            "seed for mixing contrast pair orderings "
            "(auto-derived from --seed for cluster_mean)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="token cap"
    )
    parser.add_argument(
        "--quick", action="store_true", help="smoke mode: 20 pairs, 10/side, 3 layers"
    )
    args = parser.parse_args(argv)

    layers = parse_layers(args.layers)
    n_epa = args.n_epa_per_side
    batch_size = args.batch_size
    max_pairs: int | None = None
    if args.quick:
        if layers is not None:
            _LOGGER.warning("--quick overrides --layers with 0,16,31")
        layers = [0, 16, 31]
        n_epa = 10
        batch_size = 4
        max_pairs = 20

    paths = DataPaths()
    paths.ensure_dirs()
    contrast_set = (
        Path(args.contrast_set)
        if args.contrast_set
        else paths.artifacts_dir / "contrast_set.json"
    )
    epa_templates = (
        [t.strip() for t in args.epa_templates.split(",") if t.strip()]
        if args.epa_templates
        else None
    )
    npz_path, json_path = extract_all(
        args.model,
        contrast_set,
        layers,
        paths.artifacts_dir,
        device=args.device,
        dtype=args.dtype,
        n_epa_per_side=n_epa,
        seed=args.seed,
        lowdin=args.lowdin,
        batch_size=batch_size,
        max_length=args.max_length,
        max_pairs=max_pairs,
        direction_method=args.direction_method,
        epa_templates=epa_templates,
        mix_seed=args.mix_seed,
    )
    print(f"wrote {npz_path} ({json_path.name})")


if __name__ == "__main__":
    main()
