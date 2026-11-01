"""MTurk-Interact EPA dictionary loading and lookup.

Loads the human-rated MTurk-Interact dictionaries (identities, behaviours,
modifiers) shipped in ``data/act`` into typed in-memory lookups used by the
stimulus builder and the ACT core.

The CSV files carry duplicated ``E2/P2/A2`` columns (identical to ``E/P/A`` in
the current release); they are ignored.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import BEHAVIORS_FILE, IDENTITIES_FILE, MODIFIERS_FILE

__all__ = [
    "EPATerm",
    "ACTDictionary",
    "load_epa_terms",
    "normalize_term",
]

_LOGGER = logging.getLogger(__name__)

#: Collections that can be sampled, mapped to the matching instance attribute.
_COLLECTIONS = ("identities", "behaviors", "modifiers")


@dataclass(frozen=True)
class EPATerm:
    """A single dictionary entry with human-rated EPA values.

    Attributes:
        term: Canonical (normalized) dictionary term, e.g. ``"27_year_old"``.
        e: Evaluation value (observed range -3.9 to +3.6).
        p: Potency value (observed range -3.4 to +3.5).
        a: Activity value (observed range -3.2 to +3.1).
    """

    term: str
    e: float
    p: float
    a: float

    def as_vector(self) -> tuple[float, float, float]:
        """Return the EPA as a plain ``(e, p, a)`` tuple."""
        return (self.e, self.p, self.a)


def normalize_term(term: str) -> str:
    """Normalize a dictionary term for lookup.

    Lowercases, trims whitespace, maps spaces/hyphens to underscores, and
    collapses duplicate underscores, matching the CSV term style
    (``"27_year_old"``).

    Args:
        term: Raw term in any reasonable casing or separator style.

    Returns:
        The normalized term.
    """
    normalized = term.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def load_epa_terms(path: Path | str) -> list[EPATerm]:
    """Load EPA terms from one MTurk-Interact CSV file.

    Args:
        path: Path to a CSV with columns ``term,E,P,A``. The duplicated
            ``E2/P2/A2`` columns, if present, are ignored.

    Returns:
        The parsed terms in file order.

    Malformed rows (missing term or non-numeric EPA values) are skipped with
    a logged warning.
    """
    terms: list[EPATerm] = []
    skipped = 0
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_term = (row.get("term") or "").strip()
            if not raw_term:
                skipped += 1
                continue
            try:
                e = float(row["E"])
                p = float(row["P"])
                a = float(row["A"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            terms.append(EPATerm(normalize_term(raw_term), e, p, a))
    if skipped:
        _LOGGER.warning("%s: skipped %d malformed rows", path.name, skipped)
    return terms


class ACTDictionary:
    """In-memory MTurk-Interact EPA dictionaries.

    Holds the identity, behaviour, and modifier collections with O(1)
    normalized lookup and seeded sampling. If a normalized term occurs in
    more than one collection, the first collection in the order
    identities -> behaviours -> modifiers wins the shared index (per-collection
    lists always retain all entries).

    Attributes:
        identities: Identity terms (actor/object personas).
        behaviors: Behaviour terms (events).
        modifiers: Modifier terms (adjectival qualifiers).
    """

    def __init__(
        self,
        identities: Sequence[EPATerm],
        behaviors: Sequence[EPATerm],
        modifiers: Sequence[EPATerm],
    ) -> None:
        self.identities: list[EPATerm] = list(identities)
        self.behaviors: list[EPATerm] = list(behaviors)
        self.modifiers: list[EPATerm] = list(modifiers)
        self._index: dict[str, EPATerm] = {}
        for term in self.identities + self.behaviors + self.modifiers:
            self._index.setdefault(term.term, term)

    @classmethod
    def load(cls, data_dir: Path | str) -> ACTDictionary:
        """Load all three dictionaries from an ACT data directory.

        Args:
            data_dir: Directory containing the three MTurk-Interact CSVs
                (``MTurkInteract_{Identities,Behaviors,Modifiers}.csv``).

        Returns:
            The loaded dictionary.
        """
        data_dir = Path(data_dir)
        return cls(
            load_epa_terms(data_dir / IDENTITIES_FILE),
            load_epa_terms(data_dir / BEHAVIORS_FILE),
            load_epa_terms(data_dir / MODIFIERS_FILE),
        )

    @property
    def counts(self) -> dict[str, int]:
        """Return the size of each collection."""
        return {
            "identities": len(self.identities),
            "behaviors": len(self.behaviors),
            "modifiers": len(self.modifiers),
        }

    def lookup(self, term: str) -> EPATerm | None:
        """Return the term's EPA entry, or ``None`` if absent.

        Matching is normalization-insensitive (case, spaces, hyphens).
        """
        return self._index.get(normalize_term(term))

    def require(self, term: str) -> EPATerm:
        """Return the term's EPA entry or raise ``KeyError`` if absent."""
        entry = self.lookup(term)
        if entry is None:
            raise KeyError(f"term {term!r} not found in ACT dictionary")
        return entry

    def sample(
        self,
        n: int,
        seed: int,
        collection: str = "behaviors",
        exclude: Sequence[str] = (),
    ) -> list[EPATerm]:
        """Draw ``n`` terms without replacement from a collection.

        Sampling is reproducible for a fixed seed and file order.

        Args:
            n: Number of terms to draw.
            seed: RNG seed.
            collection: One of ``"identities"``, ``"behaviors"``,
                ``"modifiers"``.
            exclude: Terms (any normalization) removed from the pool.

        Returns:
            The sampled terms.

        Raises:
            ValueError: If ``n`` exceeds the pool size or the collection name
                is unknown.
        """
        if collection not in _COLLECTIONS:
            raise ValueError(
                f"unknown collection {collection!r}; expected one of {list(_COLLECTIONS)}"
            )
        pool: list[EPATerm] = getattr(self, collection)
        excluded = {normalize_term(t) for t in exclude}
        candidates = [t for t in pool if t.term not in excluded]
        if n > len(candidates):
            raise ValueError(f"cannot sample {n} terms from a pool of {len(candidates)}")
        rng = random.Random(seed)
        return rng.sample(candidates, n)

    def __repr__(self) -> str:
        return (
            f"ACTDictionary(identities={len(self.identities)}, "
            f"behaviors={len(self.behaviors)}, modifiers={len(self.modifiers)})"
        )
