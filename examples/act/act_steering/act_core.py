"""ACT impression formation and deflection computations (torch-free).

Implements the two-event Affect Control Theory model used by the steering
pipeline, driven by the 15-term impression-formation coefficient set in
``data/act/2010impressionformation.csv``.

Term encoding
-------------
Each coefficient row carries a 9-bit code ``Zxxxxxxxxx``. The 9 bits select
basis components of the input vector

    v = [sE, sP, sA, bE, bP, bA, oE, oP, oA]

where ``s``, ``b``, ``o`` are the pre-event EPA vectors of the actor, the
behaviour, and the object. A row's term is the product of the selected
components (the all-zero code is the constant term). With ``M`` the 15x9
coefficient matrix (columns ``postAE..postOA``; rows are per-term
coefficient vectors), the post-event impression vector is

    W = term_vector(s, b, o) @ M

with 9 slots: actor E/P/A, behaviour E/P/A, object E/P/A. The event
deflection is

    U(s, b, o) = || W - [s; b; o] ||^2 .

The shipped coefficient set is at most linear in each behaviour component,
so ``W`` is affine in ``b`` and the optimal response behaviour has a
closed-form least-squares solution, refined to the box ``[-4.3, 4.3]^3``
with L-BFGS-B when the interior solution falls outside the box.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "EPA",
    "ACTCoefficients",
    "optimal_response_behaviour",
    "optimal_response",
    "two_event_state_update",
]

_LOGGER = logging.getLogger(__name__)

#: Column order of the coefficient CSV (post-event slots).
_OUTPUT_SLOTS = (
    "postAE",
    "postAP",
    "postAA",
    "postBE",
    "postBP",
    "postBA",
    "postOE",
    "postOP",
    "postOA",
)

#: Default box for behaviour EPA coordinates (ACT dictionary range).
DEFAULT_BOUNDS: tuple[float, float] = (-4.3, 4.3)


@dataclass(frozen=True)
class EPA:
    """An Evaluation/Potency/Activity vector (ACT's three dimensions).

    Attributes:
        e: Evaluation (good/bad axis).
        p: Potency (power/weakness axis).
        a: Activity (fast/slow axis).
    """

    e: float
    p: float
    a: float

    def as_array(self) -> np.ndarray:
        """Return the EPA as a float64 numpy array ``[e, p, a]``."""
        return np.array([self.e, self.p, self.a], dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> EPA:
        """Build an EPA from a 3-element array-like.

        Raises:
            ValueError: If the input is not exactly 3 elements.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"expected a 3-element array, got shape {arr.shape}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def distance_to(self, other: EPA) -> float:
        """Return the Euclidean distance to another EPA."""
        return float(np.linalg.norm(self.as_array() - other.as_array()))


def _to_array(value: EPA | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Coerce an EPA-like value (EPA, 3-tuple, or array) to a (3,) array."""
    if isinstance(value, EPA):
        return value.as_array()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3-component EPA, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class ACTCoefficients:
    """The 15-term impression-formation coefficient set.

    Attributes:
        matrix: Coefficient matrix of shape (15, 9); row ``i`` is term
            ``i`` in file order, columns follow ``_OUTPUT_SLOTS``.
        masks: 9-bit selection mask per term; bit ``j`` selects basis
            component ``j`` of ``[sE, sP, sA, bE, bP, bA, oE, oP, oA]``.
            Mask 0 is the constant term.
        linear_in_behavior: True when no term contains two behaviour
            components (required for the closed-form optimal behaviour).
    """

    matrix: np.ndarray
    masks: np.ndarray
    linear_in_behavior: bool = True

    @classmethod
    def load(cls, path: Path | str) -> ACTCoefficients:
        """Load the impression-formation coefficient set from CSV.

        Args:
            path: Path to the coefficient CSV
                (``2010impressionformation.csv``).

        Returns:
            The parsed coefficients.

        Raises:
            ValueError: If the header or a term code is malformed.
        """
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [name for name in ("coef_name", *_OUTPUT_SLOTS) if name not in fieldnames]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            rows = list(reader)
        matrix = np.empty((len(rows), 9), dtype=np.float64)
        masks = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            code = row["coef_name"].strip()
            bits = code[1:]
            if len(bits) != 9 or not all(ch in "01" for ch in bits):
                raise ValueError(f"{path}: malformed term code {code!r}")
            masks[i] = int(bits, 2)
            for j, name in enumerate(_OUTPUT_SLOTS):
                matrix[i, j] = float(row[name])
        behavior_bits = 0b000111000  # bits 3, 4, 5 = bE, bP, bA
        linear_in_behavior = all(bin(int(m) & behavior_bits).count("1") <= 1 for m in masks)
        if not linear_in_behavior:
            _LOGGER.warning(
                "%s contains products of behaviour components; the closed-form "
                "optimal behaviour is unavailable",
                path.name,
            )
        return cls(matrix=matrix, masks=masks, linear_in_behavior=linear_in_behavior)

    def term_values(self, actor, behavior, object_) -> np.ndarray:
        """Evaluate all 15 terms for the given pre-event EPAs.

        Args:
            actor: Pre-event actor EPA (EPA, tuple, or (3,) array).
            behavior: Pre-event behaviour EPA (EPA, tuple, or (3,) array).
            object_: Pre-event object EPA (EPA, tuple, or (3,) array).

        Returns:
            The 15 term values in file order.
        """
        basis = np.concatenate([_to_array(actor), _to_array(behavior), _to_array(object_)])
        terms = np.ones(len(self.masks), dtype=np.float64)
        for bit in range(9):
            selected = (self.masks >> bit) & 1
            terms[selected.astype(bool)] *= basis[bit]
        return terms

    def impressions(
        self, actor: np.ndarray, behavior: np.ndarray, object_: np.ndarray
    ) -> np.ndarray:
        """Compute the post-event impression vector W (9 slots).

        Slot order: actor E/P/A, behaviour E/P/A, object E/P/A.
        """
        return self.term_values(actor, behavior, object_) @ self.matrix

    def deflection(
        self, actor: np.ndarray, behavior: np.ndarray, object_: np.ndarray
    ) -> float:
        """Compute the event deflection U = ||W - [s; b; o]||^2.

        The behaviour is referenced against its own pre-event EPA (the
        two-event formulation).
        """
        w = self.impressions(actor, behavior, object_)
        f = np.concatenate([_to_array(actor), _to_array(behavior), _to_array(object_)])
        return float(np.sum((w - f) ** 2))

    def behavior_affine(
        self, actor: np.ndarray, object_: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose W = c + J b for fixed actor and object fundamentals.

        Args:
            actor: Actor EPA (3,).
            object_: Object EPA (3,).

        Returns:
            A tuple ``(c, J)`` with the constant part ``c`` (9,) and the
            9x3 Jacobian ``J`` with columns for bE, bP, bA.

        Raises:
            ValueError: If the coefficient set is not at most linear in
                each behaviour component.
        """
        if not self.linear_in_behavior:
            raise ValueError(
                "coefficient set contains products of behaviour components; "
                "closed-form optimal behaviour unavailable"
            )
        s, o = _to_array(actor), _to_array(object_)
        c = np.zeros(9, dtype=np.float64)
        jacobian = np.zeros((9, 3), dtype=np.float64)
        for i, mask in enumerate(self.masks):
            m = int(mask)
            bits = [b for b in range(9) if (m >> b) & 1]
            behavior_bits = [b for b in bits if 3 <= b <= 5]
            factor = 1.0
            for b in bits:
                if not 3 <= b <= 5:
                    factor *= (s if b < 3 else o)[b % 3]
            if not behavior_bits:
                c += self.matrix[i] * factor
            else:
                jacobian[:, behavior_bits[0] - 3] += self.matrix[i] * factor
        return c, jacobian


    def deflection_batch(
        self,
        actor: np.ndarray,
        object_: np.ndarray,
        behaviours: np.ndarray,
    ) -> np.ndarray:
        """Compute deflection U for a batch of candidate behaviours.

        Vectorized over the behaviour batch with fixed actor/object
        fundamentals.

        Args:
            actor: Actor EPA (3,).
            object_: Object EPA (3,).
            behaviours: Candidate behaviour EPAs (K, 3).

        Returns:
            Deflection values (K,).

        Raises:
            ValueError: If the coefficient set is not at most linear in
                each behaviour component.
        """
        if not self.linear_in_behavior:
            raise ValueError(
                "coefficient set contains products of behaviour components; "
                "batched deflection unavailable"
            )
        actor = _to_array(actor)
        object_ = _to_array(object_)
        behaviours = np.asarray(behaviours, dtype=np.float64)
        k = len(behaviours)
        factors = np.ones(len(self.masks), dtype=np.float64)
        behavior_bit = np.full(len(self.masks), -1, dtype=np.int64)
        for i, mask in enumerate(self.masks):
            m = int(mask)
            factor = 1.0
            bbit = -1
            for bit in range(9):
                if not (m >> bit) & 1:
                    continue
                if 3 <= bit <= 5:
                    bbit = bit - 3
                else:
                    factor *= (actor if bit < 3 else object_)[bit % 3]
            factors[i] = factor
            behavior_bit[i] = bbit
        terms = factors[None, :].repeat(k, axis=0)  # (K, 15)
        for bit in range(3):
            selected = behavior_bit == bit
            if selected.any():
                terms[:, selected] *= behaviours[:, bit][:, None]
        w = terms @ self.matrix  # (K, 9)
        reference = np.concatenate(
            [np.tile(actor, (k, 1)), behaviours, np.tile(object_, (k, 1))], axis=1
        )
        return np.sum((w - reference) ** 2, axis=1)


def optimal_response_behaviour(
    coeffs: ACTCoefficients,
    actor: EPA,
    object_: EPA,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> EPA:
    """Compute the behaviour EPA b* that minimizes the event deflection.

    Solves ``min_b ||A b - d||^2`` with ``A = J - I_mid`` and
    ``d = [s; 0; o] - c`` (the behaviour referenced against itself). The
    unconstrained solution is a closed-form least-squares solution; when it
    falls outside the box, L-BFGS-B refines it to the box-constrained
    optimum.

    Args:
        coeffs: The coefficient set.
        actor: The actor's current self-fundamental.
        object_: The object's current self-fundamental.
        bounds: The box applied to each behaviour EPA coordinate.

    Returns:
        The optimal behaviour EPA.
    """
    s, o = actor.as_array(), object_.as_array()
    c, jacobian = coeffs.behavior_affine(s, o)
    a_matrix = jacobian.copy()
    a_matrix[3:6, :] -= np.eye(3)
    d = np.concatenate([s, np.zeros(3), o]) - c
    b_unconstrained = np.linalg.lstsq(a_matrix, d, rcond=None)[0]
    lo, hi = bounds
    if lo <= float(b_unconstrained.min()) and float(b_unconstrained.max()) <= hi:
        return EPA.from_array(b_unconstrained)

    def objective(b: np.ndarray) -> float:
        residual = a_matrix @ b - d
        return float(residual @ residual)

    result = minimize(
        objective,
        x0=np.clip(b_unconstrained, lo, hi),
        method="L-BFGS-B",
        bounds=[(lo, hi)] * 3,
        tol=1e-14,
    )
    return EPA.from_array(np.clip(result.x, lo, hi))


def two_event_state_update(
    coeffs: ACTCoefficients,
    user: EPA,
    agent: EPA,
    user_behaviour: EPA,
) -> tuple[EPA, EPA, float]:
    """Process event 1: the user acts on the agent with a behaviour.

    Args:
        coeffs: The coefficient set.
        user: The user's current self-fundamental.
        agent: The agent's current self-fundamental.
        user_behaviour: The EPA of the user's message/behaviour.

    Returns:
        A tuple ``(user_after, agent_after, deflection)``: the post-event
        self-fundamentals of user and agent, and the event-1 deflection.
    """
    s = user.as_array()
    b = user_behaviour.as_array()
    o = agent.as_array()
    w = coeffs.impressions(s, b, o)
    u1 = float(np.sum((w - np.concatenate([s, b, o])) ** 2))
    return EPA.from_array(w[0:3]), EPA.from_array(w[6:9]), u1


def optimal_response(
    coeffs: ACTCoefficients,
    user: EPA,
    agent: EPA,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> EPA:
    """Compute the event-2 target b*: the agent's optimal response behaviour.

    Convenience wrapper around :func:`optimal_response_behaviour` with the
    conversation roles (actor = agent, object = user).
    """
    return optimal_response_behaviour(coeffs, actor=agent, object_=user, bounds=bounds)
