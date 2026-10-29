"""
ACT Core Module - Affect Control Theory Functions

Implements core ACT functions based on David R. Heise's Expressive Order (2007):
- Impression formation using coefficient matrices
- Deflection calculation
- Optimal behavior finding via numerical optimization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
from pathlib import Path


@dataclass
class EPA:
    """EPA (Evaluation, Potency, Activity) profile."""
    e: float  # Evaluation: good vs bad
    p: float  # Potency: powerful vs weak  
    a: float  # Activity: active vs passive
    
    def to_array(self) -> np.ndarray:
        return np.array([self.e, self.p, self.a])
    
    def to_dict(self) -> Dict[str, float]:
        return {'e': self.e, 'p': self.p, 'a': self.a}
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'EPA':
        return cls(e=float(arr[0]), p=float(arr[1]), a=float(arr[2]))
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'EPA':
        return cls(e=d['e'], p=d['p'], a=d['a'])
    
    def __repr__(self) -> str:
        return f"EPA(e={self.e:.2f}, p={self.p:.2f}, a={self.a:.2f})"


class ACTCoefficients:
    """
    Load and manage ACT impression formation coefficients.
    
    The coefficient matrix encodes how Actor, Behavior, and Object EPAs
    combine to produce post-event transient impressions.
    """
    
    def __init__(self, coef_path: Optional[str] = None):
        """
        Load coefficients from CSV file.
        
        Args:
            coef_path: Path to coefficient CSV (default: data/act/2010impressionformation.csv)
        """
        if coef_path is None:
            # Default path relative to this file
            coef_path = Path(__file__).parent.parent.parent / "data" / "act" / "2010impressionformation.csv"
        
        self.coef_df = pd.read_csv(coef_path)
        self._build_coefficient_lookup()
    
    def _build_coefficient_lookup(self):
        """Build lookup dictionary for fast coefficient access."""
        self.coefficients = {}
        for _, row in self.coef_df.iterrows():
            coef_name = row['coef_name']
            self.coefficients[coef_name] = {
                col: row[col] for col in self.coef_df.columns if col != 'coef_name'
            }
    
    def get_coefficient(self, coef_name: str, output_dim: str) -> float:
        """Get a specific coefficient value."""
        return self.coefficients.get(coef_name, {}).get(output_dim, 0.0)


def impression_formation(
    actor: EPA,
    behavior: EPA, 
    obj: EPA,
    coefficients: ACTCoefficients
) -> Dict[str, EPA]:
    """
    Calculate post-event transient impressions using ACT impression formation equations.
    
    Based on Heise's coefficient matrix approach where the input vector is:
    [Ae, Ap, Aa, Be, Bp, Ba, Oe, Op, Oa]
    
    The coefficient names use binary encoding (Z followed by 9 bits) where:
    - Position 1-3: Actor E, P, A
    - Position 4-6: Behavior E, P, A  
    - Position 7-9: Object E, P, A
    - '1' means that dimension is included in the term
    
    Args:
        actor: Actor's transient EPA (A)
        behavior: Behavior EPA (B)
        obj: Object's transient EPA (O)
        coefficients: ACTCoefficients instance
        
    Returns:
        Dict with 'actor', 'behavior', 'object' keys mapping to post-event EPAs
    """
    # Build input vector [Ae, Ap, Aa, Be, Bp, Ba, Oe, Op, Oa]
    input_vector = np.array([
        actor.e, actor.p, actor.a,
        behavior.e, behavior.p, behavior.a,
        obj.e, obj.p, obj.a
    ])
    
    def compute_term_value(coef_name: str) -> float:
        """
        Calculate the value of a coefficient term based on binary encoding.
        
        For example:
        - Z000000000 = 1 (constant term)
        - Z100000000 = Ae
        - Z100100000 = Ae * Be
        """
        term = 1.0
        for i, ch in enumerate(coef_name[1:]):  # Skip initial 'Z'
            if ch == '1':
                term *= input_vector[i]
        return term
    
    # Output dimensions
    output_dims = ['postAE', 'postAP', 'postAA', 'postBE', 'postBP', 'postBA', 
                   'postOE', 'postOP', 'postOA']
    
    result = {dim: 0.0 for dim in output_dims}
    
    # Sum all coefficient terms for each output dimension
    for coef_name in coefficients.coefficients:
        term_val = compute_term_value(coef_name)
        for dim in output_dims:
            coef_val = coefficients.get_coefficient(coef_name, dim)
            result[dim] += term_val * coef_val
    
    return {
        'actor': EPA(e=result['postAE'], p=result['postAP'], a=result['postAA']),
        'behavior': EPA(e=result['postBE'], p=result['postBP'], a=result['postBA']),
        'object': EPA(e=result['postOE'], p=result['postOP'], a=result['postOA'])
    }


def calculate_deflection(fundamental: EPA, transient: EPA) -> float:
    """
    Calculate deflection as squared Euclidean distance between fundamental and transient EPA.
    
    Deflection measures the "stress" in the interaction when transient impressions
    deviate from fundamental (culturally-expected) sentiments.
    
    Args:
        fundamental: The fundamental (expected) EPA profile
        transient: The transient (post-event) EPA profile
        
    Returns:
        Deflection value (sum of squared differences)
    """
    diff = fundamental.to_array() - transient.to_array()
    return float(np.sum(diff ** 2))


def total_deflection(
    actor_fundamental: EPA,
    actor_transient: EPA,
    behavior_fundamental: Optional[EPA],
    behavior_transient: EPA,
    object_fundamental: EPA,
    object_transient: EPA
) -> float:
    """
    Calculate total system deflection across Actor, Behavior, and Object.
    
    For behavior, if no fundamental is provided, uses the behavior_transient
    as its own fundamental (i.e., behavior deflection = 0).
    
    Args:
        actor_fundamental: Actor's fundamental EPA
        actor_transient: Actor's post-event transient EPA
        behavior_fundamental: Behavior's fundamental EPA (optional, often equals transient)
        behavior_transient: Behavior's post-event transient EPA
        object_fundamental: Object's fundamental EPA
        object_transient: Object's post-event transient EPA
        
    Returns:
        Total deflection value
    """
    actor_defl = calculate_deflection(actor_fundamental, actor_transient)
    
    if behavior_fundamental is not None:
        behavior_defl = calculate_deflection(behavior_fundamental, behavior_transient)
    else:
        behavior_defl = 0.0
        
    object_defl = calculate_deflection(object_fundamental, object_transient)
    
    return actor_defl + behavior_defl + object_defl


def find_optimal_behavior(
    actor_fundamental: EPA,
    object_fundamental: EPA,
    actor_transient: EPA,
    object_transient: EPA,
    coefficients: ACTCoefficients,
    bounds: Tuple[float, float] = (-4.3, 4.3),
    include_behavior_deflection: bool = True
) -> EPA:
    """
    Find optimal behavior EPA that minimizes total system deflection.
    
    Uses numerical optimization to find the behavior EPA values that,
    when applied to the current transient impressions, produce post-event
    impressions closest to the fundamental sentiments.
    
    Args:
        actor_fundamental: Actor's fundamental EPA
        object_fundamental: Object's fundamental EPA
        actor_transient: Actor's current transient EPA
        object_transient: Object's current transient EPA
        coefficients: ACTCoefficients instance
        bounds: Bounds for behavior EPA values (default: -4.3 to 4.3)
        include_behavior_deflection: If True, minimize behavior's transient deviation too
        
    Returns:
        Optimal behavior EPA
    """
    
    def objective(b_vec: np.ndarray) -> float:
        """Objective function: total deflection after the behavior."""
        behavior = EPA.from_array(b_vec)
        
        # Calculate post-event impressions
        post = impression_formation(actor_transient, behavior, object_transient, coefficients)
        
        # Calculate deflections from fundamentals
        actor_defl = calculate_deflection(actor_fundamental, post['actor'])
        object_defl = calculate_deflection(object_fundamental, post['object'])
        
        if include_behavior_deflection:
            # Behavior deflection: how much the behavior changes from its initial value
            behavior_defl = calculate_deflection(behavior, post['behavior'])
        else:
            behavior_defl = 0.0
        
        return actor_defl + behavior_defl + object_defl
    
    # Start optimization from neutral behavior
    x0 = np.zeros(3)
    
    # Set bounds for each dimension
    opt_bounds = [(bounds[0], bounds[1]) for _ in range(3)]
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        bounds=opt_bounds,
        method='L-BFGS-B'
    )
    
    return EPA.from_array(result.x)


def predict_emotion(
    identity_fundamental: EPA,
    identity_transient: EPA,
    is_actor: bool = True
) -> EPA:
    """
    Predict emotional response based on deflection between fundamental and transient EPA.
    
    Per ACT, emotions arise from the discrepancy between what we expect to feel
    (fundamental) and what we actually experience (transient) in a situation.
    
    This is a simplified emotion prediction. Full ACT uses additional coefficients.
    
    Args:
        identity_fundamental: Fundamental EPA for the identity
        identity_transient: Transient EPA after the event
        is_actor: Whether this is the actor (True) or object (False)
        
    Returns:
        Predicted emotion EPA (approximation)
    """
    # Simplified: emotion is proportional to transient impression
    # Full ACT would use emotion-specific coefficients
    return identity_transient


# =========================================================================
# Vectorised impression formation and closed-form optimal behaviour
# =========================================================================
#
# Notation (matches the paper, "Two-event ACT formulation"):
#   x = [A; B; O] in R^9           actor, behaviour, object EPA
#   tau = M g(x) in R^9            post-event transients, g = term vector
#   D(b) = ||f_A - tau_A(b)||^2 + ||b - tau_B(b)||^2 + ||f_O - tau_O(b)||^2
#
# The 2010 U.S. equation set contains no products between two behaviour
# components (every term has at most one B bit set), so for fixed A and O
# tau(b) = c + J b is affine in b and D(b) is quadratic.  The minimiser is
# then an (optionally bounded) linear least-squares problem.

_OUTPUT_ORDER = ['postAE', 'postAP', 'postAA', 'postBE', 'postBP', 'postBA',
                 'postOE', 'postOP', 'postOA']


def _term_masks_and_matrix(coefficients: ACTCoefficients) -> Tuple[np.ndarray, np.ndarray]:
    """Return (masks, M): masks is (T, 9) bool, M is (9, T) coefficients."""
    cache = getattr(coefficients, '_vec_cache', None)
    if cache is not None:
        return cache
    names = list(coefficients.coefficients.keys())
    masks = np.array([[ch == '1' for ch in n[1:]] for n in names], dtype=bool)
    M = np.array([[coefficients.get_coefficient(n, o) for n in names]
                  for o in _OUTPUT_ORDER], dtype=float)
    coefficients._vec_cache = (masks, M)
    return masks, M


def impression_formation_vec(x: np.ndarray, coefficients: ACTCoefficients) -> np.ndarray:
    """Vectorised impression formation: tau = M g(x).

    Args:
        x: Array of shape (9,) or (N, 9) ordered [A_epa, B_epa, O_epa].
        coefficients: ACTCoefficients instance.

    Returns:
        Array of the same leading shape with 9 post-event transients.
    """
    masks, M = _term_masks_and_matrix(coefficients)
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = x[None, :] if single else x
    # g[n, t] = prod_{i in mask_t} X[n, i]
    G = np.prod(np.where(masks[None, :, :], X[:, None, :], 1.0), axis=2)
    out = G @ M.T
    return out[0] if single else out


def has_behaviour_interactions(coefficients: ACTCoefficients) -> bool:
    """True if any equation term multiplies two behaviour components.

    If this returns True, tau(b) is not affine and the closed form does
    not apply; fall back to ``method='lbfgsb'``.
    """
    masks, _ = _term_masks_and_matrix(coefficients)
    return bool((masks[:, 3:6].sum(axis=1) > 1).any())


def affine_behaviour_map(
    actor_transient: np.ndarray,
    object_transient: np.ndarray,
    coefficients: ACTCoefficients,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (c, J) with tau(b) = c + J b for fixed actor/object transients.

    Exact when ``has_behaviour_interactions`` is False (evaluated at b = 0
    and the three unit vectors).
    """
    a = np.asarray(actor_transient, dtype=float)
    o = np.asarray(object_transient, dtype=float)
    X = np.zeros((4, 9))
    X[:, 0:3] = a
    X[:, 6:9] = o
    X[1:, 3:6] = np.eye(3)
    T = impression_formation_vec(X, coefficients)
    c = T[0]
    J = (T[1:] - c).T  # (9, 3)
    return c, J


def optimal_behavior_closed_form(
    actor_fundamental: np.ndarray,
    object_fundamental: np.ndarray,
    actor_transient: np.ndarray,
    object_transient: np.ndarray,
    coefficients: ACTCoefficients,
    bounds: Tuple[float, float] = (-4.3, 4.3),
    e_min: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Closed-form deflection-minimising behaviour (Heise 2007, ch. 8).

    Writes the residual as rho(b) = r0 + K b with
    r0 = [f_A; 0; f_O] - c and K = P - J (P selects the behaviour block),
    so b* = argmin ||r0 + K b||^2 = -(K^T K)^{-1} K^T r0.  If b* leaves the
    box, or if an Evaluation floor ``e_min`` is requested (guardrail, WS6.2),
    the problem is solved as bounded least squares.

    Returns:
        (b_star, info) where info has the unconstrained solution's
        deflection, the final deflection, and whether bounds were active.
    """
    from scipy.optimize import lsq_linear

    if has_behaviour_interactions(coefficients):
        raise ValueError("Equation set has behaviour x behaviour terms; "
                         "use method='lbfgsb'.")
    c, J = affine_behaviour_map(actor_transient, object_transient, coefficients)
    P = np.zeros((9, 3))
    P[3:6, :] = np.eye(3)
    K = P - J
    target = np.concatenate([np.asarray(actor_fundamental, float),
                             np.zeros(3),
                             np.asarray(object_fundamental, float)])
    r0 = target - c

    b_unc, *_ = np.linalg.lstsq(K, -r0, rcond=None)
    lo = np.full(3, bounds[0], dtype=float)
    hi = np.full(3, bounds[1], dtype=float)
    if e_min is not None:
        lo[0] = max(lo[0], float(e_min))
    feasible = bool(np.all(b_unc >= lo - 1e-12) and np.all(b_unc <= hi + 1e-12))
    if feasible:
        b_star = b_unc
    else:
        b_star = lsq_linear(K, -r0, bounds=(lo, hi), method='bvls').x

    def _defl(b):
        r = r0 + K @ b
        return float(r @ r)

    info = {
        'deflection': _defl(b_star),
        'deflection_unconstrained': _defl(b_unc),
        'constraint_active': float(not feasible),
    }
    return b_star, info


def compute_act_target(
    agent_identity: EPA,
    user_identity: EPA,
    user_behavior_epa: EPA,
    coefficients: ACTCoefficients,
    method: str = 'closed_form',
    bounds: Tuple[float, float] = (-4.3, 4.3),
    e_min: Optional[float] = None,
) -> Dict[str, object]:
    """Two-event ACT target computation (paper Eqs. for Event 1 and Event 2).

    Event 1 (user acts on agent):
        x1 = [f_U; b_U; f_G],  tau1 = M g(x1) = [tau1_U; tau1_bU; tau1_G]
    Event 2 (agent responds to user with behaviour b):
        x2(b) = [tau1_G; b; tau1_U],  tau2(b) = M g(x2(b))
        D(b) = ||f_G - tau2_A(b)||^2 + ||b - tau2_B(b)||^2 + ||f_U - tau2_O(b)||^2

    Args:
        agent_identity: f_G, agent's fundamental EPA.
        user_identity: f_U, user's fundamental EPA.
        user_behavior_epa: b_U, EPA of the user's message (read by RepE).
        coefficients: Impression-formation equations.
        method: 'closed_form' (default) or 'lbfgsb' (legacy numerical).
        bounds: Box constraint on each behaviour component.
        e_min: Optional floor on the Evaluation of the agent's behaviour.

    Returns:
        Dict with 'target' (EPA), 'tau1' (9-vector), 'deflection',
        'deflection_event1' and 'constraint_active'.
    """
    fG = agent_identity.to_array()
    fU = user_identity.to_array()
    bU = user_behavior_epa.to_array()
    tau1 = impression_formation_vec(np.concatenate([fU, bU, fG]), coefficients)
    tau1_U, tau1_G = tau1[0:3], tau1[6:9]
    defl1 = float(np.sum((fU - tau1_U) ** 2) + np.sum((bU - tau1[3:6]) ** 2)
                  + np.sum((fG - tau1_G) ** 2))

    if method == 'closed_form':
        b_star, info = optimal_behavior_closed_form(
            fG, fU, tau1_G, tau1_U, coefficients, bounds=bounds, e_min=e_min)
    elif method == 'lbfgsb':
        lo = bounds[0] if e_min is None else max(bounds[0], e_min)
        res = minimize(
            lambda b: _deflection_event2(b, fG, fU, tau1_G, tau1_U, coefficients),
            np.clip(np.zeros(3), [lo, bounds[0], bounds[0]], bounds[1]),
            bounds=[(lo, bounds[1]), (bounds[0], bounds[1]), (bounds[0], bounds[1])],
            method='L-BFGS-B')
        b_star = res.x
        info = {'deflection': float(res.fun), 'constraint_active': float('nan')}
    else:
        raise ValueError(f"Unknown method {method!r}")

    return {
        'target': EPA.from_array(b_star),
        'tau1': tau1,
        'deflection': info['deflection'],
        'deflection_event1': defl1,
        'constraint_active': info.get('constraint_active', float('nan')),
    }


def _deflection_event2(b, fG, fU, tau1_G, tau1_U, coefficients) -> float:
    tau2 = impression_formation_vec(np.concatenate([tau1_G, b, tau1_U]), coefficients)
    return float(np.sum((fG - tau2[0:3]) ** 2) + np.sum((b - tau2[3:6]) ** 2)
                 + np.sum((fU - tau2[6:9]) ** 2))


# Convenience functions for common patterns

def get_response_epa_for_deflection_minimization(
    agent_identity: EPA,
    user_identity: EPA,
    user_behavior_epa: EPA,
    coefficients: ACTCoefficients,
    method: str = 'lbfgsb',
    e_min: Optional[float] = None,
) -> EPA:
    """
    Convenience function: Given agent/user identities and user's behavior,
    find the optimal agent response behavior EPA.
    
    This models: User (actor) does behavior to Agent (object), 
    then Agent needs to respond optimally.
    
    Args:
        agent_identity: Agent's fundamental EPA
        user_identity: User's fundamental EPA
        user_behavior_epa: EPA of user's input message/behavior
        coefficients: ACTCoefficients instance
        method: ``'lbfgsb'`` (legacy default, used for all reported v1
            results) or ``'closed_form'`` (bounded least squares; see
            ``compute_act_target``).  The two agree to numerical precision
            on the 2010 equations (checked by ``analysis/act_targets.py``).
        e_min: Optional Evaluation floor (guardrail).  Forces the
            closed-form path.
        
    Returns:
        Optimal behavior EPA for agent's response
    """
    if method == 'closed_form' or e_min is not None:
        return compute_act_target(
            agent_identity, user_identity, user_behavior_epa, coefficients,
            method='closed_form', e_min=e_min)['target']

    # First, calculate transient impressions after user's action
    post_user_action = impression_formation(
        actor=user_identity,
        behavior=user_behavior_epa,
        obj=agent_identity,
        coefficients=coefficients
    )
    
    # Now find optimal behavior for agent (as actor) toward user (as object)
    # Using the transient impressions from user's action
    optimal_behavior = find_optimal_behavior(
        actor_fundamental=agent_identity,
        object_fundamental=user_identity,
        actor_transient=post_user_action['object'],  # Agent's transient (was object)
        object_transient=post_user_action['actor'],  # User's transient (was actor)
        coefficients=coefficients
    )
    
    return optimal_behavior


# Default coefficients instance (lazy-loaded)
_default_coefficients: Optional[ACTCoefficients] = None

def get_default_coefficients() -> ACTCoefficients:
    """Get or create default coefficients instance."""
    global _default_coefficients
    if _default_coefficients is None:
        _default_coefficients = ACTCoefficients()
    return _default_coefficients
