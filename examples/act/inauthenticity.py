"""
Inauthenticity Module

Implements inauthenticity constraints from Heise's ACT:
- Calculate inauthenticity as deviation from authentic expression
- Constrained optimization balancing deflection and inauthenticity
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize

from .act_core import (
    EPA, ACTCoefficients, get_default_coefficients,
    impression_formation, calculate_deflection
)


def calculate_inauthenticity(
    fundamental_epa: EPA,
    expressed_epa: EPA,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate inauthenticity of an expressed behavior.
    
    Inauthenticity measures how far the expressed behavior deviates from
    what would be authentic (natural) for the actor's fundamental identity.
    
    Per Heise's ACT, authentic behaviors are those consistent with one's
    fundamental identity - high-status individuals authentically express
    potent behaviors, kind individuals authentically express positive
    evaluation behaviors, etc.
    
    Args:
        fundamental_epa: Actor's fundamental identity EPA
        expressed_epa: EPA of the behavior being expressed
        weights: Optional weights for E, P, A dimensions (default: equal)
        
    Returns:
        Inauthenticity score (lower = more authentic)
    """
    if weights is None:
        weights = np.ones(3)
    
    fund = fundamental_epa.to_array()
    expr = expressed_epa.to_array()
    
    # Inauthenticity as weighted squared deviation
    diff = (fund - expr) ** 2
    return float(np.sum(weights * diff))


def calculate_weighted_inauthenticity(
    fundamental_epa: EPA,
    expressed_epa: EPA,
    dimension_weights: Dict[str, float] = None
) -> float:
    """
    Calculate inauthenticity with dimension-specific weights.
    
    Args:
        fundamental_epa: Actor's fundamental EPA
        expressed_epa: Expressed behavior EPA
        dimension_weights: Dict with 'e', 'p', 'a' weights
        
    Returns:
        Weighted inauthenticity score
    """
    if dimension_weights is None:
        dimension_weights = {'e': 1.0, 'p': 1.0, 'a': 1.0}
    
    weights = np.array([
        dimension_weights.get('e', 1.0),
        dimension_weights.get('p', 1.0),
        dimension_weights.get('a', 1.0)
    ])
    
    return calculate_inauthenticity(fundamental_epa, expressed_epa, weights)


@dataclass
class InauthenticityConstraints:
    """Configuration for inauthenticity constraints in optimization."""
    
    # Maximum allowed inauthenticity
    max_inauthenticity: float = 5.0
    
    # Weight in combined objective (deflection + weight * inauthenticity)
    authenticity_weight: float = 0.5
    
    # Dimension-specific weights
    e_weight: float = 1.0  # Evaluation
    p_weight: float = 1.0  # Potency
    a_weight: float = 1.0  # Activity


def constrained_optimal_behavior(
    actor_fundamental: EPA,
    object_fundamental: EPA,
    actor_transient: EPA,
    object_transient: EPA,
    coefficients: ACTCoefficients,
    constraints: Optional[InauthenticityConstraints] = None,
    bounds: Tuple[float, float] = (-4.3, 4.3)
) -> Tuple[EPA, Dict[str, float]]:
    """
    Find optimal behavior subject to inauthenticity constraints.
    
    Balances two objectives:
    1. Minimize deflection (ACT's primary goal)
    2. Minimize inauthenticity (stay true to actor's identity)
    
    Args:
        actor_fundamental: Actor's fundamental EPA
        object_fundamental: Object's fundamental EPA
        actor_transient: Actor's current transient EPA
        object_transient: Object's current transient EPA
        coefficients: ACT coefficients
        constraints: Inauthenticity constraint configuration
        bounds: Behavior EPA bounds
        
    Returns:
        Tuple of (optimal_behavior_EPA, metrics_dict)
    """
    constraints = constraints or InauthenticityConstraints()
    
    dimension_weights = {
        'e': constraints.e_weight,
        'p': constraints.p_weight,
        'a': constraints.a_weight,
    }
    
    def combined_objective(b_vec: np.ndarray) -> float:
        """Combined objective: deflection + weighted inauthenticity."""
        behavior = EPA.from_array(b_vec)
        
        # Calculate post-event impressions
        post = impression_formation(
            actor_transient, behavior, object_transient, coefficients
        )
        
        # Deflection from fundamentals
        actor_defl = calculate_deflection(actor_fundamental, post['actor'])
        behavior_defl = calculate_deflection(behavior, post['behavior'])
        object_defl = calculate_deflection(object_fundamental, post['object'])
        total_deflection = actor_defl + behavior_defl + object_defl
        
        # Inauthenticity of behavior for actor
        inauthenticity = calculate_weighted_inauthenticity(
            actor_fundamental, behavior, dimension_weights
        )
        
        # Combined objective
        return total_deflection + constraints.authenticity_weight * inauthenticity
    
    def inauthenticity_constraint(b_vec: np.ndarray) -> float:
        """Constraint: inauthenticity <= max_inauthenticity."""
        behavior = EPA.from_array(b_vec)
        inauth = calculate_weighted_inauthenticity(
            actor_fundamental, behavior, dimension_weights
        )
        return constraints.max_inauthenticity - inauth
    
    # Initial guess: actor's fundamental (most authentic)
    x0 = actor_fundamental.to_array()
    
    # Bounds and constraints
    opt_bounds = [(bounds[0], bounds[1]) for _ in range(3)]
    
    opt_constraints = []
    if constraints.max_inauthenticity < float('inf'):
        opt_constraints.append({
            'type': 'ineq',
            'fun': inauthenticity_constraint
        })
    
    # Run optimization
    if opt_constraints:
        result = minimize(
            combined_objective,
            x0,
            bounds=opt_bounds,
            constraints=opt_constraints,
            method='SLSQP'
        )
    else:
        result = minimize(
            combined_objective,
            x0,
            bounds=opt_bounds,
            method='L-BFGS-B'
        )
    
    optimal_behavior = EPA.from_array(result.x)
    
    # Calculate final metrics
    post = impression_formation(
        actor_transient, optimal_behavior, object_transient, coefficients
    )
    
    actor_defl = calculate_deflection(actor_fundamental, post['actor'])
    behavior_defl = calculate_deflection(optimal_behavior, post['behavior'])
    object_defl = calculate_deflection(object_fundamental, post['object'])
    
    inauthenticity = calculate_weighted_inauthenticity(
        actor_fundamental, optimal_behavior, dimension_weights
    )
    
    metrics = {
        'total_deflection': actor_defl + behavior_defl + object_defl,
        'actor_deflection': actor_defl,
        'behavior_deflection': behavior_defl,
        'object_deflection': object_defl,
        'inauthenticity': inauthenticity,
        'combined_objective': result.fun,
        'optimization_success': result.success,
    }
    
    return optimal_behavior, metrics


def find_authentic_behavior_range(
    actor_fundamental: EPA,
    max_inauthenticity: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the range of behaviors that are authentic for an actor.
    
    Returns bounds for behaviors within the inauthenticity threshold.
    
    Args:
        actor_fundamental: Actor's fundamental EPA
        max_inauthenticity: Maximum allowed inauthenticity
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) for E, P, A
    """
    # For simple equal weights, inauthenticity = sum((fund - expr)^2)
    # Max deviation per dimension: sqrt(max_inauthenticity / 3)
    max_dev = np.sqrt(max_inauthenticity / 3)
    
    fund = actor_fundamental.to_array()
    lower = np.maximum(fund - max_dev, -4.3)
    upper = np.minimum(fund + max_dev, 4.3)
    
    return lower, upper


class AuthenticityAwareEngine:
    """
    Extension of steering engine with authenticity awareness.
    
    Provides additional methods for authenticity-constrained steering.
    """
    
    def __init__(
        self,
        actor_fundamental: EPA,
        constraints: Optional[InauthenticityConstraints] = None
    ):
        self.actor_fundamental = actor_fundamental
        self.constraints = constraints or InauthenticityConstraints()
        self._coefficients = get_default_coefficients()
    
    def is_authentic(self, behavior_epa: EPA) -> bool:
        """Check if a behavior is within authenticity bounds."""
        inauth = calculate_inauthenticity(self.actor_fundamental, behavior_epa)
        return inauth <= self.constraints.max_inauthenticity
    
    def get_authenticity_score(self, behavior_epa: EPA) -> float:
        """
        Get authenticity score (0-1, higher = more authentic).
        
        Returns:
            Score between 0 (very inauthentic) and 1 (perfectly authentic)
        """
        inauth = calculate_inauthenticity(self.actor_fundamental, behavior_epa)
        # Convert to 0-1 score with exponential decay
        return float(np.exp(-inauth / self.constraints.max_inauthenticity))
    
    def clamp_to_authentic(self, behavior_epa: EPA) -> EPA:
        """
        Clamp a behavior EPA to the closest authentic value.
        
        Args:
            behavior_epa: Proposed behavior EPA
            
        Returns:
            Clamped behavior EPA within authenticity bounds
        """
        lower, upper = find_authentic_behavior_range(
            self.actor_fundamental,
            self.constraints.max_inauthenticity
        )
        
        clamped = np.clip(behavior_epa.to_array(), lower, upper)
        return EPA.from_array(clamped)
    
    def find_optimal(
        self,
        object_fundamental: EPA,
        actor_transient: EPA,
        object_transient: EPA
    ) -> Tuple[EPA, Dict[str, float]]:
        """Find optimal authentic behavior."""
        return constrained_optimal_behavior(
            actor_fundamental=self.actor_fundamental,
            object_fundamental=object_fundamental,
            actor_transient=actor_transient,
            object_transient=object_transient,
            coefficients=self._coefficients,
            constraints=self.constraints
        )
