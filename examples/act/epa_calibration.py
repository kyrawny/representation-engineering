"""
EPA Calibration Module

Provides calibration methods to align raw EPA vector readings with
ACT dictionary values:
- LinearRegressionCalibrator: Fast linear mapping
- FineTuningCalibrator: Gradient-based fine-tuning
- AffineCalibrator: Affine transformation with rotation
"""

import numpy as np
import pandas as pd
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
import warnings

from .act_core import EPA


# =============================================================================
# Behavior Selection for Calibration
# =============================================================================

# Behaviors suitable for conversational calibration
# These are behaviors that can be expressed through dialogue
CONVERSATIONAL_BEHAVIORS = [
    # Positive evaluation, various P/A
    "thank", "compliment", "praise", "encourage", "comfort", "reassure",
    "appreciate", "congratulate", "support", "help", "advise", "welcome",
    "forgive", "trust", "respect", "admire", "love", "care_for",
    
    # Negative evaluation, various P/A
    "criticize", "insult", "threaten", "blame", "accuse", "condemn",
    "mock", "ridicule", "belittle", "dismiss", "ignore", "reject",
    "deceive", "lie_to", "manipulate", "bully", "harass",
    
    # Neutral-ish evaluation, high potency
    "demand", "order", "command", "warn", "challenge", "confront",
    "interrogate", "question", "lecture", "instruct", "persuade",
    
    # Neutral-ish evaluation, low potency
    "beg", "plead_with", "apologize_to", "confess_to", "ask", "request_something_from",
    "appeal_to", "implore", "entreat",
    
    # High activity
    "excite", "alarm", "surprise", "shock", "amaze", "dazzle",
    "argue_with", "debate_with", "joke_with", "play_with",
    
    # Low activity
    "calm", "soothe", "comfort", "listen_to", "observe", "watch",
    "console", "sympathize_with", "understand",
    
    # Various combinations
    "explain_something_to", "inform", "tell_something_to", "remind",
    "greet", "introduce", "meet", "chat_with", "converse_with",
    "agree_with", "disagree_with", "contradict", "correct",
]


class BehaviorPromptGenerator:
    """
    Generate conversational utterances that embody behaviors from the ACT dictionary.
    
    Uses an LLM or templates to create example dialogue turns for each behavior.
    """
    
    # Template-based generation for common behaviors
    BEHAVIOR_TEMPLATES = {
        "beg": [
            "Please, I really need your help with this...",
            "I'm begging you, please consider this...",
            "Please, my family and I don't have much and we really need this...",
        ],
        "thank": [
            "Thank you so much, I really appreciate your help!",
            "I can't thank you enough for everything you've done.",
            "That means the world to me, thank you!",
        ],
        "compliment": [
            "I have to say, you did an amazing job on this.",
            "You're incredibly talented at what you do.",
            "I'm really impressed by your work here.",
        ],
        "threaten": [
            "You really don't want to go down this path.",
            "If you don't comply, there will be consequences.",
            "I suggest you reconsider, for your own sake.",
        ],
        "demand": [
            "I need you to do this right now.",
            "This is not a request. Get it done.",
            "You will provide me with the information immediately.",
        ],
        "apologize_to": [
            "I'm so sorry for what happened. Please forgive me.",
            "I owe you an apology. I was completely wrong.",
            "I deeply regret my actions and I'm truly sorry.",
        ],
        "encourage": [
            "You've got this! I believe in you!",
            "Keep going, you're doing great!",
            "Don't give up, you're almost there!",
        ],
        "criticize": [
            "This is really not up to standard.",
            "I expected better from you, honestly.",
            "There are several problems with how you handled this.",
        ],
        "comfort": [
            "It's okay, everything is going to be alright.",
            "I'm here for you, no matter what.",
            "Don't worry, we'll get through this together.",
        ],
        "dismiss": [
            "That's not really relevant to the discussion.",
            "I don't think that's worth considering.",
            "Let's move on to more important matters.",
        ],
    }
    
    def __init__(
        self, 
        behaviors_csv: Optional[str] = None,
        filter_conversational: bool = True
    ):
        """
        Initialize generator with behavior dictionary.
        
        Args:
            behaviors_csv: Path to behaviors CSV
            filter_conversational: If True, only include conversational behaviors
        """
        if behaviors_csv is None:
            behaviors_csv = Path(__file__).parent.parent.parent / "data" / "act" / "MTurkInteract_Behaviors.csv"
        
        self.behaviors_df = pd.read_csv(behaviors_csv)
        self._build_lookup()
        
        if filter_conversational:
            self.available_behaviors = [
                b for b in self.behaviors.keys() 
                if b in CONVERSATIONAL_BEHAVIORS
            ]
        else:
            self.available_behaviors = list(self.behaviors.keys())
    
    def _build_lookup(self):
        """Build behavior EPA lookup."""
        self.behaviors: Dict[str, EPA] = {}
        for _, row in self.behaviors_df.iterrows():
            name = row['term']
            epa = EPA(e=row['E'], p=row['P'], a=row['A'])
            self.behaviors[name] = epa
    
    def get_behavior_epa(self, behavior: str) -> Optional[EPA]:
        """Get EPA for a behavior."""
        return self.behaviors.get(behavior)
    
    def generate_utterance(self, behavior: str, variant: int = 0) -> Optional[str]:
        """
        Generate an utterance embodying the given behavior.
        
        Args:
            behavior: Behavior name from dictionary
            variant: Which variant to use (for behaviors with multiple templates)
            
        Returns:
            Utterance string or None if behavior not available
        """
        if behavior in self.BEHAVIOR_TEMPLATES:
            templates = self.BEHAVIOR_TEMPLATES[behavior]
            return templates[variant % len(templates)]
        
        # Fallback: construct a generic prompt
        # In production, this could call an LLM
        behavior_display = behavior.replace("_", " ")
        return f"[Expressing {behavior_display}]"
    
    def generate_calibration_pairs(
        self,
        n_samples: int = 100,
        behaviors: Optional[List[str]] = None
    ) -> List[Tuple[str, EPA]]:
        """
        Generate (utterance, EPA) pairs for calibration.
        
        Args:
            n_samples: Number of pairs to generate
            behaviors: Specific behaviors to use (default: all available)
            
        Returns:
            List of (utterance, target_EPA) tuples
        """
        behaviors = behaviors or self.available_behaviors
        pairs = []
        
        for i in range(n_samples):
            behavior = behaviors[i % len(behaviors)]
            epa = self.get_behavior_epa(behavior)
            if epa is None:
                continue
            
            utterance = self.generate_utterance(behavior, variant=i // len(behaviors))
            if utterance:
                pairs.append((utterance, epa))
        
        return pairs


# =============================================================================
# Calibration Coefficients
# =============================================================================

@dataclass
class CalibrationCoefficients:
    """Stores calibration coefficients for EPA mapping."""
    
    # Forward mapping: raw -> calibrated
    # calibrated = forward_weights @ raw + forward_bias
    forward_weights: np.ndarray  # Shape: (3, 3) or (3,) for diagonal
    forward_bias: np.ndarray     # Shape: (3,)
    
    # Metadata
    method: str = "linear"
    r2_scores: Dict[str, float] = field(default_factory=dict)
    
    def transform(self, raw_epa: np.ndarray) -> np.ndarray:
        """Transform raw EPA to calibrated EPA."""
        if self.forward_weights.ndim == 1:
            # Diagonal matrix (simple scaling)
            return self.forward_weights * raw_epa + self.forward_bias
        else:
            return self.forward_weights @ raw_epa + self.forward_bias
    
    def inverse_transform(self, calibrated_epa: np.ndarray) -> np.ndarray:
        """Transform calibrated EPA back to raw EPA space."""
        if self.forward_weights.ndim == 1:
            # Diagonal: simple inverse
            return (calibrated_epa - self.forward_bias) / (self.forward_weights + 1e-8)
        else:
            # Full matrix inverse
            try:
                inv_weights = np.linalg.inv(self.forward_weights)
                return inv_weights @ (calibrated_epa - self.forward_bias)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                inv_weights = np.linalg.pinv(self.forward_weights)
                return inv_weights @ (calibrated_epa - self.forward_bias)
    
    def to_epa(self, raw: EPA) -> EPA:
        """Transform EPA object."""
        result = self.transform(raw.to_array())
        return EPA.from_array(result)
    
    def from_epa(self, calibrated: EPA) -> EPA:
        """Inverse transform EPA object."""
        result = self.inverse_transform(calibrated.to_array())
        return EPA.from_array(result)
    
    def save(self, path: str):
        """Save coefficients to file."""
        data = {
            'forward_weights': self.forward_weights.tolist(),
            'forward_bias': self.forward_bias.tolist(),
            'method': self.method,
            'r2_scores': self.r2_scores,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationCoefficients':
        """Load coefficients from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            forward_weights=np.array(data['forward_weights']),
            forward_bias=np.array(data['forward_bias']),
            method=data.get('method', 'linear'),
            r2_scores=data.get('r2_scores', {}),
        )


# =============================================================================
# Calibrator Base Class
# =============================================================================

class EPACalibrator(ABC):
    """Abstract base class for EPA calibration strategies."""
    
    @abstractmethod
    def fit(
        self,
        raw_epas: np.ndarray,
        target_epas: np.ndarray
    ) -> CalibrationCoefficients:
        """
        Fit calibration from raw EPA readings to target (dictionary) EPA values.
        
        Args:
            raw_epas: Raw EPA readings from model, shape (n_samples, 3)
            target_epas: Target EPA values from dictionary, shape (n_samples, 3)
            
        Returns:
            Fitted CalibrationCoefficients
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the calibration method name."""
        pass


# =============================================================================
# Linear Regression Calibrator
# =============================================================================

class LinearRegressionCalibrator(EPACalibrator):
    """
    Calibrate using linear regression.
    
    Fast, requires no training. Fits separate linear models for E, P, A.
    """
    
    def __init__(self, use_ridge: bool = False, alpha: float = 1.0):
        """
        Args:
            use_ridge: Whether to use Ridge regression for regularization
            alpha: Ridge regularization strength
        """
        self.use_ridge = use_ridge
        self.alpha = alpha
    
    def get_method_name(self) -> str:
        return "ridge" if self.use_ridge else "linear"
    
    def fit(
        self,
        raw_epas: np.ndarray,
        target_epas: np.ndarray
    ) -> CalibrationCoefficients:
        """Fit linear regression from raw to target EPAs."""
        
        n_samples = raw_epas.shape[0]
        
        # Fit separate model for each dimension
        # This allows independent calibration of E, P, A
        weights = np.zeros((3, 3))
        biases = np.zeros(3)
        r2_scores = {}
        
        dim_names = ['E', 'P', 'A']
        
        for i, dim in enumerate(dim_names):
            X = raw_epas
            y = target_epas[:, i]
            
            if self.use_ridge:
                model = Ridge(alpha=self.alpha)
            else:
                model = LinearRegression()
            
            model.fit(X, y)
            
            weights[i, :] = model.coef_
            biases[i] = model.intercept_
            
            # Calculate R² score
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores[dim] = float(r2)
        
        return CalibrationCoefficients(
            forward_weights=weights,
            forward_bias=biases,
            method=self.get_method_name(),
            r2_scores=r2_scores,
        )


# =============================================================================
# Affine Calibrator
# =============================================================================

class AffineCalibrator(EPACalibrator):
    """
    Calibrate using full affine transformation.
    
    Allows rotation, scaling, and translation in EPA space.
    Richer than linear regression but may overfit with few samples.
    """
    
    def __init__(self, regularization: float = 0.01):
        """
        Args:
            regularization: L2 regularization strength for stability
        """
        self.regularization = regularization
    
    def get_method_name(self) -> str:
        return "affine"
    
    def fit(
        self,
        raw_epas: np.ndarray,
        target_epas: np.ndarray
    ) -> CalibrationCoefficients:
        """Fit affine transformation from raw to target EPAs."""
        
        n_samples = raw_epas.shape[0]
        
        # Solve: target = raw @ W.T + b
        # This is equivalent to fitting a multivariate linear model
        
        # Add regularization for stability
        X = raw_epas
        Y = target_epas
        
        # Add bias term by augmenting X
        X_aug = np.hstack([X, np.ones((n_samples, 1))])  # (n, 4)
        
        # Solve with regularization: (X.T @ X + λI) @ W = X.T @ Y
        reg_matrix = self.regularization * np.eye(4)
        reg_matrix[3, 3] = 0  # Don't regularize bias term
        
        try:
            W = np.linalg.solve(
                X_aug.T @ X_aug + reg_matrix,
                X_aug.T @ Y
            )  # (4, 3)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
        
        weights = W[:3, :].T  # (3, 3)
        biases = W[3, :]  # (3,)
        
        # Calculate R² scores
        Y_pred = X @ weights.T + biases
        r2_scores = {}
        for i, dim in enumerate(['E', 'P', 'A']):
            ss_res = np.sum((Y[:, i] - Y_pred[:, i]) ** 2)
            ss_tot = np.sum((Y[:, i] - np.mean(Y[:, i])) ** 2)
            r2_scores[dim] = float(1 - ss_res / (ss_tot + 1e-8))
        
        return CalibrationCoefficients(
            forward_weights=weights,
            forward_bias=biases,
            method="affine",
            r2_scores=r2_scores,
        )


# =============================================================================
# Fine-Tuning Calibrator
# =============================================================================

class FineTuningCalibrator(EPACalibrator):
    """
    Calibrate by fine-tuning the representation reading pipeline.
    
    Uses gradient descent to adjust how EPA vectors are read from the model.
    This is more complex but can achieve better alignment.
    
    Note: Requires access to the underlying model and pipeline.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 16
    ):
        """
        Args:
            learning_rate: Learning rate for gradient descent
            n_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
    
    def get_method_name(self) -> str:
        return "fine_tuning"
    
    def fit(
        self,
        raw_epas: np.ndarray,
        target_epas: np.ndarray
    ) -> CalibrationCoefficients:
        """
        Fit using gradient descent.
        
        Note: This is a simplified version that learns a linear transform.
        Full fine-tuning would adjust the actual model/pipeline.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Convert to tensors
        X = torch.tensor(raw_epas, dtype=torch.float32)
        Y = torch.tensor(target_epas, dtype=torch.float32)
        
        # Simple linear layer as transform
        transform = nn.Linear(3, 3)
        
        # Initialize close to identity
        with torch.no_grad():
            transform.weight.copy_(torch.eye(3))
            transform.bias.zero_()
        
        optimizer = optim.Adam(transform.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        n_samples = X.shape[0]
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_Y = Y_shuffled[i:i+self.batch_size]
                
                optimizer.zero_grad()
                pred = transform(batch_X)
                loss = criterion(pred, batch_Y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        
        # Extract learned parameters
        weights = transform.weight.detach().numpy()
        biases = transform.bias.detach().numpy()
        
        # Calculate final R² scores
        with torch.no_grad():
            Y_pred = transform(X).numpy()
        
        r2_scores = {}
        for i, dim in enumerate(['E', 'P', 'A']):
            ss_res = np.sum((target_epas[:, i] - Y_pred[:, i]) ** 2)
            ss_tot = np.sum((target_epas[:, i] - np.mean(target_epas[:, i])) ** 2)
            r2_scores[dim] = float(1 - ss_res / (ss_tot + 1e-8))
        
        return CalibrationCoefficients(
            forward_weights=weights,
            forward_bias=biases,
            method="fine_tuning",
            r2_scores=r2_scores,
        )


# =============================================================================
# Calibration Pipeline
# =============================================================================

def calibrate_from_behaviors(
    read_epa_fn: Callable[[str], EPA],
    calibrator: Optional[EPACalibrator] = None,
    n_samples: int = 100,
    behaviors: Optional[List[str]] = None
) -> CalibrationCoefficients:
    """
    End-to-end calibration using behavior-based prompts.
    
    Args:
        read_epa_fn: Function that reads EPA from an utterance
        calibrator: Calibrator to use (default: LinearRegressionCalibrator)
        n_samples: Number of calibration samples
        behaviors: Specific behaviors to use
        
    Returns:
        Fitted CalibrationCoefficients
    """
    generator = BehaviorPromptGenerator()
    calibrator = calibrator or LinearRegressionCalibrator()
    
    # Generate calibration pairs
    pairs = generator.generate_calibration_pairs(n_samples, behaviors)
    
    # Read raw EPAs
    raw_epas = []
    target_epas = []
    
    for utterance, target_epa in pairs:
        try:
            raw_epa = read_epa_fn(utterance)
            raw_epas.append(raw_epa.to_array())
            target_epas.append(target_epa.to_array())
        except Exception as e:
            warnings.warn(f"Failed to read EPA for '{utterance[:30]}...': {e}")
    
    if len(raw_epas) < 10:
        raise ValueError(f"Only {len(raw_epas)} valid samples, need at least 10")
    
    raw_epas = np.array(raw_epas)
    target_epas = np.array(target_epas)
    
    # Fit calibration
    return calibrator.fit(raw_epas, target_epas)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_calibrator(method: str = "linear", **kwargs) -> EPACalibrator:
    """
    Get calibrator by method name.
    
    Args:
        method: One of "linear", "ridge", "affine", "fine_tuning"
        **kwargs: Additional arguments for the calibrator
        
    Returns:
        EPACalibrator instance
    """
    if method == "linear":
        return LinearRegressionCalibrator(**kwargs)
    elif method == "ridge":
        return LinearRegressionCalibrator(use_ridge=True, **kwargs)
    elif method == "affine":
        return AffineCalibrator(**kwargs)
    elif method == "fine_tuning":
        return FineTuningCalibrator(**kwargs)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
