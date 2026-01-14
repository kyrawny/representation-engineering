"""
Conversation Steering Module

Core steering engine for ACT-based conversational LLM control:
- ConversationState: Track conversation history and EPA states
- DeflectionController: PID-style error correction with decay
- ACTSteeringEngine: Main steering interface
- PromptFormatConfig: LLM prompt template configuration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
from collections import deque

from .act_core import (
    EPA, ACTCoefficients, get_default_coefficients,
    impression_formation, calculate_deflection, find_optimal_behavior,
    get_response_epa_for_deflection_minimization
)
from .identity_manager import (
    Identity, Modifier, ModifiedIdentity, 
    IdentityDatabase, get_identity_database,
    create_identity
)
from .epa_calibration import CalibrationCoefficients


# =============================================================================
# Prompt Format Configuration  
# =============================================================================

@dataclass
class PromptFormatConfig:
    """
    Configuration for LLM prompt formatting.
    
    Allows customization for different models (Llama, Mistral, etc.)
    """
    bos_token: str = "<|begin_of_text|>"
    eos_token: str = "<|eot_id|>"
    
    system_start: str = "<|start_header_id|>system<|end_header_id|>\n\n"
    system_end: str = "<|eot_id|>"
    
    user_start: str = "<|start_header_id|>user<|end_header_id|>\n\n"
    user_end: str = "<|eot_id|>"
    
    assistant_start: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_end: str = "<|eot_id|>"
    
    include_bos: bool = True
    
    def format_prompt(
        self,
        system_prompt: str,
        user_message: str,
        assistant_prefix: str = ""
    ) -> str:
        """Format a complete prompt."""
        parts = []
        
        if self.include_bos:
            parts.append(self.bos_token)
        
        parts.append(self.system_start)
        parts.append(system_prompt)
        parts.append(self.system_end)
        
        parts.append(self.user_start)
        parts.append(user_message)
        parts.append(self.user_end)
        
        parts.append(self.assistant_start)
        if assistant_prefix:
            parts.append(assistant_prefix)
        
        return "".join(parts)
    
    def format_for_reading(self, text: str, context: str = "What do you think?") -> str:
        """Format text as assistant response for EPA reading."""
        system = "You are in a conversation."
        return self.format_prompt(system, context, text)
    
    @classmethod
    def llama3_instruct(cls) -> 'PromptFormatConfig':
        """Get Llama 3 Instruct format (default)."""
        return cls()
    
    @classmethod
    def mistral_instruct(cls) -> 'PromptFormatConfig':
        """Get Mistral Instruct format."""
        return cls(
            bos_token="<s>",
            eos_token="</s>",
            system_start="[INST] ",
            system_end=" ",
            user_start="",
            user_end=" [/INST]",
            assistant_start="",
            assistant_end="</s>",
        )


# =============================================================================
# Conversation State
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    epa_read: Optional[EPA] = None  # EPA reading of this turn
    epa_target: Optional[EPA] = None  # Target EPA (for assistant turns)
    deflection: float = 0.0  # Deflection at this turn


@dataclass
class ConversationState:
    """
    Tracks the full state of an ACT conversation.
    
    Maintains:
    - Agent and user identities (fundamental EPAs)
    - Current transient impressions
    - Conversation history with EPA readings
    - Cumulative metrics
    """
    
    # Identities
    agent_identity: ModifiedIdentity
    user_identity: ModifiedIdentity
    
    # Transient impressions (updated after each turn)
    agent_transient: EPA = field(default=None)
    user_transient: EPA = field(default=None)
    
    # Conversation history
    history: List[ConversationTurn] = field(default_factory=list)
    
    # Metrics
    total_deflection: float = 0.0
    turn_count: int = 0
    
    def __post_init__(self):
        # Initialize transients to fundamentals
        if self.agent_transient is None:
            self.agent_transient = self.agent_identity.epa
        if self.user_transient is None:
            self.user_transient = self.user_identity.epa
    
    @property
    def agent_fundamental(self) -> EPA:
        return self.agent_identity.epa
    
    @property
    def user_fundamental(self) -> EPA:
        return self.user_identity.epa
    
    def add_turn(
        self,
        role: str,
        content: str,
        epa_read: Optional[EPA] = None,
        epa_target: Optional[EPA] = None,
        deflection: float = 0.0
    ):
        """Add a conversation turn to history."""
        turn = ConversationTurn(
            role=role,
            content=content,
            epa_read=epa_read,
            epa_target=epa_target,
            deflection=deflection
        )
        self.history.append(turn)
        self.turn_count += 1
        self.total_deflection += deflection
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return self.history[-n:] if self.history else []
    
    def reset_transients(self):
        """Reset transient impressions to fundamentals."""
        self.agent_transient = self.agent_fundamental
        self.user_transient = self.user_fundamental


# =============================================================================
# Deflection Controller
# =============================================================================

class ContextMode(Enum):
    """How the deflection controller uses conversation context."""
    TURN_BY_TURN = "turn_by_turn"  # Only current turn, no history
    HISTORY = "history"  # Use conversation history with windowing


@dataclass
class DeflectionControllerConfig:
    """Configuration for the deflection controller."""
    
    enabled: bool = True
    
    # Context settings
    context_mode: ContextMode = ContextMode.TURN_BY_TURN
    window_size: int = 5  # Number of past turns to consider (for HISTORY mode)
    
    # Decay settings
    use_decay: bool = True
    decay_rate: float = 0.8  # Older errors weighted by decay_rate^turns_ago
    
    # PID gains
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.05  # Derivative gain


class DeflectionController:
    """
    PID-style error correction for deflection minimization.
    
    Tracks the error between optimal and actual EPA values over time
    and adjusts future targets to compensate.
    """
    
    def __init__(self, config: Optional[DeflectionControllerConfig] = None):
        self.config = config or DeflectionControllerConfig()
        
        # Error history for PID
        self.error_history: deque = deque(maxlen=100)
        self.previous_error: Optional[np.ndarray] = None
        self.integral_error: np.ndarray = np.zeros(3)
    
    @property
    def enabled(self) -> bool:
        return self.config.enabled
    
    @enabled.setter  
    def enabled(self, value: bool):
        self.config.enabled = value
    
    def reset(self):
        """Reset controller state."""
        self.error_history.clear()
        self.previous_error = None
        self.integral_error = np.zeros(3)
    
    def _compute_weighted_integral(self) -> np.ndarray:
        """Compute weighted integral of errors with decay."""
        if not self.error_history:
            return np.zeros(3)
        
        integral = np.zeros(3)
        n_errors = len(self.error_history)
        
        for i, error in enumerate(self.error_history):
            if self.config.use_decay:
                # More recent errors have higher weight
                turns_ago = n_errors - 1 - i
                weight = self.config.decay_rate ** turns_ago
            else:
                weight = 1.0
            integral += weight * error
        
        return integral
    
    def compute_adjustment(
        self,
        optimal_epa: EPA,
        actual_epa: EPA,
        conversation_state: Optional[ConversationState] = None
    ) -> EPA:
        """
        Compute PID-adjusted target EPA.
        
        Args:
            optimal_epa: ACT-computed optimal behavior EPA
            actual_epa: Actually achieved EPA (read from response)
            conversation_state: Optional state for history-based computation
            
        Returns:
            Adjusted target EPA for next response
        """
        if not self.config.enabled:
            return optimal_epa
        
        # Current error
        optimal_arr = optimal_epa.to_array()
        actual_arr = actual_epa.to_array()
        error = optimal_arr - actual_arr
        
        # Record error
        self.error_history.append(error)
        
        # Apply window if using history mode
        if self.config.context_mode == ContextMode.HISTORY:
            window = self.config.window_size
            if len(self.error_history) > window:
                # Only use recent errors
                recent_errors = list(self.error_history)[-window:]
                self.error_history.clear()
                self.error_history.extend(recent_errors)
        
        # Proportional term
        p_term = self.config.kp * error
        
        # Integral term (with decay)
        self.integral_error = self._compute_weighted_integral()
        i_term = self.config.ki * self.integral_error
        
        # Derivative term
        if self.previous_error is not None:
            d_term = self.config.kd * (error - self.previous_error)
        else:
            d_term = np.zeros(3)
        
        self.previous_error = error.copy()
        
        # Compute adjusted target
        adjustment = p_term + i_term + d_term
        adjusted = optimal_arr + adjustment
        
        # Clip to reasonable EPA bounds
        adjusted = np.clip(adjusted, -4.3, 4.3)
        
        return EPA.from_array(adjusted)
    
    def get_current_error_magnitude(self) -> float:
        """Get magnitude of current error."""
        if not self.error_history:
            return 0.0
        latest = self.error_history[-1]
        return float(np.linalg.norm(latest))


# =============================================================================
# ACT Steering Engine
# =============================================================================

class ACTSteeringEngine:
    """
    Main steering engine for ACT-controlled conversations.
    
    Coordinates:
    - Identity management
    - EPA reading and optimal behavior computation
    - Deflection control
    - Response steering
    """
    
    def __init__(
        self,
        agent_identity: str = "assistant",
        user_identity: str = "person",
        agent_modifiers: Optional[List[str]] = None,
        user_modifiers: Optional[List[str]] = None,
        coefficients: Optional[ACTCoefficients] = None,
        calibration: Optional[CalibrationCoefficients] = None,
        controller_config: Optional[DeflectionControllerConfig] = None,
        prompt_format: Optional[PromptFormatConfig] = None,
    ):
        """
        Initialize the steering engine.
        
        Args:
            agent_identity: Agent's base identity name
            user_identity: User's base identity name
            agent_modifiers: Optional modifiers for agent
            user_modifiers: Optional modifiers for user
            coefficients: ACT coefficients (uses default if None)
            calibration: EPA calibration coefficients (optional)
            controller_config: Deflection controller config
            prompt_format: LLM prompt format config
        """
        # Set up identities
        self.agent = create_identity(agent_identity, agent_modifiers)
        self.user = create_identity(user_identity, user_modifiers)
        
        # ACT coefficients
        self.coefficients = coefficients or get_default_coefficients()
        
        # Calibration (optional)
        self.calibration = calibration
        
        # Controller
        self.controller = DeflectionController(controller_config)
        
        # Prompt format
        self.prompt_format = prompt_format or PromptFormatConfig.llama3_instruct()
        
        # Initialize conversation state
        self.state = ConversationState(
            agent_identity=self.agent,
            user_identity=self.user
        )
        
        # Callbacks for EPA reading/steering (set externally)
        self._read_epa_fn: Optional[Callable[[str], EPA]] = None
        self._steer_fn: Optional[Callable[[str, EPA], str]] = None
    
    def set_read_epa_function(self, fn: Callable[[str], EPA]):
        """Set the function for reading EPA from text."""
        self._read_epa_fn = fn
    
    def set_steer_function(self, fn: Callable[[str, EPA], str]):
        """Set the function for steering LLM generation."""
        self._steer_fn = fn
    
    def set_identities(
        self,
        agent_identity: str,
        user_identity: str,
        agent_modifiers: Optional[List[str]] = None,
        user_modifiers: Optional[List[str]] = None
    ):
        """Update identities and reset conversation state."""
        self.agent = create_identity(agent_identity, agent_modifiers)
        self.user = create_identity(user_identity, user_modifiers)
        self.reset()
    
    def reset(self):
        """Reset conversation state and controller."""
        self.state = ConversationState(
            agent_identity=self.agent,
            user_identity=self.user
        )
        self.controller.reset()
    
    def read_epa(self, text: str) -> EPA:
        """Read EPA from text, applying calibration if available."""
        if self._read_epa_fn is None:
            raise RuntimeError("EPA reading function not set. Call set_read_epa_function().")
        
        raw_epa = self._read_epa_fn(text)
        
        if self.calibration:
            return self.calibration.to_epa(raw_epa)
        return raw_epa
    
    def compute_optimal_response_epa(self, user_message_epa: EPA) -> EPA:
        """Compute optimal EPA for agent's response."""
        return get_response_epa_for_deflection_minimization(
            agent_identity=self.state.agent_fundamental,
            user_identity=self.state.user_fundamental,
            user_behavior_epa=user_message_epa,
            coefficients=self.coefficients
        )
    
    def process_user_message(self, message: str) -> EPA:
        """
        Process a user message and return optimal response EPA.
        
        Args:
            message: User's message text
            
        Returns:
            Target EPA for agent's response
        """
        # Read EPA of user message
        user_epa = self.read_epa(message)
        
        # Record user turn
        self.state.add_turn(
            role="user",
            content=message,
            epa_read=user_epa
        )
        
        # Compute optimal response EPA
        optimal_epa = self.compute_optimal_response_epa(user_epa)
        
        # Update transients based on user's action
        post = impression_formation(
            actor=self.state.user_transient,
            behavior=user_epa,
            obj=self.state.agent_transient,
            coefficients=self.coefficients
        )
        self.state.user_transient = post['actor']
        self.state.agent_transient = post['object']
        
        return optimal_epa
    
    def generate_response(self, prompt: str, target_epa: EPA) -> str:
        """
        Generate a steered response.
        
        Args:
            prompt: Prompt for the LLM
            target_epa: Target EPA for steering
            
        Returns:
            Generated response text
        """
        if self._steer_fn is None:
            raise RuntimeError("Steering function not set. Call set_steer_function().")
        
        # Convert calibrated target back to raw space if needed
        if self.calibration:
            raw_target = self.calibration.from_epa(target_epa)
        else:
            raw_target = target_epa
        
        return self._steer_fn(prompt, raw_target)
    
    def process_response(self, response: str, target_epa: EPA) -> Tuple[EPA, float]:
        """
        Process a generated response.
        
        Args:
            response: Generated response text
            target_epa: What we targeted
            
        Returns:
            Tuple of (actual_epa, deflection)
        """
        # Read actual EPA achieved
        actual_epa = self.read_epa(response)
        
        # Calculate deflection
        deflection = calculate_deflection(target_epa, actual_epa)
        
        # Update controller
        if self.controller.enabled:
            self.controller.compute_adjustment(target_epa, actual_epa, self.state)
        
        # Record assistant turn
        self.state.add_turn(
            role="assistant",
            content=response,
            epa_read=actual_epa,
            epa_target=target_epa,
            deflection=deflection
        )
        
        # Update transients based on agent's response
        post = impression_formation(
            actor=self.state.agent_transient,
            behavior=actual_epa,
            obj=self.state.user_transient,
            coefficients=self.coefficients
        )
        self.state.agent_transient = post['actor']
        self.state.user_transient = post['object']
        
        return actual_epa, deflection
    
    def get_adjusted_target(self, optimal_epa: EPA) -> EPA:
        """Get PID-adjusted target EPA based on controller state."""
        if not self.controller.enabled or not self.state.history:
            return optimal_epa
        
        # Get last response's actual EPA
        for turn in reversed(self.state.history):
            if turn.role == "assistant" and turn.epa_read:
                return self.controller.compute_adjustment(
                    optimal_epa,
                    turn.epa_read,
                    self.state
                )
        
        return optimal_epa
    
    def chat(self, user_message: str, prompt: Optional[str] = None) -> str:
        """
        Complete chat turn: process user message, compute optimal EPA, generate response.
        
        Args:
            user_message: User's message
            prompt: Optional custom prompt (uses default formatting if None)
            
        Returns:
            Generated response
        """
        # Process user message to get optimal EPA
        optimal_epa = self.process_user_message(user_message)
        
        # Adjust target based on controller
        target_epa = self.get_adjusted_target(optimal_epa)
        
        # Format prompt if not provided
        if prompt is None:
            system = f"You are a {self.agent.name} speaking with a {self.user.name}."
            prompt = self.prompt_format.format_prompt(system, user_message)
        
        # Generate response
        response = self.generate_response(prompt, target_epa)
        
        # Process response
        self.process_response(response, target_epa)
        
        return response
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current conversation metrics."""
        return {
            'total_deflection': self.state.total_deflection,
            'turn_count': self.state.turn_count,
            'avg_deflection': (
                self.state.total_deflection / max(1, self.state.turn_count)
            ),
            'current_error': self.controller.get_current_error_magnitude(),
            'agent_transient_e': self.state.agent_transient.e,
            'agent_transient_p': self.state.agent_transient.p,
            'agent_transient_a': self.state.agent_transient.a,
        }
