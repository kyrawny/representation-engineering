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
    get_response_epa_for_deflection_minimization, total_deflection
)
from .identity_manager import (
    Identity, Modifier, ModifiedIdentity, 
    IdentityDatabase, get_identity_database,
    create_identity
)
from .epa_calibration import CalibrationCoefficients
from .utils import steer_generation, read_epa


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


class DecayTiming(Enum):
    """When to apply transient decay."""
    BEFORE_TURN = "before_turn"  # Apply decay before processing each turn
    AFTER_TURN = "after_turn"    # Apply decay after processing each turn


@dataclass
class DeflectionControllerConfig:
    """Configuration for the deflection controller."""
    
    enabled: bool = True
    
    # Context settings
    context_mode: ContextMode = ContextMode.TURN_BY_TURN
    window_size: int = 5  # Number of past turns to consider (for HISTORY mode)
    context_window_size: int = 0  # Number of previous messages to include in prompts (0 = none)
    
    # Decay settings
    use_decay: bool = True
    decay_rate: float = 0.8  # Rate at which transients decay toward fundamentals (higher = slower decay)
    decay_timing: DecayTiming = DecayTiming.AFTER_TURN  # When to apply decay
    
    # PID gains (applied to total deflection scalar)
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.05  # Derivative gain


class DeflectionController:
    """
    PID-style error correction for deflection minimization.
    
    Tracks total system deflection (across actor, behavior, object) over time
    and adjusts future targets to compensate. Also handles transient decay
    toward fundamental sentiments.
    """
    
    def __init__(self, config: Optional[DeflectionControllerConfig] = None):
        self.config = config or DeflectionControllerConfig()
        
        # Error history for PID (total deflection scalar values)
        self.deflection_history: deque = deque(maxlen=100)
        self.previous_deflection: Optional[float] = None
        self.integral_deflection: float = 0.0
        
        # Per-dimension error tracking for EPA adjustment
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
        self.deflection_history.clear()
        self.previous_deflection = None
        self.integral_deflection = 0.0
        
        self.error_history.clear()
        self.previous_error = None
        self.integral_error = np.zeros(3)
    
    def _compute_weighted_integral(self) -> float:
        """Compute weighted integral of deflection errors with decay."""
        if not self.deflection_history:
            return 0.0
        
        integral = 0.0
        n_errors = len(self.deflection_history)
        
        for i, deflection in enumerate(self.deflection_history):
            if self.config.use_decay:
                # More recent errors have higher weight
                turns_ago = n_errors - 1 - i
                weight = self.config.decay_rate ** turns_ago
            else:
                weight = 1.0
            integral += weight * deflection
        
        return integral
    
    def record_deflection(
        self,
        total_defl: float,
        target_epa: EPA,
        actual_epa: EPA,
    ):
        """
        Record a deflection measurement for PID control.
        
        Args:
            total_defl: Total system deflection (actor + object)
            target_epa: What we targeted for the behavior
            actual_epa: What we actually achieved
        """
        # Record total deflection for PID
        self.deflection_history.append(total_defl)
        
        # Record per-dimension error for EPA adjustment
        optimal_arr = target_epa.to_array()
        actual_arr = actual_epa.to_array()
        error = optimal_arr - actual_arr
        self.error_history.append(error)
        
        # Apply window if using history mode
        if self.config.context_mode == ContextMode.HISTORY:
            window = self.config.window_size
            if len(self.deflection_history) > window:
                recent = list(self.deflection_history)[-window:]
                self.deflection_history.clear()
                self.deflection_history.extend(recent)
            if len(self.error_history) > window:
                recent_errors = list(self.error_history)[-window:]
                self.error_history.clear()
                self.error_history.extend(recent_errors)
    
    def compute_adjustment(
        self,
        optimal_epa: EPA,
    ) -> EPA:
        """
        Compute PID-adjusted target EPA based on deflection history.
        
        Args:
            optimal_epa: ACT-computed optimal behavior EPA
            
        Returns:
            Adjusted target EPA for next response
        """
        if not self.config.enabled or not self.deflection_history:
            return optimal_epa
        
        if not self.error_history:
            return optimal_epa
        
        # Current error (per-dimension)
        error = self.error_history[-1] if self.error_history else np.zeros(3)
        
        # Compute PID terms based on total deflection
        current_deflection = self.deflection_history[-1] if self.deflection_history else 0.0
        
        # Proportional term (scaled by current deflection relative to history)
        # Higher deflection = larger adjustment
        p_scale = current_deflection / max(1.0, np.mean(list(self.deflection_history)) + 1e-8)
        p_term = self.config.kp * p_scale * error
        
        # Integral term (with decay on deflection)
        self.integral_deflection = self._compute_weighted_integral()
        i_scale = min(2.0, self.integral_deflection / max(1.0, len(self.deflection_history)))
        
        # Compute weighted integral of errors
        integral_error = np.zeros(3)
        n_errors = len(self.error_history)
        for i, err in enumerate(self.error_history):
            if self.config.use_decay:
                turns_ago = n_errors - 1 - i
                weight = self.config.decay_rate ** turns_ago
            else:
                weight = 1.0
            integral_error += weight * err
        self.integral_error = integral_error
        
        i_term = self.config.ki * i_scale * self.integral_error
        
        # Derivative term
        if self.previous_error is not None:
            d_term = self.config.kd * (error - self.previous_error)
        else:
            d_term = np.zeros(3)
        
        self.previous_error = error.copy()
        self.previous_deflection = current_deflection
        
        # Compute adjusted target
        optimal_arr = optimal_epa.to_array()
        adjustment = p_term + i_term + d_term
        adjusted = optimal_arr + adjustment
        
        # Clip to reasonable EPA bounds
        adjusted = np.clip(adjusted, -4.3, 4.3)
        
        return EPA.from_array(adjusted)
    
    def get_current_deflection(self) -> float:
        """Get the most recent total deflection."""
        if not self.deflection_history:
            return 0.0
        return float(self.deflection_history[-1])
    
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
        # Built-in steering/reading support
        model = None,
        tokenizer = None,
        rep_pipeline = None,
        rep_readers: Optional[Dict] = None,
        reading_layers: Optional[List[int]] = None,
        steering_layers: Optional[List[int]] = None,
        rep_control_pipeline = None,
        steering_coefficient: float = 1.0,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
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
            model: Language model for built-in steering (optional)
            tokenizer: Tokenizer for built-in steering (optional)
            rep_pipeline: RepReadingPipeline for built-in EPA reading (optional)
            rep_readers: Dict mapping dimension to RepReader for reading/steering (optional)
            reading_layers: Layer indices for EPA reading (optional)
            steering_layers: Layer indices for steering (optional)
            steering_coefficient: Multiplier for steering strength (default 1.0)
            max_new_tokens: Maximum tokens to generate (default 128)
            temperature: Sampling temperature (default 0.7)
            top_p: Top-p sampling parameter (default 0.95)
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
        
        # Built-in steering/reading components
        self.model = model
        self.tokenizer = tokenizer
        self.rep_pipeline = rep_pipeline
        self.rep_readers = rep_readers
        self.reading_layers = reading_layers
        self.steering_layers = steering_layers
        self.rep_control_pipeline = rep_control_pipeline
        self.steering_coefficient = steering_coefficient
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Callbacks for EPA reading/steering (set externally for custom implementations)
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
    
    def read_epa(
        self,
        text: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        context_window: Optional[int] = None
    ) -> EPA:
        """
        Read EPA from text, applying calibration if available.
        
        Uses built-in read_epa() if rep_pipeline/rep_readers/reading_layers are configured,
        otherwise falls back to the custom read function callback.
        
        Args:
            text: Text to read EPA from
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt (e.g., previous message)
            context_window: Number of previous messages to include (uses config if None)
            
        Returns:
            EPA reading (calibrated if calibration is available)
        """
        # Build context if context window is requested
        window_size = context_window if context_window is not None else self.controller.config.context_window_size
        context_str = ""
        if window_size > 0:
            context_str = self._format_context_window(window_size)
        
        # Use built-in reading if rep_pipeline is configured
        if self.rep_pipeline is not None and self.rep_readers is not None:
            if self.reading_layers is None:
                raise RuntimeError("reading_layers must be set when using built-in EPA reading.")
            
            raw_epa = read_epa(
                pipeline=self.rep_pipeline,
                rep_readers=self.rep_readers,
                text=text,
                layers=self.reading_layers,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                padding=True,
                truncation=True,
            )
        elif self._read_epa_fn is not None:
            # Fall back to custom read function
            raw_epa = self._read_epa_fn(text)
        else:
            raise RuntimeError(
                "EPA reading not configured. Either set rep_pipeline/rep_readers/reading_layers "
                "in the constructor, or call set_read_epa_function()."
            )
        
        if self.calibration:
            return self.calibration.to_epa(raw_epa)
        return raw_epa
    
    def _format_context_window(self, window_size: int) -> str:
        """
        Format recent conversation history for inclusion in prompts.
        
        Creates alternating user/assistant messages for context.
        Only includes previous messages, not the current one being processed.
        
        Args:
            window_size: Number of previous messages to include
            
        Returns:
            Formatted context string
        """
        if window_size <= 0 or not self.state.history:
            return ""
        
        recent = self.state.get_recent_turns(window_size)
        parts = []
        
        for turn in recent:
            if turn.role == "user":
                parts.append(f"{self.prompt_format.user_start}{turn.content}{self.prompt_format.user_end}")
            else:
                parts.append(f"{self.prompt_format.assistant_start}{turn.content}{self.prompt_format.assistant_end}")
        
        return "".join(parts)
    
    def get_steered_read_prompts(self) -> Tuple[str, Optional[str]]:
        """
        Get the system and user prompts for EPA reading in a steered conversation.
        
        Returns:
            Tuple of (system_prompt, user_prompt) where user_prompt is the previous message
        """
        system_prompt = f"You are a {self.user.name} speaking with a {self.agent.name}."
        
        # Get the previous message (last user turn)
        user_prompt = None
        for turn in reversed(self.state.history):
            if turn.role == "user":
                user_prompt = turn.content
                break
        
        return system_prompt, user_prompt
    
    def compute_optimal_response_epa(self) -> EPA:
        """
        Compute optimal EPA for agent's response based on cumulative transient impressions.
        
        Uses the current transient impressions from the entire conversation,
        not just the previous message.
        
        Returns:
            Optimal behavior EPA for agent's response
        """
        return find_optimal_behavior(
            actor_fundamental=self.state.agent_fundamental,
            object_fundamental=self.state.user_fundamental,
            actor_transient=self.state.agent_transient,
            object_transient=self.state.user_transient,
            coefficients=self.coefficients
        )
    
    def apply_transient_decay(self):
        """
        Apply decay to transient impressions, moving them toward fundamentals.
        
        This implements the ACT concept that transient impressions gradually
        fade back toward fundamental sentiments over time.
        """
        if not self.controller.config.use_decay:
            return
        
        decay_rate = self.controller.config.decay_rate
        
        # Decay agent transient toward fundamental
        agent_transient_arr = self.state.agent_transient.to_array()
        agent_fundamental_arr = self.state.agent_fundamental.to_array()
        agent_decayed = decay_rate * agent_transient_arr + (1 - decay_rate) * agent_fundamental_arr
        self.state.agent_transient = EPA.from_array(agent_decayed)
        
        # Decay user transient toward fundamental
        user_transient_arr = self.state.user_transient.to_array()
        user_fundamental_arr = self.state.user_fundamental.to_array()
        user_decayed = decay_rate * user_transient_arr + (1 - decay_rate) * user_fundamental_arr
        self.state.user_transient = EPA.from_array(user_decayed)
    
    def process_user_message(self, message: str) -> EPA:
        """
        Process a user message and return optimal response EPA.
        
        Args:
            message: User's message text
            
        Returns:
            Target EPA for agent's response
        """
        # Apply decay before turn if configured
        if self.controller.config.decay_timing == DecayTiming.BEFORE_TURN:
            self.apply_transient_decay()
        
        # Read EPA of user message
        user_epa = self.read_epa(message)
        
        # Record user turn
        self.state.add_turn(
            role="user",
            content=message,
            epa_read=user_epa
        )
        
        # Update transients based on user's action
        post = impression_formation(
            actor=self.state.user_transient,
            behavior=user_epa,
            obj=self.state.agent_transient,
            coefficients=self.coefficients
        )
        self.state.user_transient = post['actor']
        self.state.agent_transient = post['object']
        
        # Compute optimal response EPA using cumulative transients
        optimal_epa = self.compute_optimal_response_epa()
        
        return optimal_epa
    
    def _sigmoid_clamp(self, epa: EPA, scale: float = 1.0) -> EPA:
        """
        Apply sigmoid-like clamping to EPA values, tapering to [-1, 1].
        
        Uses tanh which naturally produces a smooth S-curve that asymptotes
        at -1 and +1. The scale parameter controls how quickly values
        approach the limits (higher = sharper transition).
        
        Args:
            epa: EPA values to clamp
            scale: How sharply to transition (default 2.0 means values at ±1 
                   are already ~76% of the way to the limit)
        
        Returns:
            EPA with values smoothly clamped to [-1, 1]
        """
        return EPA(
            e=float(np.tanh(epa.e / scale)),
            p=float(np.tanh(epa.p / scale)),
            a=float(np.tanh(epa.a / scale))
        )
    
    def generate_response(self, prompt: str, target_epa: EPA) -> str:
        """
        Generate a steered response.
        
        Uses built-in steer_generation() if model/tokenizer/rep_readers are configured,
        otherwise falls back to the custom steer function callback.
        
        Args:
            prompt: Prompt for the LLM
            target_epa: Target EPA for steering
            
        Returns:
            Generated response text
        """
        # Convert calibrated target back to raw space if needed
        if self.calibration:
            raw_target = self.calibration.from_epa(target_epa)
            print(f"Target EPA: {target_epa}")
            print(f"Raw Target (before clamping): {raw_target}")
        else:
            raw_target = target_epa
        
        # Apply sigmoid clamping to taper values to [-1, 1]
        raw_target = self._sigmoid_clamp(raw_target)
        print(f"Raw Target (after clamping): {raw_target}")

        print(prompt)
        
        # Use built-in steering if model is configured
        if self.model is not None and self.tokenizer is not None and self.rep_readers is not None:
            if self.steering_layers is None:
                raise RuntimeError("steering_layers must be set when using built-in steering.")
            
            return steer_generation(
                model=self.model,
                tokenizer=self.tokenizer,
                rep_readers=self.rep_readers,
                layers=self.steering_layers,
                prompt=prompt,
                target_epa=(raw_target.e, raw_target.p, raw_target.a),
                steering_coefficient=self.steering_coefficient,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                rep_control_pipeline=self.rep_control_pipeline,
            )
        
        # Fall back to custom steer function
        if self._steer_fn is None:
            raise RuntimeError(
                "Steering not configured. Either set model/tokenizer/rep_readers/steering_layers "
                "in the constructor, or call set_steer_function()."
            )
        
        return self._steer_fn(prompt, raw_target)
    
    def process_response(self, response: str, target_epa: EPA) -> Tuple[EPA, float]:
        """
        Process a generated response.
        
        Calculates total system deflection (actor + object) based on the post-event
        transient impressions compared to fundamental sentiments.
        
        Args:
            response: Generated response text
            target_epa: What we targeted
            
        Returns:
            Tuple of (actual_epa, total_deflection)
        """
        # Read actual EPA achieved
        actual_epa = self.read_epa(response)
        
        # Calculate post-event transients with the actual behavior
        post = impression_formation(
            actor=self.state.agent_transient,
            behavior=actual_epa,
            obj=self.state.user_transient,
            coefficients=self.coefficients
        )
        
        # Calculate total system deflection (actor + object vs fundamentals)
        deflection = total_deflection(
            actor_fundamental=self.state.agent_fundamental,
            actor_transient=post['actor'],
            behavior_fundamental=None,  # No specific behavior fundamental
            behavior_transient=post['behavior'],
            object_fundamental=self.state.user_fundamental,
            object_transient=post['object']
        )
        
        # Update controller with deflection and EPA error
        if self.controller.enabled:
            self.controller.record_deflection(deflection, target_epa, actual_epa)
        
        # Record assistant turn
        self.state.add_turn(
            role="assistant",
            content=response,
            epa_read=actual_epa,
            epa_target=target_epa,
            deflection=deflection
        )
        
        # Update transients based on agent's response
        self.state.agent_transient = post['actor']
        self.state.user_transient = post['object']
        
        # Apply decay after turn if configured
        if self.controller.config.decay_timing == DecayTiming.AFTER_TURN:
            self.apply_transient_decay()
        
        return actual_epa, deflection
    
    def get_adjusted_target(self, optimal_epa: EPA) -> EPA:
        """Get PID-adjusted target EPA based on controller state."""
        if not self.controller.enabled or not self.controller.deflection_history:
            return optimal_epa
        
        return self.controller.compute_adjustment(optimal_epa)
    
    def chat(
        self,
        user_message: str,
        prompt: Optional[str] = None,
        context_window: Optional[int] = None
    ) -> str:
        """
        Complete chat turn: process user message, compute optimal EPA, generate response.
        
        Args:
            user_message: User's message
            prompt: Optional custom prompt (uses default formatting if None)
            context_window: Number of previous messages to include in prompt (uses config if None)
            
        Returns:
            Generated response
        """
        # Build context BEFORE processing user message (so current msg isn't in history yet)
        window_size = context_window if context_window is not None else self.controller.config.context_window_size
        context = self._format_context_window(window_size) if window_size > 0 else ""
        
        # Process user message to get optimal EPA
        optimal_epa = self.process_user_message(user_message)
        
        # Adjust target based on controller
        target_epa = self.get_adjusted_target(optimal_epa)
        
        # Format prompt if not provided
        if prompt is None:
            system = f"You are a {self.agent.name} speaking with a {self.user.name}."
            
            # Format prompt with context
            prompt = self.prompt_format.format_prompt(system, user_message)
            if context:
                # Insert context before the current user message
                # This is a simplified approach - for production, consider a more robust insertion
                system_end_idx = prompt.find(self.prompt_format.system_end) + len(self.prompt_format.system_end)
                prompt = prompt[:system_end_idx] + context + prompt[system_end_idx:]
        
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
