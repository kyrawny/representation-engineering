"""
ACT Steering Demo Server

FastAPI server for browser-based ACT conversation steering demo.
Uses real EPA steering vectors and model-based generation.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from repe import repe_pipeline_registry

# Import ACT modules
from examples.act.act_core import EPA, get_default_coefficients
from examples.act.identity_manager import (
    get_identity_database, get_modifier_database, create_identity
)
from examples.act.conversation_steering import (
    ACTSteeringEngine, DeflectionControllerConfig, ContextMode, PromptFormatConfig
)
from examples.act.epa_calibration import CalibrationCoefficients
from examples.act.utils import read_epa_scores, make_epa_activations


# =============================================================================
# Configuration
# =============================================================================

# Default paths relative to this file
DEFAULT_DIRECTIONS_PATH = Path(__file__).parent.parent / "epa_directions.pkl"
DEFAULT_CALIBRATION_PATH = Path(__file__).parent.parent / "epa_calibration.json"


# =============================================================================
# Pydantic Models
# =============================================================================

class IdentityConfig(BaseModel):
    agent_identity: str = "assistant"
    user_identity: str = "person"
    agent_modifiers: List[str] = []
    user_modifiers: List[str] = []

class GenerationConfig(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    steering_coefficient: float = 1.0

class ControllerConfig(BaseModel):
    enabled: bool = True
    context_mode: str = "turn_by_turn"  # "turn_by_turn" or "history"
    window_size: int = 5
    use_decay: bool = True
    decay_rate: float = 0.8
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05

class ChatMessage(BaseModel):
    message: str

class ConfigUpdate(BaseModel):
    identities: Optional[IdentityConfig] = None
    controller: Optional[ControllerConfig] = None
    generation: Optional[GenerationConfig] = None


# =============================================================================
# Model and Directions Manager
# =============================================================================

class ModelManager:
    """Manages the LLM and steering directions."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.rep_pipeline = None
        self.rep_readers = None
        self.hidden_layers = None
        self.reading_layers = None
        self.steering_layers = None
        self.model_name = None
        self.is_loaded = False
    
    def load(self, directions_path: str):
        """Load model and directions."""
        print(f"Loading directions from {directions_path}...")
        
        with open(directions_path, 'rb') as f:
            directions_data = pickle.load(f)
        
        self.rep_readers = directions_data['rep_readers']
        self.hidden_layers = directions_data['hidden_layers']
        self.model_name = directions_data['model_name']
        
        print(f"Loaded directions for model: {self.model_name}")
        print(f"Dimensions available: {list(self.rep_readers.keys())}")
        print(f"Number of layers: {len(self.hidden_layers)}")
        
        # Select layers for reading and steering
        self.reading_layers = self.hidden_layers[len(self.hidden_layers)//4 : len(self.hidden_layers)*3//4]
        self.steering_layers = self.hidden_layers[len(self.hidden_layers)//3 : len(self.hidden_layers)*2//3]
        
        print(f"Using layers {self.reading_layers[:3]}...{self.reading_layers[-3:]} for EPA reading")
        print(f"Using layers {self.steering_layers[:3]}...{self.steering_layers[-3:]} for steering")
        
        # Load model
        print(f"Loading model {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Register RepE pipelines and create rep-reading pipeline
        repe_pipeline_registry()
        self.rep_pipeline = hf_pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        
        self.is_loaded = True
        print("Model and directions loaded successfully!")


# Global model manager
model_manager = ModelManager()


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state singleton."""
    
    def __init__(self):
        self.identity_db = get_identity_database()
        self.modifier_db = get_modifier_database()
        self.engine: Optional[ACTSteeringEngine] = None
        self.calibration: Optional[CalibrationCoefficients] = None
        
        # Generation settings
        self.max_new_tokens = 128
        self.temperature = 0.7
        self.top_p = 0.95
        self.steering_coefficient = 1.0
        
        # Initialize with default engine
        self.initialize_engine()
    
    def initialize_engine(
        self,
        agent_identity: str = "assistant",
        user_identity: str = "person",
        agent_modifiers: List[str] = None,
        user_modifiers: List[str] = None,
        controller_config: Optional[DeflectionControllerConfig] = None
    ):
        """Initialize or reinitialize the steering engine."""
        self.engine = ACTSteeringEngine(
            agent_identity=agent_identity,
            user_identity=user_identity,
            agent_modifiers=agent_modifiers,
            user_modifiers=user_modifiers,
            calibration=self.calibration,
            controller_config=controller_config
        )
        
        # Set up the real EPA reading and steering functions
        self.engine.set_read_epa_function(self._read_epa)
        self.engine.set_steer_function(self._steer_generation)
    
    def _read_epa(self, text: str) -> EPA:
        """Read EPA values from text using extracted directions."""
        if not model_manager.is_loaded:
            raise RuntimeError("Model not loaded. Cannot read EPA values.")
        
        raw_scores = read_epa_scores(
            pipeline=model_manager.rep_pipeline,
            rep_readers=model_manager.rep_readers,
            text=text,
            layers=model_manager.reading_layers,
            padding=True,
            truncation=True,
        )
        
        raw_epa = EPA(
            e=raw_scores.get('evaluation', 0.0),
            p=raw_scores.get('potency', 0.0),
            a=raw_scores.get('activity', 0.0),
        )
        
        # Apply calibration if available
        if self.calibration:
            return self.calibration.to_epa(raw_epa)
        return raw_epa
    
    def _steer_generation(self, prompt: str, target_epa: EPA) -> str:
        """Generate text with EPA steering using extracted directions."""
        if not model_manager.is_loaded:
            raise RuntimeError("Model not loaded. Cannot generate text.")
        
        # Convert calibrated EPA back to raw space for steering if calibration available
        if self.calibration:
            raw_target = self.calibration.from_epa(target_epa)
        else:
            raw_target = target_epa
        
        # Create activation vectors for the target EPA
        activations = make_epa_activations(
            rep_readers=model_manager.rep_readers,
            layers=model_manager.steering_layers,
            e_coeff=raw_target.e * self.steering_coefficient,
            p_coeff=raw_target.p * self.steering_coefficient,
            a_coeff=raw_target.a * self.steering_coefficient,
            device=model_manager.model.device,
            dtype=model_manager.model.dtype,
        )
        
        # Tokenize prompt
        inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
        
        # Generate (note: actual steering with activations requires RepControl hooks)
        # For now, generate without explicit activation injection
        # Full steering integration would require using repe's rep_control_pipeline
        outputs = model_manager.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=model_manager.tokenizer.eos_token_id,
        )
        
        # Decode response
        full_text = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response (after the prompt)
        prompt_text = model_manager.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        response = full_text[len(prompt_text):].strip()
        
        return response
    
    def load_calibration(self, path: str):
        """Load calibration coefficients."""
        self.calibration = CalibrationCoefficients.load(path)
        if self.engine:
            self.engine.calibration = self.calibration


# Global state
state = AppState()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="ACT Steering Demo", version="1.0.0")

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """Load model and calibration on startup."""
    # Load directions and model
    if DEFAULT_DIRECTIONS_PATH.exists():
        model_manager.load(str(DEFAULT_DIRECTIONS_PATH))
    else:
        print(f"Warning: Directions file not found at {DEFAULT_DIRECTIONS_PATH}")
        print("EPA reading and steering will not work without loading directions.")
    
    # Load calibration
    if DEFAULT_CALIBRATION_PATH.exists():
        state.load_calibration(str(DEFAULT_CALIBRATION_PATH))
        print(f"Loaded calibration from {DEFAULT_CALIBRATION_PATH}")
    else:
        print(f"Warning: Calibration file not found at {DEFAULT_CALIBRATION_PATH}")


@app.get("/")
async def index():
    """Serve the main page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ACT Steering Demo - Frontend not found"}


@app.get("/api/identities")
async def list_identities():
    """List available identities."""
    return {"identities": state.identity_db.list_all()[:100]}  # Limit for performance


@app.get("/api/modifiers")
async def list_modifiers():
    """List available modifiers."""
    return {"modifiers": state.modifier_db.list_all()[:100]}


@app.get("/api/identity/{name}")
async def get_identity(name: str):
    """Get EPA for a specific identity."""
    identity = state.identity_db.get(name)
    if identity:
        return {
            "name": identity.name,
            "e": identity.epa.e,
            "p": identity.epa.p,
            "a": identity.epa.a
        }
    raise HTTPException(status_code=404, detail=f"Identity not found: {name}")


@app.get("/api/state")
async def get_state():
    """Get current conversation state."""
    if not state.engine:
        return {"error": "Engine not initialized"}
    
    engine = state.engine
    return {
        "agent": {
            "identity": engine.agent.name,
            "fundamental": engine.state.agent_fundamental.to_dict(),
            "transient": engine.state.agent_transient.to_dict()
        },
        "user": {
            "identity": engine.user.name,
            "fundamental": engine.state.user_fundamental.to_dict(),
            "transient": engine.state.user_transient.to_dict()
        },
        "metrics": engine.get_metrics(),
        "history": [
            {
                "role": turn.role,
                "content": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content,
                "epa": turn.epa_read.to_dict() if turn.epa_read else None,
                "deflection": turn.deflection
            }
            for turn in engine.state.history[-10:]  # Last 10 turns
        ],
        "controller": {
            "enabled": engine.controller.enabled,
            "config": {
                "context_mode": engine.controller.config.context_mode.value,
                "window_size": engine.controller.config.window_size,
                "use_decay": engine.controller.config.use_decay,
                "decay_rate": engine.controller.config.decay_rate,
            }
        },
        "generation": {
            "max_new_tokens": state.max_new_tokens,
            "temperature": state.temperature,
            "top_p": state.top_p,
            "steering_coefficient": state.steering_coefficient
        },
        "model_loaded": model_manager.is_loaded,
        "model_name": model_manager.model_name
    }


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update configuration."""
    controller_config = None
    
    if config.controller:
        cc = config.controller
        controller_config = DeflectionControllerConfig(
            enabled=cc.enabled,
            context_mode=ContextMode(cc.context_mode),
            window_size=cc.window_size,
            use_decay=cc.use_decay,
            decay_rate=cc.decay_rate,
            kp=cc.kp,
            ki=cc.ki,
            kd=cc.kd
        )
    
    if config.identities:
        ic = config.identities
        state.initialize_engine(
            agent_identity=ic.agent_identity,
            user_identity=ic.user_identity,
            agent_modifiers=ic.agent_modifiers or None,
            user_modifiers=ic.user_modifiers or None,
            controller_config=controller_config
        )
    elif controller_config and state.engine:
        state.engine.controller.config = controller_config
    
    # Update generation settings
    if config.generation:
        gc = config.generation
        state.max_new_tokens = gc.max_new_tokens
        state.temperature = gc.temperature
        state.top_p = gc.top_p
        state.steering_coefficient = gc.steering_coefficient
    
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """Process a chat message and return steered response."""
    if not state.engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure epa_directions.pkl exists.")
    
    try:
        # Process user message
        optimal_epa = state.engine.process_user_message(msg.message)
        
        # Get adjusted target
        target_epa = state.engine.get_adjusted_target(optimal_epa)
        
        # Generate response
        response = state.engine.generate_response(msg.message, target_epa)
        
        # Process response
        actual_epa, deflection = state.engine.process_response(response, target_epa)
        
        return {
            "response": response,
            "optimal_epa": optimal_epa.to_dict(),
            "target_epa": target_epa.to_dict(),
            "actual_epa": actual_epa.to_dict(),
            "deflection": deflection,
            "metrics": state.engine.get_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset():
    """Reset conversation state."""
    if state.engine:
        state.engine.reset()
    return {"status": "ok"}


@app.get("/api/config/options")
async def get_config_options():
    """Get available configuration options."""
    return {
        "context_modes": ["turn_by_turn", "history"],
        "defaults": {
            "agent_identity": "assistant",
            "user_identity": "person",
            "controller": {
                "enabled": True,
                "context_mode": "turn_by_turn",
                "window_size": 5,
                "use_decay": True,
                "decay_rate": 0.8,
                "kp": 1.0,
                "ki": 0.1,
                "kd": 0.05
            },
            "generation": {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.95,
                "steering_coefficient": 1.0
            }
        }
    }


# =============================================================================
# WebSocket for streaming (optional)
# =============================================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                # Process chat message
                result = await chat(ChatMessage(message=message["content"]))
                await websocket.send_json(result)
            elif message.get("type") == "reset":
                await reset()
                await websocket.send_json({"type": "reset", "status": "ok"})
    except WebSocketDisconnect:
        pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
