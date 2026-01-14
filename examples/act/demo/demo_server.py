"""
ACT Steering Demo Server

FastAPI server for browser-based ACT conversation steering demo.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Import ACT modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.act.act_core import EPA, get_default_coefficients
from examples.act.identity_manager import (
    get_identity_database, get_modifier_database, create_identity
)
from examples.act.conversation_steering import (
    ACTSteeringEngine, DeflectionControllerConfig, ContextMode, PromptFormatConfig
)
from examples.act.epa_calibration import CalibrationCoefficients


# =============================================================================
# Pydantic Models
# =============================================================================

class IdentityConfig(BaseModel):
    agent_identity: str = "assistant"
    user_identity: str = "person"
    agent_modifiers: List[str] = []
    user_modifiers: List[str] = []

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
        
        # Set up mock functions for demo (real implementation would use actual model)
        self.engine.set_read_epa_function(self._mock_read_epa)
        self.engine.set_steer_function(self._mock_steer)
    
    def _mock_read_epa(self, text: str) -> EPA:
        """Mock EPA reading for demo purposes."""
        # Simple heuristic-based reading
        text_lower = text.lower()
        
        e = 0.0
        p = 0.0
        a = 0.0
        
        # Evaluation heuristics
        positive_words = ['thank', 'great', 'wonderful', 'love', 'appreciate', 'happy', 'good']
        negative_words = ['hate', 'terrible', 'awful', 'angry', 'bad', 'disappointed']
        
        for word in positive_words:
            if word in text_lower:
                e += 0.5
        for word in negative_words:
            if word in text_lower:
                e -= 0.5
        
        # Potency heuristics  
        strong_words = ['demand', 'must', 'require', 'command', 'need']
        weak_words = ['please', 'maybe', 'perhaps', 'beg', 'hope']
        
        for word in strong_words:
            if word in text_lower:
                p += 0.5
        for word in weak_words:
            if word in text_lower:
                p -= 0.3
        
        # Activity heuristics
        active_words = ['urgent', 'quick', 'fast', 'hurry', 'immediately']
        passive_words = ['calm', 'slow', 'relax', 'wait', 'patient']
        
        for word in active_words:
            if word in text_lower:
                a += 0.5
        for word in passive_words:
            if word in text_lower:
                a -= 0.3
        
        # Clamp values
        e = max(-4.3, min(4.3, e))
        p = max(-4.3, min(4.3, p))
        a = max(-4.3, min(4.3, a))
        
        return EPA(e=e, p=p, a=a)
    
    def _mock_steer(self, prompt: str, target_epa: EPA) -> str:
        """Mock steering for demo purposes."""
        # Generate a mock response based on target EPA
        e, p, a = target_epa.e, target_epa.p, target_epa.a
        
        responses = []
        
        # Evaluation-based
        if e > 1.0:
            responses.append("I'm happy to help you with that!")
        elif e < -1.0:
            responses.append("I understand this is frustrating.")
        else:
            responses.append("I see what you mean.")
        
        # Potency-based
        if p > 1.0:
            responses.append("Here's exactly what you should do:")
        elif p < -1.0:
            responses.append("If you'd like, I could suggest...")
        else:
            responses.append("Let me share my thoughts:")
        
        # Activity-based
        if a > 1.0:
            responses.append("Let's get this done right away!")
        elif a < -1.0:
            responses.append("Take your time to consider this carefully.")
        else:
            responses.append("We can work through this together.")
        
        return " ".join(responses)
    
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
        }
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
    
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """Process a chat message and return steered response."""
    if not state.engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
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
