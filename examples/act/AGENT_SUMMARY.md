# ACT (Affect Control Theory) Module Summary

This document provides a comprehensive overview of the `examples/act` directory for coding agents working on this codebase.

---

## Overview

This module implements **Affect Control Theory (ACT)** from David Heise's *Expressive Order* (2007) using **Representation Engineering** techniques. It enables LLMs to read and control conversational affect along three dimensions:

| Dimension | Positive | Negative | Description |
|-----------|----------|----------|-------------|
| **Evaluation (E)** | Good | Bad | Morality, altruism, social desirability |
| **Potency (P)** | Potent | Impotent | Power, authority, dominance |
| **Activity (A)** | Active | Inactive | Energy, speed, liveliness |

---

## Architecture

```
examples/act/
├── Core Python Modules
│   ├── act_core.py            # ACT mathematical functions
│   ├── conversation_steering.py # Steering engine and controller
│   ├── identity_manager.py     # Identity/modifier databases
│   ├── epa_calibration.py      # Raw-to-ACT calibration
│   ├── inauthenticity.py       # Authenticity constraints
│   └── utils.py                # RepE utilities, visualization
│
├── Jupyter Notebooks
│   ├── act_epa_extraction.ipynb    # Extract EPA directions
│   ├── act_epa_calibration.ipynb   # Fit calibration coefficients
│   ├── act_reading_control.ipynb   # Read/control demo
│   ├── act_steering_demo.ipynb     # Full steering demo
│   ├── act_visualizations.ipynb    # t-SNE, LAT scans
│   └── act_conversation_steering.ipynb
│
├── Data Files
│   ├── epa_directions.pkl      # Extracted EPA direction vectors
│   └── epa_calibration.json    # Calibration coefficients
│
└── demo/                       # Web UI demo
    ├── demo_server.py          # FastAPI server
    └── static/                 # Frontend (HTML/JS/CSS)
```

---

## Core Modules

### [act_core.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/act_core.py)
Core ACT mathematical functions:
- `EPA` - Dataclass for EPA profiles with array/dict conversion
- `ACTCoefficients` - Loads impression formation coefficient matrix
- `impression_formation()` - Calculates post-event transient impressions
- `calculate_deflection()` - Measures "stress" between fundamental and transient EPA
- `find_optimal_behavior()` - Optimization to minimize total deflection

### [conversation_steering.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/conversation_steering.py)
Main steering engine:
- `PromptFormatConfig` - Prompt templates for Llama/Mistral models
- `ConversationState` - Tracks agent/user identities and conversation history
- `DeflectionController` - PID-style controller with decay for error correction
- `ACTSteeringEngine` - Main orchestrator that:
  - Reads EPA from user messages
  - Computes optimal response EPA
  - Applies PID adjustment
  - Generates steered responses

### [identity_manager.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/identity_manager.py)
Identity and modifier databases from ACT dictionaries:
- `IdentityDatabase` - Load/search identities (doctor, friend, etc.)
- `ModifierDatabase` - Load/search modifiers (angry, young, etc.)
- `create_identity()` - Apply modifiers to base identities
- `re_identify()` - Find closest identity to a transient EPA

### [epa_calibration.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/epa_calibration.py)
Calibration from raw vector readings to ACT dictionary scale:
- `CalibrationCoefficients` - Stores/loads calibration weights
- `LinearRegressionCalibrator` - Linear mapping (recommended)
- `BehaviorPromptGenerator` - Generates calibration data from behavior templates

### [inauthenticity.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/inauthenticity.py)
Authenticity constraints for steering:
- `calculate_inauthenticity()` - How far expression deviates from identity
- `constrained_optimal_behavior()` - Optimize deflection + authenticity
- `AuthenticityAwareEngine` - Steering with authenticity bounds

### [utils.py](file:///c:/Users/Kyra/Documents/Repos/representation-engineering/examples/act/utils.py)
Representation Engineering utilities:
- `read_epa_scores()` - Read EPA from text using rep_readers
- `make_epa_activations()` - Create steering activation vectors
- `create_epa_dataset()` - Generate contrastive training data
- Visualization: `plot_tsne_epa()`, `plot_lat_scan()`, `plot_per_token_detection()`

---

## Key Data Files

### epa_directions.pkl
Extracted EPA direction vectors. Contains:
```python
{
    'rep_readers': Dict[str, RepReader],  # 'evaluation', 'potency', 'activity'
    'hidden_layers': List[int],           # Layer indices used
    'model_name': str                      # 'meta-llama/Llama-3.1-8B-Instruct'
}
```

### epa_calibration.json
Linear calibration coefficients:
```python
{
    'forward_weights': [[3x3 matrix]],
    'forward_bias': [3-element vector],
    'method': 'linear',
    'r2_scores': {'E': 0.69, 'P': 0.33, 'A': 0.69}
}
```

---

## Typical Workflow

1. **Extract Directions** (`act_epa_extraction.ipynb`)
   ```python
   # Creates contrastive prompts and extracts PCA directions
   # Outputs: epa_directions.pkl
   ```

2. **Calibrate** (`act_epa_calibration.ipynb`)
   ```python
   # Fits linear mapping from raw readings to ACT scale
   # Outputs: epa_calibration.json
   ```

3. **Use Steering Engine**
   ```python
   from examples.act.conversation_steering import ACTSteeringEngine
   from examples.act.epa_calibration import CalibrationCoefficients
   
   calibration = CalibrationCoefficients.load("epa_calibration.json")
   engine = ACTSteeringEngine(
       agent_identity="assistant",
       user_identity="customer",
       calibration=calibration
   )
   
   # Set callbacks for EPA reading and generation
   engine.set_read_epa_function(read_epa_fn)
   engine.set_steer_function(steer_fn)
   
   # Process conversation
   response = engine.chat(user_message)
   ```

---

## Demo Web App

The `demo/` subdirectory contains a FastAPI web application:

```bash
# Run the server
python -m examples.act.demo.demo_server
# Open http://localhost:8000
```

Features:
- Interactive chat with steering visualization
- EPA bar graphs and metrics display
- Configurable generation parameters (temperature, max_tokens, etc.)
- Identity and controller settings

---

## Dependencies

- **Core**: numpy, scipy, pandas, transformers, torch
- **RepE**: The parent `repe` package for representation reading/control
- **Demo**: fastapi, uvicorn

---

## Data Sources

ACT dictionaries loaded from `data/act/`:
- `MTurkInteract_Identities.csv` - Identity EPA profiles
- `MTurkInteract_Modifiers.csv` - Modifier EPA profiles
- `MTurkInteract_Behaviors.csv` - Behavior EPA profiles
- `2010impressionformation.csv` - Impression formation coefficients

---

## Key Concepts

### Deflection
The squared Euclidean distance between fundamental (expected) and transient (actual) EPA. ACT's core principle is that social actors minimize deflection.

### Transient Impressions
Post-event EPA profiles computed via `impression_formation()`. Actor/Object impressions shift based on the behavior performed.

### Re-identification
When transient impressions diverge too far from fundamentals, actors may relabel identities to reduce deflection.

### Inauthenticity
Measures how far an expressed behavior deviates from what's natural for the actor's identity. Used as an optimization constraint.
