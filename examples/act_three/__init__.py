"""
act_three — Affect Control Theory + Representation Engineering.

This package provides a complete pipeline for:
1. Reading EPA (Evaluation, Potency, Activity) from text using
   representation reading with calibrated per-layer weights.
2. Computing optimal response EPA via ACT deflection minimisation.
3. Steering LLM generation toward target EPA values.

Key classes:
    - ``EPA`` — Evaluation, Potency, Activity dataclass
    - ``ACTCoefficients`` — impression formation coefficients
    - ``EPAReader`` — calibrated EPA reading from text
    - ``EPASteerer`` — representation-engineering-based steering
    - ``ACTPipeline`` — end-to-end orchestrator

Quick start::

    from examples.act_three import ACTPipeline, EPA

    pipe = ACTPipeline(
        agent_identity=EPA(e=1.5, p=1.0, a=0.5),
        user_identity=EPA(e=1.0, p=0.5, a=0.3),
    )
    pipe.load_model()
    pipe.load_directions("epa_directions.pkl")
    pipe.setup_reader("epa_reading_tuning_v2_results.json")
    pipe.setup_steerer()

    response = pipe.process_message("Hello, how are you?")
"""

# Core ACT types and functions
from .act_core import (
    EPA,
    ACTCoefficients,
    calculate_deflection,
    find_optimal_behavior,
    get_default_coefficients,
    get_response_epa_for_deflection_minimization,
    impression_formation,
    predict_emotion,
    total_deflection,
)

# Prompt formatting
from .prompt_formatting import (
    DIMENSION_NAMES,
    DIM_KEY,
    EPA_DIMENSIONS,
    format_for_reading,
    format_llama3_prompt,
    get_epa_extraction_template,
)

# Model registry and model-agnostic prompt formatting
from .model_registry import (
    MODEL_REGISTRY,
    format_chat_prompt,
    format_for_reading_generic,
    get_model_config,
    get_short_name,
)

# Dataset creation
from .dataset import (
    create_all_epa_datasets,
    create_epa_dataset,
    load_act_data,
)

# Direction extraction
from .direction_extraction import (
    extract_epa_directions,
    load_directions,
    save_directions,
)

# Direction validation & layer selection
from .direction_validation import (
    LayerSelectionResult,
    compute_layer_accuracies,
    select_steering_layers,
)

# Calibrated EPA reading
from .epa_reader import (
    DimensionConfig,
    EPAReader,
    EPAReaderConfig,
    compute_phase1_correlations,
    fit_calibration,
    select_layers_elasticnet,
    select_layers_greedy,
    select_layers_ridge,
    select_layers_sffs,
    select_layers_simple,
)

# EPA steering
from .epa_steerer import (
    EPASteerer,
    make_epa_activations,
    steer_generation,
)

# Calibration
from .epa_calibration import (
    BehaviorPromptGenerator,
    CalibrationCoefficients,
    EPACalibrator,
    LinearRegressionCalibrator,
    AffineCalibrator,
    FineTuningCalibrator,
    calibrate_from_behaviors,
    get_calibrator,
)

# End-to-end pipeline
from .pipeline import ACTPipeline

__all__ = [
    # Core types
    "EPA",
    "ACTCoefficients",
    # ACT functions
    "impression_formation",
    "calculate_deflection",
    "total_deflection",
    "find_optimal_behavior",
    "predict_emotion",
    "get_response_epa_for_deflection_minimization",
    "get_default_coefficients",
    # Prompt formatting
    "format_llama3_prompt",
    "format_chat_prompt",
    "format_for_reading",
    "format_for_reading_generic",
    "get_epa_extraction_template",
    "EPA_DIMENSIONS",
    "DIMENSION_NAMES",
    "DIM_KEY",
    # Model registry
    "MODEL_REGISTRY",
    "get_model_config",
    "get_short_name",
    # Dataset
    "load_act_data",
    "create_epa_dataset",
    "create_all_epa_datasets",
    # Direction extraction
    "extract_epa_directions",
    "save_directions",
    "load_directions",
    # Direction validation
    "LayerSelectionResult",
    "compute_layer_accuracies",
    "select_steering_layers",
    # EPA reading
    "EPAReader",
    "EPAReaderConfig",
    "DimensionConfig",
    "compute_phase1_correlations",
    "fit_calibration",
    "select_layers_simple",
    "select_layers_greedy",
    "select_layers_sffs",
    "select_layers_ridge",
    "select_layers_elasticnet",
    # EPA steering
    "EPASteerer",
    "make_epa_activations",
    "steer_generation",
    # Calibration
    "BehaviorPromptGenerator",
    "CalibrationCoefficients",
    "EPACalibrator",
    "LinearRegressionCalibrator",
    "AffineCalibrator",
    "FineTuningCalibrator",
    "calibrate_from_behaviors",
    "get_calibrator",
    # Pipeline
    "ACTPipeline",
]
