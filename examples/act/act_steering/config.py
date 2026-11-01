"""Configuration objects for the ACT deflection-steering package.

This module is deliberately dependency-free (no ``torch``, no
``transformers``) so the pure-logic core of the package can be imported and
tested on CPU.

All configuration is expressed as dataclasses with defaults, so
``ExperimentConfig()`` always yields a runnable configuration. Paths are
resolved relative to the repository root and can be overridden per
experiment; no absolute paths are hardcoded anywhere else in the package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

#: Root of the representation-engineering repository.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

#: Directory of this package (``examples/act``).
PACKAGE_DIR: Path = REPO_ROOT / "examples" / "act"

#: Default ACT data directory (MTurk-Interact dictionaries, equation set).
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "act"

#: Default location for extracted direction artifacts.
DEFAULT_ARTIFACTS_DIR: Path = PACKAGE_DIR / "artifacts"

#: Default location for evaluation result files and trace logs.
DEFAULT_RESULTS_DIR: Path = PACKAGE_DIR / "results"

#: Default generation model (Llama family; in-loop contrast mode is Llama/Mistral only).
DEFAULT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

#: Default RNG seed for all stochastic operations.
DEFAULT_SEED: int = 42

#: File names inside the ACT data directory.
COEFFICIENTS_FILE = "2010impressionformation.csv"
IDENTITIES_FILE = "MTurkInteract_Identities.csv"
BEHAVIORS_FILE = "MTurkInteract_Behaviors.csv"
MODIFIERS_FILE = "MTurkInteract_Modifiers.csv"
USER_INPUTS_FILE = "user_inputs.json"
TRUNCATED_OUTPUTS_FILE = "all_truncated_outputs.json"


@dataclass(frozen=True)
class DataPaths:
    """Filesystem locations for ACT data, artifacts, and results.

    Attributes:
        data_dir: Directory containing the MTurk-Interact dictionaries and
            the impression-formation coefficient set.
        artifacts_dir: Directory for extracted steering/reader direction
            artifacts (created on demand).
        results_dir: Directory for JSON evaluation results and trace logs
            (created on demand).
    """

    data_dir: Path = DEFAULT_DATA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR

    @property
    def coefficients_file(self) -> Path:
        """Return the path to the impression-formation coefficient CSV."""
        return self.data_dir / COEFFICIENTS_FILE

    @property
    def identities_file(self) -> Path:
        """Return the path to the MTurk-Interact identities CSV."""
        return self.data_dir / IDENTITIES_FILE

    @property
    def behaviors_file(self) -> Path:
        """Return the path to the MTurk-Interact behaviours CSV."""
        return self.data_dir / BEHAVIORS_FILE

    @property
    def modifiers_file(self) -> Path:
        """Return the path to the MTurk-Interact modifiers CSV."""
        return self.data_dir / MODIFIERS_FILE

    @property
    def user_inputs_file(self) -> Path:
        """Return the path to the conversational user-input JSON list."""
        return self.data_dir / USER_INPUTS_FILE

    @property
    def truncated_outputs_file(self) -> Path:
        """Return the path to the truncated-continuation JSON list."""
        return self.data_dir / TRUNCATED_OUTPUTS_FILE

    def ensure_dirs(self) -> None:
        """Create the artifact and results directories if they do not exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Generation-model settings shared by all entry points.

    Attributes:
        name: HuggingFace model identifier.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        dtype: Torch dtype name (``"float16"`` on CUDA, ``"float32"`` on CPU).
        max_new_tokens: Maximum response length per turn.
        max_context_length: Maximum context length for tokenization.
    """

    name: str = DEFAULT_MODEL
    device: str = "cuda"
    dtype: str = "float16"
    max_new_tokens: int = 256
    max_context_length: int = 2048


@dataclass
class SteeringConfig:
    """Actuation settings for the deflection-contrast steerer.

    Attributes:
        layers: Control layer indices; empty means auto-select the last
            decoder layer.
        contrast_tokens: Number of tail positions averaged when extracting
            the contrast direction.
        alpha: Base steering strength applied at the reference deflection.
        alpha_min: Lower clamp of the per-turn proportional gain.
        alpha_max: Upper clamp; must stay below the degeneration threshold
            measured by the dose-response sweep.
        kappa: Proportional gain constant mapping ``U_pre`` to alpha.
        u_ref: Reference deflection value that maps to the base alpha.
        mode: ``"static"`` (cached direction), ``"reextract"`` (re-extract on
            context change), or ``"inloop"`` (per-step contrast; Llama/Mistral
            only, no KV cache).
        token_pos: Where the controller is applied (``"end"`` for tail
            control of the generated response).
    """

    layers: list[int] = field(default_factory=list)
    contrast_tokens: int = 8
    alpha: float = 0.5
    alpha_min: float = 0.0
    alpha_max: float = 1.5
    kappa: float = 1.0
    u_ref: float = 10.0
    mode: str = "static"
    token_pos: str = "end"


@dataclass
class MonitorConfig:
    """Settings for the per-token deflection monitor.

    Attributes:
        reader_layers: Layers at which the reader direction is read; empty
            means auto-select.
        threshold: Deflection score above which a per-token trigger fires.
        clip_sigma: Outlier clip distance in standard deviations
            (honesty-frontier pattern: median +/- clip_sigma * sigma).
        trigger_window: Consecutive triggering turns required to fire a
            re-identification repair.
    """

    reader_layers: list[int] = field(default_factory=list)
    threshold: float = 1.0
    clip_sigma: float = 5.0
    trigger_window: int = 3


@dataclass
class SelectorConfig:
    """Settings for ACT best-of-k candidate selection.

    Attributes:
        k: Number of candidates generated per turn.
        temperature: Sampling temperature for candidates (k > 1).
        top_p: Nucleus sampling parameter for candidates (k > 1).
    """

    k: int = 4
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class StateConfig:
    """Settings for the multi-turn conversation state (``act_steering.state``).

    Attributes:
        drift_decay: Per-turn multiplicative factor applied to the deviation
            of each fundamental from its identity baseline after a completed
            turn. ``1.0`` (default) means no drift; values in ``(0, 1)`` pull
            the fundamentals back toward the identity EPA.
        deflection_threshold: Event-1 deflection U considered unresolved
            when it exceeds this value.
        sustained_turns: Consecutive unresolved user messages required to
            latch the trigger.

    Raises:
        ValueError: If any parameter is out of range.
    """

    drift_decay: float = 1.0
    deflection_threshold: float = 10.0
    sustained_turns: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.drift_decay <= 1.0:
            raise ValueError(f"drift_decay must be in (0, 1], got {self.drift_decay}")
        if self.deflection_threshold < 0.0:
            raise ValueError(
                f"deflection_threshold must be non-negative, got {self.deflection_threshold}"
            )
        if self.sustained_turns < 1:
            raise ValueError(f"sustained_turns must be >= 1, got {self.sustained_turns}")


@dataclass
class PipelineConfig:
    """Settings for the per-turn control loop (``act_steering.pipeline``).

    Attributes:
        mode: Mode label recorded in the trace log (e.g. ``"static"``,
            ``"proportional"``, ``"best_of_k"``).
        k: Number of best-of-k candidates generated per turn.
        monitor_positions: Number of response tail positions scored by the
            per-token monitor (0 disables the monitor).
        repair: ``"none"`` or ``"reidentify"`` — whether the repair policy is
            consulted when a turn is flagged.
        repair_threshold: Post-event deflection U above which the turn is
            flagged for repair (in addition to monitor-trace and state
            trigger flags).
        max_repair_attempts: Number of repair candidates tried per flagged turn.
        fallback_response: Static fallback response used when every candidate
            fails scoring (None records the turn as a failed turn).

    Raises:
        ValueError: If ``k < 1`` or ``max_repair_attempts < 0``.
    """

    mode: str = "static"
    k: int = 4
    monitor_positions: int = 0
    repair: str = "none"
    repair_threshold: float = 10.0
    max_repair_attempts: int = 1
    fallback_response: str | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.max_repair_attempts < 0:
            raise ValueError(
                f"max_repair_attempts must be >= 0, got {self.max_repair_attempts}"
            )
        if self.repair not in ("none", "reidentify"):
            raise ValueError(f"repair must be 'none' or 'reidentify', got {self.repair!r}")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration bundling all sub-configs.

    Attributes:
        seed: Master RNG seed for every stochastic operation.
        data: Data / artifact / result paths.
        model: Generation model settings.
        steering: Actuation settings.
        monitor: Monitor settings.
        selector: Best-of-k selector settings.
        state: Conversation-state settings.
        pipeline: Per-turn control-loop settings.
    """

    seed: int = DEFAULT_SEED
    data: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    state: StateConfig = field(default_factory=StateConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (paths rendered as strings)."""
        return _jsonify(asdict(self))


def _jsonify(value: Any) -> Any:
    """Recursively convert paths to strings for JSON serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value
