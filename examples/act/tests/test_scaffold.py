"""Scaffold tests: package import, config defaults, and data file presence."""

from __future__ import annotations

import json
from pathlib import Path


def test_package_imports() -> None:
    """The package must import without torch or a GPU."""
    import act_steering

    assert act_steering.__version__


def test_experiment_config_defaults() -> None:
    """A default ExperimentConfig must construct with the documented defaults."""
    from act_steering.config import DEFAULT_SEED, ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.seed == DEFAULT_SEED
    assert cfg.model.name == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.selector.k == 4
    assert cfg.steering.token_pos == "end"


def test_config_to_dict_is_json_serializable() -> None:
    """to_dict must render paths as strings for JSON metadata logging."""
    from act_steering.config import ExperimentConfig

    cfg = ExperimentConfig()
    data = cfg.to_dict()
    json.dumps(data)  # must not raise
    assert data["model"]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert isinstance(data["data"]["data_dir"], str)
    assert data["data"]["data_dir"].replace("\\", "/").endswith("data/act")


def test_default_data_files_exist(data_dir: Path) -> None:
    """Every referenced ACT data file must exist in the repository."""
    from act_steering.config import DataPaths

    paths = DataPaths()
    for path in (
        paths.coefficients_file,
        paths.identities_file,
        paths.behaviors_file,
        paths.modifiers_file,
        paths.user_inputs_file,
        paths.truncated_outputs_file,
    ):
        assert path.exists(), f"missing data file: {path}"
        assert path.is_relative_to(data_dir) or str(path).startswith(str(data_dir))
