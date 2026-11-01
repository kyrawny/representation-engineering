"""Shared pytest fixtures for the ACT steering test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

#: Repository root (three levels up: tests -> act -> examples -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the root of the representation-engineering repository."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the ACT data directory (``data/act``)."""
    return REPO_ROOT / "data" / "act"
