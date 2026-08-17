"""Shared real-model fixtures for integration-level engine tests."""

import pytest

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine


@pytest.fixture(scope="session")
def engine() -> InferenceEngine:
    """Load TinyGPT2 once so many correctness tests remain fast and deterministic."""

    return InferenceEngine(
        EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu")
    )
