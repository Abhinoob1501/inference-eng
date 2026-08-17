import pytest

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine


@pytest.fixture(scope="session")
def engine() -> InferenceEngine:
    return InferenceEngine(
        EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu")
    )
