"""Configuration parsing tests, especially deployment-facing environment values."""

import pytest

from infeng.config import EngineConfig


def test_engine_config_reads_scheduler_and_cache_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFENG_SCHEDULER_BATCH_WINDOW_MS", "7.5")
    monkeypatch.setenv("INFENG_SCHEDULER_QUEUE_CAPACITY", "12")
    monkeypatch.setenv("INFENG_TOKENIZATION_CACHE_CAPACITY", "0")
    monkeypatch.setenv("INFENG_USE_KV_CACHE", "false")

    config = EngineConfig.from_env()

    assert config.scheduler_batch_window_ms == 7.5
    assert config.scheduler_queue_capacity == 12
    assert config.tokenization_cache_capacity == 0
    assert config.use_kv_cache is False


def test_engine_config_rejects_invalid_boolean_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFENG_USE_KV_CACHE", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        EngineConfig.from_env()
