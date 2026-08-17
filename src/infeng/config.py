"""Startup and runtime configuration for the inference engine."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    model_revision: str | None = None
    device: str = "auto"  # auto | cpu | cuda
    max_prompt_tokens: int = 512
    max_new_tokens: int = 512
    max_batch_size: int = 8
    max_concurrent_requests: int = 1
    queue_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        for name in (
            "max_prompt_tokens",
            "max_new_tokens",
            "max_batch_size",
            "max_concurrent_requests",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be > 0")
        if self.generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be > 0")

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build configuration from ``INFENG_*`` environment variables."""

        return cls(
            model_name=os.getenv("INFENG_MODEL_NAME", cls.model_name),
            model_revision=os.getenv("INFENG_MODEL_REVISION") or None,
            device=os.getenv("INFENG_DEVICE", cls.device),
            max_prompt_tokens=_env_int("INFENG_MAX_PROMPT_TOKENS", cls.max_prompt_tokens),
            max_new_tokens=_env_int("INFENG_MAX_NEW_TOKENS", cls.max_new_tokens),
            max_batch_size=_env_int("INFENG_MAX_BATCH_SIZE", cls.max_batch_size),
            max_concurrent_requests=_env_int(
                "INFENG_MAX_CONCURRENT_REQUESTS", cls.max_concurrent_requests
            ),
            queue_timeout_seconds=_env_float(
                "INFENG_QUEUE_TIMEOUT_SECONDS", cls.queue_timeout_seconds
            ),
            generation_timeout_seconds=_env_float(
                "INFENG_GENERATION_TIMEOUT_SECONDS", cls.generation_timeout_seconds
            ),
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)
