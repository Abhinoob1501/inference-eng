"""Startup and runtime configuration for the inference engine."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Process-wide settings that do not change between individual requests.

    Request-specific decoding options live in :class:`SamplingParams`. Keeping
    these two categories separate prevents one caller from changing global safety
    limits or scheduler behavior for everybody else.
    """

    # Hugging Face model identity. A revision can pin a branch, tag, or commit so
    # benchmark results can be reproduced even if the upstream repository changes.
    model_name: str = "sshleifer/tiny-gpt2"
    model_revision: str | None = None

    # ``auto`` chooses CUDA when PyTorch can see it, otherwise CPU.
    device: str = "auto"  # auto | cpu | cuda

    # Hard resource limits are checked before model.generate() allocates decode
    # state. They protect both memory and request latency.
    max_prompt_tokens: int = 512
    max_new_tokens: int = 512
    max_batch_size: int = 8
    max_concurrent_requests: int = 1

    # The scheduler briefly waits for compatible requests so it can turn several
    # individual HTTP calls into one tensor batch.
    scheduler_batch_window_ms: float = 20.0
    scheduler_queue_capacity: int = 64

    # This cache stores token IDs only; it is deliberately not a model KV cache.
    tokenization_cache_capacity: int = 256

    # Queue timeout covers waiting for an engine slot. Generation timeout is a
    # cooperative deadline checked between autoregressive decoding steps.
    queue_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 60.0

    # Hugging Face generation normally uses the attention KV cache. Exposing this
    # switch makes the performance effect measurable in the benchmark harness.
    use_kv_cache: bool = True

    def __post_init__(self) -> None:
        """Fail during startup instead of discovering invalid limits mid-request."""

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        for name in (
            "max_prompt_tokens",
            "max_new_tokens",
            "max_batch_size",
            "max_concurrent_requests",
            "scheduler_queue_capacity",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be > 0")
        if self.scheduler_batch_window_ms < 0:
            raise ValueError("scheduler_batch_window_ms must be >= 0")
        if self.tokenization_cache_capacity < 0:
            raise ValueError("tokenization_cache_capacity must be >= 0")
        if self.generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be > 0")

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build configuration from ``INFENG_*`` environment variables."""

        # Environment variables make one package artifact usable in local, CI,
        # CPU, and GPU deployments without editing source code.
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
            scheduler_batch_window_ms=_env_float(
                "INFENG_SCHEDULER_BATCH_WINDOW_MS", cls.scheduler_batch_window_ms
            ),
            scheduler_queue_capacity=_env_int(
                "INFENG_SCHEDULER_QUEUE_CAPACITY", cls.scheduler_queue_capacity
            ),
            tokenization_cache_capacity=_env_int(
                "INFENG_TOKENIZATION_CACHE_CAPACITY",
                cls.tokenization_cache_capacity,
            ),
            queue_timeout_seconds=_env_float(
                "INFENG_QUEUE_TIMEOUT_SECONDS", cls.queue_timeout_seconds
            ),
            generation_timeout_seconds=_env_float(
                "INFENG_GENERATION_TIMEOUT_SECONDS", cls.generation_timeout_seconds
            ),
            use_kv_cache=_env_bool("INFENG_USE_KV_CACHE", cls.use_kv_cache),
        )


def _env_int(name: str, default: int) -> int:
    """Read an optional integer while preserving the dataclass default."""

    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    """Read an optional float while preserving the dataclass default."""

    value = os.getenv(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    """Parse common human-friendly boolean spellings with strict errors."""

    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
