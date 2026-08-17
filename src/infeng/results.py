"""Typed engine results shared by the API and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["stop", "length", "timeout"]


@dataclass(frozen=True)
class GenerationResult:
    """A completion and its logical (unpadded) accounting metadata."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    finish_reason: FinishReason


@dataclass(frozen=True)
class StreamEvent:
    """An incremental text chunk or the final generation result."""

    event: Literal["chunk", "done"]
    text: str = ""
    result: GenerationResult | None = None
