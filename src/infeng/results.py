"""Typed engine results shared by the API and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["stop", "length", "timeout"]


@dataclass(frozen=True)
class GenerationResult:
    """A completion and its logical (unpadded) accounting metadata."""

    # ``text`` is completion-only across single, batch, streaming, and HTTP APIs.
    text: str
    # Counts are logical counts: left-padding added for a batch is never billed.
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    finish_reason: FinishReason


@dataclass(frozen=True)
class StreamEvent:
    """An incremental text chunk or the final generation result."""

    # TextIteratorStreamer emits decoded fragments, which may contain more or less
    # than one token. Calling them chunks avoids promising token-sized boundaries.
    event: Literal["chunk", "done"]
    text: str = ""
    result: GenerationResult | None = None
