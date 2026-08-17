"""Per-request generation behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    seed: int | None = None
