"""Per-request generation behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    """Decoding options supplied independently for each generation request.

    Sampling-only fields are accepted in greedy mode but ignored by the engine.
    This makes ``do_sample=False`` ergonomic for API clients whose serializers
    still send default temperature/top-k/top-p values.
    """

    # Maximum decode work requested by this caller; EngineConfig adds a global cap.
    max_new_tokens: int = 32
    # Temperature rescales logits before sampling. Values below 1 sharpen the
    # distribution and values above 1 make it more random.
    temperature: float = 1.0
    # top_k=0 disables top-k filtering. top_p=1 disables nucleus filtering.
    top_k: int = 50
    top_p: float = 1.0
    # False selects deterministic argmax/greedy decoding.
    do_sample: bool = True
    # A seed provides repeatability. The engine isolates it from global RNG state.
    seed: int | None = None
