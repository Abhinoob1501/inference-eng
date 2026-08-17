"""Unit tests for bounded, defensive-copy tokenization cache behavior."""

import torch

from infeng.cache import TokenizationCache


def encoded(token: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[token]]),
        "attention_mask": torch.tensor([[1]]),
    }


def test_tokenization_cache_tracks_hits_and_misses() -> None:
    cache = TokenizationCache(capacity=2)
    assert cache.get("one") is None
    cache.put("one", encoded(1))

    cached = cache.get("one")

    assert cached is not None
    assert cached["input_ids"].item() == 1
    assert cache.snapshot()["hits"] == 1
    assert cache.snapshot()["misses"] == 1


def test_tokenization_cache_is_bounded_and_lru() -> None:
    cache = TokenizationCache(capacity=2)
    cache.put("one", encoded(1))
    cache.put("two", encoded(2))
    # Touching "one" makes "two" the least-recently-used eviction candidate.
    assert cache.get("one") is not None
    cache.put("three", encoded(3))

    assert cache.get("two") is None
    assert cache.get("one") is not None
    assert cache.get("three") is not None
    assert cache.snapshot()["evictions"] == 1


def test_disabled_cache_never_stores_entries() -> None:
    cache = TokenizationCache(capacity=0)
    cache.put("one", encoded(1))

    assert cache.get("one") is None
    assert cache.snapshot()["size"] == 0
