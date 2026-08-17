"""Bounded preprocessing caches used by the inference engine."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import torch


class TokenizationCache:
    """Thread-safe LRU cache for exact prompt tokenization results.

    Tokenization is cheaper than model inference but still repeated work for common
    system prompts. Cached tensors stay on CPU so the cache does not reserve scarce
    GPU memory. Values are cloned at the boundary so callers cannot mutate the copy
    owned by the cache.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _clone(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in value.items()}

    def get(self, prompt: str) -> dict[str, torch.Tensor] | None:
        """Return a defensive copy and mark the entry as most recently used."""

        with self._lock:
            value = self._entries.get(prompt)
            if value is None:
                self._misses += 1
                return None
            self._entries.move_to_end(prompt)
            self._hits += 1
            return self._clone(value)

    def put(self, prompt: str, value: dict[str, torch.Tensor]) -> None:
        """Insert a value and evict least-recently-used entries over capacity."""

        if self.capacity == 0:
            return
        with self._lock:
            self._entries[prompt] = self._clone(value)
            self._entries.move_to_end(prompt)
            # ``last=False`` removes the oldest key from OrderedDict.
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
                self._evictions += 1

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe cache statistics for the metrics endpoint."""

        with self._lock:
            requests = self._hits + self._misses
            return {
                "capacity": self.capacity,
                "size": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / requests, 4) if requests else 0.0,
            }
