"""Thread-safe in-process serving metrics."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .results import GenerationResult


class EngineMetrics:
    """Accumulate lightweight counters without adding a metrics dependency.

    The lock matters because FastAPI, the scheduler worker, and streaming threads
    can finish requests concurrently. ``snapshot`` copies values while holding the
    lock, producing an internally consistent response for ``GET /metrics``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._latency_ms = 0.0

    def record_results(
        self, mode: str, results: Iterable[GenerationResult], latency_ms: float
    ) -> None:
        # Materialize once because callers may pass any iterable and we traverse it
        # several times to calculate different counters.
        materialized = list(results)
        with self._lock:
            self._counters[f"{mode}_calls"] += 1
            self._counters["requests_completed"] += len(materialized)
            self._counters["prompt_tokens"] += sum(
                result.prompt_tokens for result in materialized
            )
            self._counters["generated_tokens"] += sum(
                result.generated_tokens for result in materialized
            )
            self._counters["timeouts"] += sum(
                result.finish_reason == "timeout" for result in materialized
            )
            self._latency_ms += latency_ms

    def record_failure(self, mode: str) -> None:
        """Count failures separately for single, batch, and streaming paths."""

        with self._lock:
            self._counters[f"{mode}_failures"] += 1
            self._counters["requests_failed"] += 1

    def snapshot(self) -> dict[str, Any]:
        """Build a JSON-serializable point-in-time metrics view."""

        with self._lock:
            counters = dict(self._counters)
            calls = sum(
                counters.get(f"{mode}_calls", 0)
                for mode in ("single", "batch", "stream")
            )
            return {
                **counters,
                "average_generation_latency_ms": round(
                    self._latency_ms / calls, 2
                )
                if calls
                else 0.0,
            }
