"""Bounded admission-window batching for compatible generation requests."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from .engine import InferenceEngine
from .errors import EngineBusyError, EngineNotReadyError, GenerationError
from .results import GenerationResult
from .sampler import SamplingParams


@dataclass(frozen=True)
class _PendingRequest:
    """One queued caller and the Future used to return its eventual result."""

    prompt: str
    params: SamplingParams
    future: Future[GenerationResult]
    submitted_at: float


class DynamicBatchScheduler:
    """Coalesce nearby compatible requests into static model batches.

    This is admission-window dynamic batching: new requests can join during a
    short window before decoding starts. It intentionally does not claim to add
    requests to an already-running decode batch.
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self._condition = threading.Condition()
        self._queue: deque[_PendingRequest] = deque()
        self._worker: threading.Thread | None = None
        self._stopping = False
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, int | float] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "batches": 0,
            "batched_requests": 0,
            "max_observed_batch_size": 0,
            "total_queue_wait_ms": 0.0,
            "max_queue_wait_ms": 0.0,
        }

    def start(self) -> None:
        """Start the one background worker; repeated calls are harmless."""

        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._stopping:
                raise EngineNotReadyError("scheduler has been stopped")
            self._worker = threading.Thread(
                target=self._run,
                daemon=True,
                name="infeng-batch-scheduler",
            )
            self._worker.start()

    def close(self, timeout: float = 5.0) -> None:
        """Reject queued work and wait briefly for the active model call to finish."""

        with self._condition:
            self._stopping = True
            # Remove queued requests atomically so the worker cannot claim them
            # after shutdown has begun.
            pending = list(self._queue)
            self._queue.clear()
            self._condition.notify_all()
        for request in pending:
            # A cancelled Future cannot receive an exception. For all other queued
            # callers, set_running_or_notify_cancel reserves the Future before the
            # shutdown exception is delivered.
            if request.future.set_running_or_notify_cancel():
                request.future.set_exception(
                    EngineNotReadyError("scheduler stopped before request execution")
                )
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def submit(
        self, prompt: str, params: SamplingParams
    ) -> Future[GenerationResult]:
        """Enqueue without blocking and return a cancellable Future."""

        future: Future[GenerationResult] = Future()
        pending = _PendingRequest(prompt, params, future, time.perf_counter())
        with self._condition:
            if self._stopping:
                raise EngineNotReadyError("scheduler is not accepting requests")
            # Cancelled entries should stop consuming bounded queue capacity even
            # if the worker has not reached them yet.
            self._queue = deque(
                request for request in self._queue if not request.future.cancelled()
            )
            if len(self._queue) >= self.engine.config.scheduler_queue_capacity:
                raise EngineBusyError("scheduler queue is full; retry later")
            self._queue.append(pending)
            self._increment("submitted")
            self._condition.notify()
        future.add_done_callback(self._record_cancellation)
        return future

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        """Synchronous convenience wrapper used by ordinary FastAPI routes."""

        future = self.submit(prompt, params)
        # Add the independent budgets instead of waiting forever if a worker or
        # backend unexpectedly stalls. The model deadline itself remains
        # cooperative and is checked between decode steps.
        timeout = (
            self.engine.config.queue_timeout_seconds
            + self.engine.config.generation_timeout_seconds
            + self.engine.config.scheduler_batch_window_ms / 1000
            + 1.0
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise GenerationError("scheduled generation exceeded its request timeout") from exc

    def snapshot(self) -> dict[str, Any]:
        """Return scheduler health, queue pressure, and batching efficiency."""

        with self._condition:
            queue_depth = len(self._queue)
            running = self._worker is not None and self._worker.is_alive()
        with self._metrics_lock:
            metrics = dict(self._metrics)
        batches = metrics["batches"]
        return {
            **metrics,
            "total_queue_wait_ms": round(metrics["total_queue_wait_ms"], 2),
            "max_queue_wait_ms": round(metrics["max_queue_wait_ms"], 2),
            "queue_depth": queue_depth,
            "queue_capacity": self.engine.config.scheduler_queue_capacity,
            "batch_window_ms": self.engine.config.scheduler_batch_window_ms,
            "average_batch_size": round(metrics["batched_requests"] / batches, 2)
            if batches
            else 0.0,
            "average_queue_wait_ms": round(
                metrics["total_queue_wait_ms"] / metrics["batched_requests"], 2
            )
            if metrics["batched_requests"]
            else 0.0,
            "running": running and not self._stopping,
        }

    def _record_cancellation(self, future: Future[GenerationResult]) -> None:
        """Future callbacks let callers cancel without touching scheduler locks."""

        if future.cancelled():
            self._increment("cancelled")

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metrics[name] += amount

    @staticmethod
    def _can_batch(params: SamplingParams) -> bool:
        """Preserve per-request seeded determinism by running seeded calls alone."""

        # Batched multinomial sampling consumes random numbers by row. Changing the
        # number of neighbors would therefore change a seeded request's output.
        return not (params.do_sample and params.seed is not None)

    def _collect_batch(self) -> list[_PendingRequest] | None:
        """Pop the oldest request, then gather compatible neighbors briefly."""

        with self._condition:
            # This loop is written to re-check both queue and shutdown state after
            # every wake-up; Condition.wait may wake without a new request.
            while True:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if not self._queue:
                    return None
                first = self._queue.popleft()
                if not first.future.cancelled():
                    break

            batch = [first]
            if not self._can_batch(first.params):
                return batch

            deadline = (
                time.perf_counter()
                + self.engine.config.scheduler_batch_window_ms / 1000
            )
            while len(batch) < self.engine.config.max_batch_size:
                # Remove cancellations before scanning so they neither consume a
                # batch slot nor make an apparently full queue block producers.
                self._queue = deque(
                    request
                    for request in self._queue
                    if not request.future.cancelled()
                )
                compatible_index = next(
                    (
                        index
                        for index, request in enumerate(self._queue)
                        if not request.future.cancelled()
                        and request.params == first.params
                    ),
                    None,
                )
                if compatible_index is not None:
                    # Scanning beyond an incompatible request improves batching,
                    # but the incompatible request stays at the front and becomes
                    # the leader of the very next batch. This bounds unfairness to
                    # one model call rather than allowing starvation.
                    batch.append(self._queue[compatible_index])
                    del self._queue[compatible_index]
                    continue

                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stopping:
                    break
                # New submissions notify the condition, letting the window close
                # early as soon as enough compatible work has arrived.
                self._condition.wait(timeout=remaining)
            return batch

    def _run(self) -> None:
        """Continuously execute collected batches until close() requests shutdown."""

        while True:
            batch = self._collect_batch()
            if batch is None:
                return
            # set_running_or_notify_cancel performs the atomic race with a caller
            # cancelling its Future. Only successfully claimed requests reach the
            # model.
            active = [
                request
                for request in batch
                if request.future.set_running_or_notify_cancel()
            ]
            if not active:
                continue

            batch_size = len(active)
            now = time.perf_counter()
            queue_waits = [
                (now - request.submitted_at) * 1000 for request in active
            ]
            self._increment("batches")
            self._increment("batched_requests", batch_size)
            with self._metrics_lock:
                self._metrics["max_observed_batch_size"] = max(
                    self._metrics["max_observed_batch_size"], batch_size
                )
                self._metrics["total_queue_wait_ms"] += sum(queue_waits)
                self._metrics["max_queue_wait_ms"] = max(
                    self._metrics["max_queue_wait_ms"], max(queue_waits)
                )

            try:
                # Avoid adding padding overhead when only one request arrived.
                if batch_size == 1:
                    results = [
                        self.engine.generate(active[0].prompt, active[0].params)
                    ]
                else:
                    results = self.engine.generate_batch(
                        [request.prompt for request in active], active[0].params
                    )
            except Exception as exc:
                # Every Future in a physical model batch observes the same backend
                # exception because the model call succeeds or fails as one unit.
                self._increment("failed", batch_size)
                for request in active:
                    request.future.set_exception(exc)
                continue

            self._increment("completed", batch_size)
            # generate_batch preserves input order, so zipping maps each result
            # back to the Future owned by the corresponding HTTP request.
            for request, result in zip(active, results, strict=True):
                request.future.set_result(result)
