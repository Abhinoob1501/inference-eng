"""Deterministic concurrency tests for admission-window dynamic batching."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from infeng.errors import EngineBusyError
from infeng.results import GenerationResult
from infeng.sampler import SamplingParams
from infeng.scheduler import DynamicBatchScheduler


def make_result(prompt: str) -> GenerationResult:
    return GenerationResult(
        text=f" completion:{prompt}",
        prompt_tokens=1,
        generated_tokens=1,
        total_tokens=2,
        model="fake-model",
        latency_ms=1.0,
        finish_reason="length",
    )


class RecordingEngine:
    """Record physical single/batch calls so grouping behavior is directly visible."""

    def __init__(self, *, capacity: int = 16, window_ms: float = 20.0) -> None:
        self.config = SimpleNamespace(
            max_batch_size=8,
            scheduler_batch_window_ms=window_ms,
            scheduler_queue_capacity=capacity,
            queue_timeout_seconds=1.0,
            generation_timeout_seconds=1.0,
        )
        self.lock = threading.Lock()
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def generate(self, prompt, params):
        with self.lock:
            self.single_calls.append(prompt)
        return make_result(prompt)

    def generate_batch(self, prompts, params):
        with self.lock:
            self.batch_calls.append(list(prompts))
        return [make_result(prompt) for prompt in prompts]


def await_result(future: Future[GenerationResult]) -> GenerationResult:
    return future.result(timeout=2.0)


def test_compatible_requests_are_coalesced() -> None:
    engine = RecordingEngine(window_ms=10.0)
    scheduler = DynamicBatchScheduler(engine)
    params = SamplingParams(max_new_tokens=2, do_sample=False)
    first = scheduler.submit("one", params)
    second = scheduler.submit("two", params)

    # Submitting before the worker starts guarantees both compatible requests are
    # visible during one admission window without relying on timing sleeps.
    scheduler.start()
    try:
        assert await_result(first).text == " completion:one"
        assert await_result(second).text == " completion:two"
    finally:
        scheduler.close()

    assert engine.batch_calls == [["one", "two"]]
    assert scheduler.snapshot()["average_batch_size"] == 2.0


def test_seeded_sampling_requests_are_not_coalesced() -> None:
    engine = RecordingEngine(window_ms=10.0)
    scheduler = DynamicBatchScheduler(engine)
    params = SamplingParams(max_new_tokens=2, do_sample=True, seed=42)
    first = scheduler.submit("one", params)
    second = scheduler.submit("two", params)

    scheduler.start()
    try:
        await_result(first)
        await_result(second)
    finally:
        scheduler.close()

    # Neighbor-dependent random-number consumption would break seeded determinism,
    # so each seeded request must become its own physical call.
    assert engine.single_calls == ["one", "two"]
    assert engine.batch_calls == []
    assert scheduler.snapshot()["batches"] == 2


def test_queue_capacity_enforces_backpressure() -> None:
    scheduler = DynamicBatchScheduler(RecordingEngine(capacity=1))
    scheduler.submit("one", SamplingParams(do_sample=False))

    with pytest.raises(EngineBusyError, match="queue is full"):
        scheduler.submit("two", SamplingParams(do_sample=False))

    scheduler.close()


def test_cancelled_request_releases_queue_capacity() -> None:
    scheduler = DynamicBatchScheduler(RecordingEngine(capacity=1))
    cancelled = scheduler.submit("one", SamplingParams(do_sample=False))
    assert cancelled.cancel()

    replacement = scheduler.submit("two", SamplingParams(do_sample=False))
    scheduler.start()
    try:
        assert await_result(replacement).text == " completion:two"
    finally:
        scheduler.close()


def test_oldest_incompatible_request_is_served_next() -> None:
    engine = RecordingEngine(window_ms=1.0)
    scheduler = DynamicBatchScheduler(engine)
    first_params = SamplingParams(max_new_tokens=2, do_sample=False)
    other_params = SamplingParams(max_new_tokens=3, do_sample=False)
    first = scheduler.submit("first", first_params)
    other = scheduler.submit("other", other_params)
    third = scheduler.submit("third", first_params)

    scheduler.start()
    try:
        await_result(first)
        await_result(other)
        await_result(third)
    finally:
        scheduler.close()

    # The compatible third request may join the first, but the older incompatible
    # request must lead the next batch rather than starving behind newer work.
    assert engine.batch_calls == [["first", "third"]]
    assert engine.single_calls == ["other"]
    assert scheduler.snapshot()["average_queue_wait_ms"] >= 0


def test_cancelled_request_is_not_executed() -> None:
    engine = RecordingEngine()
    scheduler = DynamicBatchScheduler(engine)
    future = scheduler.submit("cancel me", SamplingParams(do_sample=False))
    assert future.cancel()
    scheduler.start()
    scheduler.close()

    assert engine.single_calls == []
    assert engine.batch_calls == []
    assert scheduler.snapshot()["cancelled"] == 1
