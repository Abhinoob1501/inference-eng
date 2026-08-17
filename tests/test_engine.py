"""Core-engine integration tests using the real TinyGPT2 model."""

from __future__ import annotations

from dataclasses import replace

import pytest

from infeng.engine import InferenceEngine
from infeng.errors import EngineBusyError, InvalidRequestError
from infeng.results import GenerationResult
from infeng.sampler import SamplingParams


def test_single_generation_contract(engine: InferenceEngine) -> None:
    output = engine.generate(
        "Hello", SamplingParams(max_new_tokens=4, do_sample=False)
    )

    assert isinstance(output, GenerationResult)
    assert output.prompt_tokens == 1
    assert output.generated_tokens == 4
    assert output.total_tokens == output.prompt_tokens + output.generated_tokens
    assert output.finish_reason == "length"
    assert output.text and not output.text.startswith("Hello")


def test_seeded_sampling_is_deterministic(engine: InferenceEngine) -> None:
    params = SamplingParams(max_new_tokens=8, do_sample=True, seed=42)

    output1 = engine.generate("Hello World", params)
    output2 = engine.generate("Hello World", params)

    assert output1.text == output2.text


def test_greedy_ignores_sampling_only_values(engine: InferenceEngine) -> None:
    output1 = engine.generate(
        "Hello",
        SamplingParams(
            max_new_tokens=4,
            do_sample=False,
            temperature=3.0,
            top_k=50,
            top_p=0.5,
            seed=1,
        ),
    )
    output2 = engine.generate(
        "Hello", SamplingParams(max_new_tokens=4, do_sample=False, seed=999)
    )

    assert output1.text == output2.text


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_new_tokens=0),
        SamplingParams(temperature=0),
        SamplingParams(top_k=-1),
        SamplingParams(top_p=0),
    ],
)
def test_invalid_sampling_params_raise(
    engine: InferenceEngine, params: SamplingParams
) -> None:
    with pytest.raises(InvalidRequestError):
        engine.generate("Hello", params)


def test_variable_length_batch_has_logical_token_counts(
    engine: InferenceEngine,
) -> None:
    outputs = engine.generate_batch(
        ["Hi", "Hello there"],
        SamplingParams(max_new_tokens=4, do_sample=False),
    )

    # The shorter row is left-padded to width two internally, but that artificial
    # token must not appear in generated or total usage.
    assert [output.prompt_tokens for output in outputs] == [1, 2]
    assert [output.generated_tokens for output in outputs] == [4, 4]
    assert [output.total_tokens for output in outputs] == [5, 6]


def test_stream_matches_non_streaming_contract(engine: InferenceEngine) -> None:
    events = list(
        engine.stream_generate(
            "Hello", SamplingParams(max_new_tokens=4, do_sample=False)
        )
    )
    chunks = "".join(event.text for event in events if event.event == "chunk")
    done = events[-1]

    assert done.event == "done"
    assert done.result is not None
    # Incremental fragments must concatenate to the same canonical completion that
    # clients receive in the final metadata event.
    assert chunks == done.result.text
    assert not chunks.startswith("Hello")
    assert done.result.generated_tokens == 4


def test_seeded_streaming_is_deterministic(engine: InferenceEngine) -> None:
    params = SamplingParams(max_new_tokens=4, do_sample=True, seed=123)
    first = list(engine.stream_generate("Hello", params))[-1]
    second = list(engine.stream_generate("Hello", params))[-1]

    assert first.result is not None
    assert second.result is not None
    assert first.result.text == second.result.text


def test_request_limits_are_enforced(engine: InferenceEngine) -> None:
    params = SamplingParams(max_new_tokens=2, do_sample=False)

    with pytest.raises(InvalidRequestError, match="non-blank"):
        engine.generate("   ", params)
    with pytest.raises(InvalidRequestError, match="must not be empty"):
        engine.generate_batch([], params)
    with pytest.raises(InvalidRequestError, match="batch exceeds"):
        engine.generate_batch(["hello"] * 9, params)


def test_queue_timeout_rejects_overload(
    engine: InferenceEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        engine,
        "config",
        replace(engine.config, queue_timeout_seconds=0.01),
    )
    # Occupying the only physical model slot simulates another in-flight request
    # without adding a slow real generation call to the test.
    engine._slots.acquire()
    try:
        with pytest.raises(EngineBusyError, match="busy"):
            engine.generate(
                "Hello", SamplingParams(max_new_tokens=1, do_sample=False)
            )
    finally:
        engine._slots.release()


def test_generation_deadline_returns_timeout_reason(
    engine: InferenceEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        engine,
        "config",
        replace(engine.config, generation_timeout_seconds=0.000001),
    )

    # A near-zero cooperative deadline should stop after the first decode step and
    # surface a typed finish reason rather than raising or hanging.
    output = engine.generate(
        "Hello", SamplingParams(max_new_tokens=8, do_sample=False)
    )

    assert output.finish_reason == "timeout"
    assert output.generated_tokens < 8


def test_kv_cache_can_be_disabled_without_changing_greedy_output(
    engine: InferenceEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = SamplingParams(max_new_tokens=4, do_sample=False)
    cached = engine.generate("Hello", params)
    monkeypatch.setattr(
        engine,
        "config",
        replace(engine.config, use_kv_cache=False),
    )

    uncached = engine.generate("Hello", params)

    assert cached.text == uncached.text
    assert engine.metrics_snapshot()["kv_cache_enabled"] is False


def test_engine_metrics_include_tokens(engine: InferenceEngine) -> None:
    metrics = engine.metrics_snapshot()
    assert metrics["requests_completed"] >= 1
    assert metrics["prompt_tokens"] >= 1
    assert metrics["generated_tokens"] >= 1


def test_repeated_prompt_uses_tokenization_cache(engine: InferenceEngine) -> None:
    before = engine.tokenization_cache.snapshot()["hits"]
    params = SamplingParams(max_new_tokens=1, do_sample=False)
    engine.generate("cache this exact prompt", params)
    engine.generate("cache this exact prompt", params)

    after = engine.tokenization_cache.snapshot()
    assert after["hits"] >= before + 1
    assert after["size"] >= 1
