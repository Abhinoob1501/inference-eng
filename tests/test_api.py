"""HTTP contract tests that replace expensive model inference with a fake engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from infeng.api import create_app
from infeng.errors import EngineBusyError, InvalidRequestError
from infeng.results import GenerationResult, StreamEvent


def result(text: str = " world") -> GenerationResult:
    return GenerationResult(
        text=text,
        prompt_tokens=1,
        generated_tokens=1,
        total_tokens=2,
        model="fake-model",
        latency_ms=1.25,
        finish_reason="length",
    )


class FakeEngine:
    """Implement the engine protocol with instant, predictable responses.

    API tests should verify routing, schemas, lifecycle, and wire formats. Real
    generation correctness belongs in test_engine.py, where the model loads once.
    """

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_name="fake-model",
            max_batch_size=8,
            scheduler_batch_window_ms=1.0,
            scheduler_queue_capacity=16,
            queue_timeout_seconds=1.0,
            generation_timeout_seconds=1.0,
            use_kv_cache=True,
        )
        self.device = "cpu"
        self.last_params = None

    def generate(self, prompt, params):
        self.last_params = params
        return result()

    def generate_batch(self, prompts, params):
        self.last_params = params
        return [result(f" completion-{index}") for index, _ in enumerate(prompts)]

    def stream_generate(self, prompt, params):
        self.last_params = params
        yield StreamEvent(event="chunk", text=" world")
        yield StreamEvent(event="done", result=result())


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(fake_engine: FakeEngine):
    with TestClient(create_app(lambda: fake_engine)) as test_client:
        yield test_client


def test_liveness_and_readiness(client: TestClient) -> None:
    assert client.get("/live").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ok",
        "model": "fake-model",
        "device": "cpu",
        "kv_cache_enabled": True,
    }


def test_greedy_defaults_are_accepted(
    client: TestClient, fake_engine: FakeEngine
) -> None:
    response = client.post(
        "/generate",
        json={"prompt": "Hello", "max_new_tokens": 2, "do_sample": False},
    )

    assert response.status_code == 200
    assert response.json()["text"] == " world"
    assert response.json()["finish_reason"] == "length"
    assert fake_engine.last_params.do_sample is False
    assert fake_engine.last_params.top_k == 50
    assert response.headers["x-request-id"]


def test_batch_endpoint(client: TestClient) -> None:
    response = client.post(
        "/generate/batch",
        json={"prompts": ["one", "two"], "do_sample": False},
    )

    assert response.status_code == 200
    assert [item["text"] for item in response.json()["outputs"]] == [
        " completion-0",
        " completion-1",
    ]


def test_stream_endpoint_uses_sse(client: TestClient) -> None:
    response = client.post(
        "/generate/stream",
        json={"prompt": "Hello", "do_sample": False},
    )

    assert response.status_code == 200
    # TestClient buffers the body, but the frame text still verifies the same SSE
    # contract consumed incrementally by scripts/stream_client.py.
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: chunk" in response.text
    assert '"text": " world"' in response.text
    assert "event: done" in response.text
    assert '"finish_reason": "length"' in response.text


def test_metrics_endpoint_reports_scheduler(client: TestClient) -> None:
    client.post("/generate", json={"prompt": "Hello", "do_sample": False})
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["scheduler"]["submitted"] >= 1
    assert response.json()["scheduler"]["completed"] >= 1
    assert response.json()["scheduler"]["running"] is True


def test_openai_completion_contract(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Hello",
            "max_tokens": 2,
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("cmpl-")
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == " world"
    assert payload["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


def test_openai_stream_contract(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello",
            "max_tokens": 2,
            "temperature": 0,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"object": "text_completion"' in response.text
    assert '"text": " world"' in response.text
    assert '"finish_reason": "length"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_openai_prompt_list_returns_indexed_choices(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"prompt": ["one", "two"], "max_tokens": 2, "temperature": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [choice["index"] for choice in payload["choices"]] == [0, 1]
    assert payload["usage"]["prompt_tokens"] == 2
    assert payload["usage"]["completion_tokens"] == 2
    assert payload["usage"]["total_tokens"] == 4


def test_openai_rejects_wrong_model(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "not-loaded", "prompt": "Hello"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"


def test_schema_rejects_blank_prompt(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (InvalidRequestError("bad prompt"), 422, "invalid_request"),
        (EngineBusyError("busy"), 503, "engine_busy"),
    ],
)
def test_domain_errors_are_structured(error, status, code) -> None:
    fake = FakeEngine()

    def raise_error(prompt, params):
        raise error

    fake.generate = raise_error
    with TestClient(create_app(lambda: fake)) as test_client:
        response = test_client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == status
    assert response.json()["error"] == code
    assert response.json()["request_id"]


def test_startup_failure_keeps_liveness_available() -> None:
    def fail():
        raise RuntimeError("load failed")

    # The app should stay diagnosable even when model construction fails.
    with TestClient(create_app(fail)) as test_client:
        assert test_client.get("/live").status_code == 200
        response = test_client.get("/ready")
        generation = test_client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "not_ready"
    assert generation.status_code == 503
    assert generation.json()["error"] == "not_ready"
