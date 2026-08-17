"""FastAPI boundary: lifecycle, validation, error mapping, and wire protocols.

The core engine stays unaware of HTTP. This module owns process startup/shutdown,
routes, Server-Sent Events, request IDs, and the OpenAI compatibility envelope.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import EngineConfig
from .engine import InferenceEngine
from .errors import (
    EngineBusyError,
    EngineNotReadyError,
    GenerationError,
    InvalidRequestError,
)
from .results import StreamEvent
from .scheduler import DynamicBatchScheduler
from .schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
    OpenAICompletionChoice,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIUsage,
)

logger = logging.getLogger("infeng.api")
EngineFactory = Callable[[], InferenceEngine]


def _default_engine_factory() -> InferenceEngine:
    """Create the production engine from deployment environment variables."""

    return InferenceEngine(EngineConfig.from_env())


def create_app(engine_factory: EngineFactory = _default_engine_factory) -> FastAPI:
    """Build an application; injection keeps API tests independent of model loads."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Loading in lifespan avoids downloading/allocating a model merely because
        # another module imported ``infeng.api`` (for docs, tooling, or tests).
        application.state.engine = None
        application.state.scheduler = None
        application.state.startup_error = None
        try:
            application.state.engine = engine_factory()
            # Ordinary single-request routes enter this scheduler. Explicit batch
            # and streaming routes retain their specialized engine paths.
            application.state.scheduler = DynamicBatchScheduler(
                application.state.engine
            )
            application.state.scheduler.start()
        except Exception as exc:
            # Keep the HTTP process alive so /live works and /ready can report a
            # controlled 503 instead of the server crashing during import/startup.
            application.state.startup_error = exc
            logger.exception("engine_startup_failed")
        yield
        # Stop accepting queued requests before releasing the model reference.
        scheduler = getattr(application.state, "scheduler", None)
        if scheduler is not None:
            scheduler.close()
        application.state.scheduler = None
        application.state.engine = None

    application = FastAPI(title="InfEng", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]):
        """Attach a correlation ID and emit one structured completion log."""

        # Respect an upstream ID when present so logs can be joined across services.
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )
        return response

    def error_response(
        request: Request, status_code: int, error: str, detail: str
    ) -> JSONResponse:
        """Select the native or OpenAI-compatible error envelope by route."""

        if request.url.path.startswith("/v1/"):
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "message": detail,
                        "type": error,
                        "param": None,
                        "code": error,
                    }
                },
            )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "detail": detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Domain exceptions remain transport-neutral in engine.py. These handlers are
    # the single place where they acquire HTTP status codes and JSON shapes.
    @application.exception_handler(InvalidRequestError)
    async def invalid_request_handler(
        request: Request, exc: InvalidRequestError
    ) -> JSONResponse:
        return error_response(request, 422, "invalid_request", str(exc))

    @application.exception_handler(EngineBusyError)
    async def busy_handler(request: Request, exc: EngineBusyError) -> JSONResponse:
        return error_response(request, 503, "engine_busy", str(exc))

    @application.exception_handler(EngineNotReadyError)
    async def not_ready_handler(
        request: Request, exc: EngineNotReadyError
    ) -> JSONResponse:
        return error_response(request, 503, "not_ready", str(exc))

    @application.exception_handler(GenerationError)
    async def generation_handler(request: Request, exc: GenerationError) -> JSONResponse:
        return error_response(request, 500, "generation_failed", str(exc))

    def get_engine(request: Request) -> InferenceEngine:
        """Fetch the lifespan-owned engine or raise a stable readiness error."""

        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            startup_error = getattr(request.app.state, "startup_error", None)
            detail = "model is not ready"
            if startup_error is not None:
                detail += f" ({type(startup_error).__name__})"
            raise EngineNotReadyError(detail)
        return engine

    def get_scheduler(request: Request) -> DynamicBatchScheduler:
        """Fetch the running scheduler used for dynamically batched requests."""

        scheduler = getattr(request.app.state, "scheduler", None)
        if scheduler is None:
            raise EngineNotReadyError("request scheduler is not ready")
        return scheduler

    @application.get("/live")
    def live() -> dict[str, str]:
        # Liveness answers only "is the API process responsive?". It intentionally
        # does not depend on model startup and should not trigger restarts for a
        # transient model download/configuration failure.
        return {"status": "ok"}

    @application.get("/ready", response_model=None)
    @application.get("/health", response_model=None)
    def ready(request: Request) -> JSONResponse | dict[str, Any]:
        # Readiness is stricter: traffic is safe only after both model and scheduler
        # exist. /health remains as a backwards-compatible alias.
        engine = getattr(request.app.state, "engine", None)
        scheduler = getattr(request.app.state, "scheduler", None)
        if engine is None or scheduler is None:
            return error_response(request, 503, "not_ready", "model is not ready")
        return {
            "status": "ok",
            "model": engine.config.model_name,
            "device": engine.device,
            "kv_cache_enabled": getattr(engine.config, "use_kv_cache", True),
        }

    @application.get("/metrics")
    def metrics(request: Request) -> dict[str, Any]:
        """Expose dependency-free JSON metrics for learning and local monitoring."""

        engine = get_engine(request)
        scheduler = get_scheduler(request)
        engine_metrics = (
            engine.metrics_snapshot() if hasattr(engine, "metrics_snapshot") else {}
        )
        return {
            "engine": engine_metrics,
            "scheduler": scheduler.snapshot(),
        }

    @application.post("/generate", response_model=GenerateResponse)
    def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
        # Concurrent calls with identical params may be coalesced by the scheduler.
        result = get_scheduler(request).generate(
            body.prompt, body.to_sampling_params()
        )
        return GenerateResponse.model_validate(result)

    @application.post("/generate/batch", response_model=BatchGenerateResponse)
    def generate_batch(
        request: Request, body: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        # This explicit endpoint already supplies a complete batch, so sending each
        # row through the admission scheduler would only split/reassemble it.
        results = get_engine(request).generate_batch(
            body.prompts, body.to_sampling_params()
        )
        return BatchGenerateResponse(
            outputs=[GenerateResponse.model_validate(result) for result in results]
        )

    @application.post("/generate/stream")
    def generate_stream(request: Request, body: GenerateRequest) -> StreamingResponse:
        # A long-lived stream bypasses admission batching because each caller needs
        # chunks delivered and cancelled independently.
        engine = get_engine(request)

        def encode(event: str, data: dict[str, Any]) -> str:
            # SSE frames are separated by a blank line. The event name lets clients
            # distinguish partial text, final metadata, and an in-stream error.
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def event_stream() -> Iterator[str]:
            # Once response headers have been sent, HTTP status cannot change.
            # Expected generation failures therefore become explicit SSE events.
            try:
                for event in engine.stream_generate(
                    body.prompt, body.to_sampling_params()
                ):
                    yield _encode_stream_event(event, encode)
            except InvalidRequestError as exc:
                yield encode("error", {"error": "invalid_request", "detail": str(exc)})
            except EngineBusyError as exc:
                yield encode("error", {"error": "engine_busy", "detail": str(exc)})
            except GenerationError as exc:
                yield encode("error", {"error": "generation_failed", "detail": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/v1/completions", response_model=None)
    def openai_completions(
        request: Request, body: OpenAICompletionRequest
    ) -> OpenAICompletionResponse | StreamingResponse:
        """Serve the supported OpenAI text-completions compatibility subset."""

        engine = get_engine(request)
        if body.model is not None and body.model != engine.config.model_name:
            raise InvalidRequestError(
                f"requested model {body.model!r} is not loaded"
            )

        # Internally normalize the API's string-or-list prompt union to one list.
        prompts = [body.prompt] if isinstance(body.prompt, str) else body.prompt
        params = body.to_sampling_params()
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if body.stream:
            # Multiple streamed choices require interleaving indexed event streams;
            # keep this learning implementation explicit by supporting one only.
            if len(prompts) != 1:
                raise InvalidRequestError(
                    "streaming completions require exactly one prompt"
                )
            return StreamingResponse(
                _openai_completion_stream(
                    engine,
                    prompts[0],
                    params,
                    completion_id,
                    created,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if len(prompts) == 1:
            # A single prompt benefits from dynamic admission batching with other
            # concurrent HTTP callers; an explicit list is already a physical batch.
            results = [get_scheduler(request).generate(prompts[0], params)]
        else:
            results = engine.generate_batch(prompts, params)
        return OpenAICompletionResponse(
            id=completion_id,
            created=created,
            model=engine.config.model_name,
            choices=[
                OpenAICompletionChoice(
                    text=result.text,
                    index=index,
                    finish_reason=_openai_finish_reason(result.finish_reason),
                )
                for index, result in enumerate(results)
            ],
            usage=OpenAIUsage(
                prompt_tokens=sum(result.prompt_tokens for result in results),
                completion_tokens=sum(
                    result.generated_tokens for result in results
                ),
                total_tokens=sum(result.total_tokens for result in results),
            ),
        )

    return application


def _encode_stream_event(
    event: StreamEvent,
    encode: Callable[[str, dict[str, Any]], str],
) -> str:
    """Convert an engine StreamEvent to this project's named SSE protocol."""

    if event.event == "chunk":
        return encode("chunk", {"text": event.text})
    if event.result is None:
        raise GenerationError("stream ended without a final result")
    response = GenerateResponse.model_validate(event.result)
    return encode("done", {"response": response.model_dump()})


def _openai_completion_stream(
    engine: InferenceEngine,
    prompt: str,
    params: Any,
    completion_id: str,
    created: int,
) -> Iterator[str]:
    """Convert engine chunks to OpenAI-style ``data:`` SSE frames."""

    try:
        for event in engine.stream_generate(prompt, params):
            if event.event == "chunk":
                payload = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.config.model_name,
                    "choices": [
                        {
                            "text": event.text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            elif event.result is not None:
                payload = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.config.model_name,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": _openai_finish_reason(
                                event.result.finish_reason
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": event.result.prompt_tokens,
                        "completion_tokens": event.result.generated_tokens,
                        "total_tokens": event.result.total_tokens,
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"
        # OpenAI clients use this sentinel rather than a named ``done`` event.
        yield "data: [DONE]\n\n"
    except (InvalidRequestError, EngineBusyError, GenerationError) as exc:
        payload = {
            "error": {
                "message": str(exc),
                "type": "inference_error",
            }
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"


def _openai_finish_reason(reason: str) -> str:
    """Map the internal timeout reason onto OpenAI's supported length reason."""

    return "stop" if reason == "stop" else "length"


app = create_app()
