"""FastAPI boundary for the inference engine."""

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
from .schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
)

logger = logging.getLogger("infeng.api")
EngineFactory = Callable[[], InferenceEngine]


def _default_engine_factory() -> InferenceEngine:
    return InferenceEngine(EngineConfig.from_env())


def create_app(engine_factory: EngineFactory = _default_engine_factory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.engine = None
        application.state.startup_error = None
        try:
            application.state.engine = engine_factory()
        except Exception as exc:
            application.state.startup_error = exc
            logger.exception("engine_startup_failed")
        yield
        application.state.engine = None

    application = FastAPI(title="InfEng", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]):
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
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "detail": detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

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
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            startup_error = getattr(request.app.state, "startup_error", None)
            detail = "model is not ready"
            if startup_error is not None:
                detail += f" ({type(startup_error).__name__})"
            raise EngineNotReadyError(detail)
        return engine

    @application.get("/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", response_model=None)
    @application.get("/health", response_model=None)
    def ready(request: Request) -> JSONResponse | dict[str, str]:
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return error_response(request, 503, "not_ready", "model is not ready")
        return {
            "status": "ok",
            "model": engine.config.model_name,
            "device": engine.device,
        }

    @application.post("/generate", response_model=GenerateResponse)
    def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
        result = get_engine(request).generate(body.prompt, body.to_sampling_params())
        return GenerateResponse.model_validate(result)

    @application.post("/generate/batch", response_model=BatchGenerateResponse)
    def generate_batch(
        request: Request, body: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        results = get_engine(request).generate_batch(
            body.prompts, body.to_sampling_params()
        )
        return BatchGenerateResponse(
            outputs=[GenerateResponse.model_validate(result) for result in results]
        )

    @application.post("/generate/stream")
    def generate_stream(request: Request, body: GenerateRequest) -> StreamingResponse:
        engine = get_engine(request)

        def encode(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def event_stream() -> Iterator[str]:
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

    return application


def _encode_stream_event(
    event: StreamEvent,
    encode: Callable[[str, dict[str, Any]], str],
) -> str:
    if event.event == "chunk":
        return encode("chunk", {"text": event.text})
    if event.result is None:
        raise GenerationError("stream ended without a final result")
    response = GenerateResponse.model_validate(event.result)
    return encode("done", {"response": response.model_dump()})


app = create_app()
