"""Core model loading and text-generation implementation."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .config import EngineConfig
from .errors import EngineBusyError, GenerationError, InvalidRequestError
from .results import FinishReason, GenerationResult, StreamEvent
from .sampler import SamplingParams


class _RequestStoppingCriteria(StoppingCriteria):
    """Stop generation when its deadline expires or its consumer disconnects."""

    def __init__(self, deadline: float, cancelled: threading.Event | None = None) -> None:
        self.deadline = deadline
        self.cancelled = cancelled
        self.timed_out = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if self.cancelled is not None and self.cancelled.is_set():
            return True
        if time.perf_counter() >= self.deadline:
            self.timed_out = True
            return True
        return False


class InferenceEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.device = self._resolve_device(self.config.device)
        pretrained_kwargs = (
            {"revision": self.config.model_revision} if self.config.model_revision else {}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, **pretrained_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **pretrained_kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        self._slots = threading.BoundedSemaphore(self.config.max_concurrent_requests)
        self._sampling_lock = threading.Lock()
        self.context_window = self._resolve_context_window()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise InvalidRequestError("CUDA was requested but is not available")
        return device

    def _resolve_context_window(self) -> int:
        candidates = [
            getattr(self.model.config, "max_position_embeddings", None),
            getattr(self.model.config, "n_positions", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        finite = [
            int(value)
            for value in candidates
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        return (
            min(finite)
            if finite
            else self.config.max_prompt_tokens + self.config.max_new_tokens
        )

    @staticmethod
    def validate_sampling_params(params: SamplingParams) -> None:
        if params.max_new_tokens < 1:
            raise InvalidRequestError("max_new_tokens must be >= 1")
        if params.temperature <= 0:
            raise InvalidRequestError("temperature must be > 0")
        if not 0.0 < params.top_p <= 1.0:
            raise InvalidRequestError("top_p must be in (0, 1]")
        if params.top_k < 0:
            raise InvalidRequestError("top_k must be >= 0")

    def _validate_request_size(self, prompt_tokens: int, params: SamplingParams) -> None:
        if params.max_new_tokens > self.config.max_new_tokens:
            raise InvalidRequestError(
                f"max_new_tokens exceeds the configured limit of {self.config.max_new_tokens}"
            )
        if prompt_tokens < 1:
            raise InvalidRequestError("prompt must contain at least one token")
        if prompt_tokens > self.config.max_prompt_tokens:
            raise InvalidRequestError(
                f"prompt exceeds the configured limit of {self.config.max_prompt_tokens} tokens"
            )
        if prompt_tokens + params.max_new_tokens > self.context_window:
            raise InvalidRequestError(
                "prompt tokens plus max_new_tokens exceed the model context window "
                f"of {self.context_window}"
            )

    def _generation_kwargs(
        self,
        params: SamplingParams,
        stopping: _RequestStoppingCriteria,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([stopping]),
        }
        if params.do_sample:
            kwargs.update(
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
            )
        return kwargs

    @contextmanager
    def _generation_slot(self) -> Iterator[None]:
        acquired = self._slots.acquire(timeout=self.config.queue_timeout_seconds)
        if not acquired:
            raise EngineBusyError("the inference engine is busy; retry later")
        try:
            yield
        finally:
            self._slots.release()

    @contextmanager
    def _randomness(self, params: SamplingParams) -> Iterator[None]:
        if not params.do_sample:
            yield
            return

        with self._sampling_lock:
            if params.seed is None:
                yield
                return

            cuda_devices = [torch.cuda.current_device()] if self.device == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(params.seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(params.seed)
                yield

    def _completion_result(
        self,
        output: torch.Tensor,
        input_width: int,
        prompt_tokens: int,
        latency_ms: float,
        timed_out: bool,
    ) -> GenerationResult:
        generated = output[input_width:]
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = {int(token_id) for token_id in eos_ids}

        generated_count = int(generated.shape[0])
        stopped_on_eos = False
        for index, token_id in enumerate(generated.tolist()):
            if int(token_id) in eos_set:
                generated_count = index + 1
                stopped_on_eos = True
                break

        completion_ids = generated[:generated_count]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        finish_reason: FinishReason
        if timed_out:
            finish_reason = "timeout"
        elif stopped_on_eos:
            finish_reason = "stop"
        else:
            finish_reason = "length"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_count,
            total_tokens=prompt_tokens + generated_count,
            model=self.config.model_name,
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
        )

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        """Generate completion-only text for one prompt."""

        self.validate_sampling_params(params)
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("prompt must be a non-blank string")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_width = int(inputs["input_ids"].shape[1])
        self._validate_request_size(input_width, params)
        stopping = _RequestStoppingCriteria(
            time.perf_counter() + self.config.generation_timeout_seconds
        )

        start = time.perf_counter()
        try:
            with self._generation_slot(), self._randomness(params), torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(params, stopping),
                )
        except (EngineBusyError, InvalidRequestError):
            raise
        except Exception as exc:
            raise GenerationError("model generation failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return self._completion_result(
            output_ids[0], input_width, input_width, latency_ms, stopping.timed_out
        )

    def generate_batch(
        self, prompts: list[str], params: SamplingParams
    ) -> list[GenerationResult]:
        """Generate completions for a bounded, variable-length prompt batch."""

        self.validate_sampling_params(params)
        if not prompts:
            raise InvalidRequestError("prompts must not be empty")
        if len(prompts) > self.config.max_batch_size:
            raise InvalidRequestError(
                f"batch exceeds the configured limit of {self.config.max_batch_size} prompts"
            )
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise InvalidRequestError("every prompt must be a non-blank string")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_lengths = [
            int(length) for length in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        for prompt_tokens in prompt_lengths:
            self._validate_request_size(prompt_tokens, params)
        input_width = int(inputs["input_ids"].shape[1])
        stopping = _RequestStoppingCriteria(
            time.perf_counter() + self.config.generation_timeout_seconds
        )

        start = time.perf_counter()
        try:
            with self._generation_slot(), self._randomness(params), torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(params, stopping),
                )
        except (EngineBusyError, InvalidRequestError):
            raise
        except Exception as exc:
            raise GenerationError("batch model generation failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        return [
            self._completion_result(
                output,
                input_width,
                prompt_lengths[index],
                latency_ms,
                stopping.timed_out,
            )
            for index, output in enumerate(output_ids)
        ]

    def stream_generate(
        self, prompt: str, params: SamplingParams
    ) -> Iterator[StreamEvent]:
        """Yield completion chunks followed by one canonical final result."""

        self.validate_sampling_params(params)
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("prompt must be a non-blank string")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_width = int(inputs["input_ids"].shape[1])
        self._validate_request_size(input_width, params)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        cancelled = threading.Event()
        stopping = _RequestStoppingCriteria(
            time.perf_counter() + self.config.generation_timeout_seconds,
            cancelled,
        )
        state: dict[str, Any] = {}
        start = time.perf_counter()

        def run_generation() -> None:
            try:
                with self._generation_slot(), self._randomness(params), torch.inference_mode():
                    state["output_ids"] = self.model.generate(
                        **inputs,
                        **self._generation_kwargs(params, stopping),
                        streamer=streamer,
                    )
            except Exception as exc:
                state["error"] = exc
                streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(
            target=run_generation,
            daemon=True,
            name="infeng-stream",
        )
        thread.start()

        try:
            for text in streamer:
                if text:
                    yield StreamEvent(event="chunk", text=text)
            thread.join()
            if "error" in state:
                error = state["error"]
                if isinstance(error, (EngineBusyError, InvalidRequestError)):
                    raise error
                raise GenerationError("streaming model generation failed") from error
            latency_ms = (time.perf_counter() - start) * 1000
            result = self._completion_result(
                state["output_ids"][0],
                input_width,
                input_width,
                latency_ms,
                stopping.timed_out,
            )
            yield StreamEvent(event="done", result=result)
        finally:
            cancelled.set()
            if thread.is_alive():
                thread.join(timeout=5.0)
