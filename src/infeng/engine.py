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

from .cache import TokenizationCache
from .config import EngineConfig
from .errors import EngineBusyError, GenerationError, InvalidRequestError
from .metrics import EngineMetrics
from .results import FinishReason, GenerationResult, StreamEvent
from .sampler import SamplingParams


class _RequestStoppingCriteria(StoppingCriteria):
    """Stop generation when its deadline expires or its consumer disconnects.

    Transformers calls this object after each decode step. That makes stopping
    cooperative: a currently executing model forward pass finishes, then the next
    token is not started.
    """

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
    """Own the tokenizer/model and implement all three generation modes.

    The engine deliberately has no HTTP knowledge. It accepts Python types and
    raises domain errors, while ``api.py`` converts those errors and results into
    wire formats. This separation also lets tests and benchmarks call the same core
    behavior without starting a web server.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Load one model instance and initialize its concurrency controls."""

        self.config = config or EngineConfig()
        self.device = self._resolve_device(self.config.device)
        # Passing no revision follows the Hugging Face default branch. Supplying a
        # revision pins both tokenizer and model to the same upstream state.
        pretrained_kwargs = (
            {"revision": self.config.model_revision} if self.config.model_revision else {}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, **pretrained_kwargs
        )
        if self.tokenizer.pad_token is None:
            # GPT-2 was trained without a padding token. Reusing EOS is safe as long
            # as attention_mask marks those artificial positions as padding.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models must be left padded: the last column then contains a
        # real prompt token for every row, where next-token logits are read.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **pretrained_kwargs
        )
        self.model.to(self.device)
        # eval() disables training-only behavior such as dropout. Individual calls
        # additionally use torch.inference_mode() to avoid autograd allocations.
        self.model.eval()
        # The semaphore caps expensive model calls. The sampling lock is narrower:
        # it protects process-wide PyTorch RNG state only when randomness is used.
        self._slots = threading.BoundedSemaphore(self.config.max_concurrent_requests)
        self._sampling_lock = threading.Lock()
        self.context_window = self._resolve_context_window()
        self.metrics = EngineMetrics()
        self.tokenization_cache = TokenizationCache(
            self.config.tokenization_cache_capacity
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise InvalidRequestError("CUDA was requested but is not available")
        return device

    def _resolve_context_window(self) -> int:
        """Find the most conservative finite context length advertised upstream."""

        # Different model families use different configuration names. Some
        # tokenizers use a huge sentinel integer to mean "unknown", so those values
        # are filtered rather than treated as real context windows.
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
        """Validate decoding mathematics independently of the HTTP schema."""

        # Engine-level validation is still necessary because callers can use this
        # class directly without going through Pydantic/FastAPI.
        if params.max_new_tokens < 1:
            raise InvalidRequestError("max_new_tokens must be >= 1")
        if params.temperature <= 0:
            raise InvalidRequestError("temperature must be > 0")
        if not 0.0 < params.top_p <= 1.0:
            raise InvalidRequestError("top_p must be in (0, 1]")
        if params.top_k < 0:
            raise InvalidRequestError("top_k must be >= 0")

    def _validate_request_size(self, prompt_tokens: int, params: SamplingParams) -> None:
        """Enforce configured and model-native size limits before allocation."""

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
        """Translate our stable SamplingParams contract into Transformers kwargs."""

        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": self.config.use_kv_cache,
            "stopping_criteria": StoppingCriteriaList([stopping]),
        }
        if params.do_sample:
            # Omitting these fields in greedy mode is intentional. Passing sampling
            # controls while do_sample=False causes warnings in Transformers and
            # made otherwise-valid API defaults fail in the original implementation.
            kwargs.update(
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
            )
        return kwargs

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenize on CPU, using the bounded exact-prompt cache when possible."""

        cached = self.tokenization_cache.get(prompt)
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
        )
        value = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        self.tokenization_cache.put(prompt, value)
        return value

    def _tokenize_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        """Build a left-padded tensor batch from independently cached prompts."""

        encoded = [self._tokenize_prompt(prompt) for prompt in prompts]
        lengths = [int(item["input_ids"].shape[1]) for item in encoded]
        width = max(lengths)
        # Start with a rectangle of pad IDs, then right-align each real prompt.
        # attention_mask is the source of truth for each prompt's logical length.
        input_ids = torch.full(
            (len(prompts), width),
            fill_value=self.tokenizer.pad_token_id,
            dtype=encoded[0]["input_ids"].dtype,
        )
        attention_mask = torch.zeros(
            (len(prompts), width),
            dtype=encoded[0]["attention_mask"].dtype,
        )
        for index, (item, length) in enumerate(zip(encoded, lengths, strict=True)):
            input_ids[index, -length:] = item["input_ids"][0]
            attention_mask[index, -length:] = item["attention_mask"][0]
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    @contextmanager
    def _generation_slot(self) -> Iterator[None]:
        """Acquire/release scarce model capacity even when generation raises."""

        acquired = self._slots.acquire(timeout=self.config.queue_timeout_seconds)
        if not acquired:
            raise EngineBusyError("the inference engine is busy; retry later")
        try:
            yield
        finally:
            self._slots.release()

    @contextmanager
    def _randomness(self, params: SamplingParams) -> Iterator[None]:
        """Serialize sampling and isolate explicitly seeded RNG changes."""

        if not params.do_sample:
            yield
            return

        with self._sampling_lock:
            if params.seed is None:
                # Unseeded sampling is serialized but intentionally advances the
                # normal process RNG so consecutive calls remain genuinely random.
                yield
                return

            cuda_devices = [torch.cuda.current_device()] if self.device == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                # fork_rng restores the caller's CPU/CUDA RNG states on exit. A
                # seeded request therefore cannot perturb unrelated later requests.
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
        """Decode completion-only IDs and calculate logical, unpadded usage."""

        # output begins with the full padded input width for every batch row. Slice
        # by input_width, but bill using prompt_tokens; confusing those two values
        # was the original variable-length batch accounting bug.
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
        # Batched generation pads rows that stop early. Counting through the first
        # EOS (inclusive) avoids billing the trailing padding added for other rows.
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
        """Generate completion-only text for one prompt.

        Flow: validate -> tokenize -> acquire capacity/RNG context -> generate ->
        decode a typed result -> record metrics.
        """

        self.validate_sampling_params(params)
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("prompt must be a non-blank string")
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in self._tokenize_prompt(prompt).items()
        }
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
            self.metrics.record_failure("single")
            raise
        except Exception as exc:
            self.metrics.record_failure("single")
            raise GenerationError("model generation failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        result = self._completion_result(
            output_ids[0], input_width, input_width, latency_ms, stopping.timed_out
        )
        self.metrics.record_results("single", [result], latency_ms)
        return result

    def generate_batch(
        self, prompts: list[str], params: SamplingParams
    ) -> list[GenerationResult]:
        """Generate completions for a bounded, variable-length prompt batch.

        All rows share SamplingParams because Transformers performs one physical
        decode call. The scheduler groups requests by exact params before invoking
        this method.
        """

        self.validate_sampling_params(params)
        if not prompts:
            raise InvalidRequestError("prompts must not be empty")
        if len(prompts) > self.config.max_batch_size:
            raise InvalidRequestError(
                f"batch exceeds the configured limit of {self.config.max_batch_size} prompts"
            )
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise InvalidRequestError("every prompt must be a non-blank string")

        inputs = self._tokenize_batch(prompts)
        # Sum masks, not tensor width, because shorter rows include left padding.
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
            self.metrics.record_failure("batch")
            raise
        except Exception as exc:
            self.metrics.record_failure("batch")
            raise GenerationError("batch model generation failed") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        results = [
            self._completion_result(
                output,
                input_width,
                prompt_lengths[index],
                latency_ms,
                stopping.timed_out,
            )
            for index, output in enumerate(output_ids)
        ]
        self.metrics.record_results("batch", results, latency_ms)
        return results

    def stream_generate(
        self, prompt: str, params: SamplingParams
    ) -> Iterator[StreamEvent]:
        """Yield completion chunks followed by one canonical final result.

        Transformers generation is a blocking producer while TextIteratorStreamer
        is a blocking consumer. A background thread lets both make progress and the
        outer Python generator forward chunks to the HTTP response immediately.
        """

        self.validate_sampling_params(params)
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("prompt must be a non-blank string")
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in self._tokenize_prompt(prompt).items()
        }
        input_width = int(inputs["input_ids"].shape[1])
        self._validate_request_size(input_width, params)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            # Without skip_prompt the original input is echoed as the first chunk.
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
            """Producer thread: run the model and capture result or exception."""

            try:
                with self._generation_slot(), self._randomness(params), torch.inference_mode():
                    state["output_ids"] = self.model.generate(
                        **inputs,
                        **self._generation_kwargs(params, stopping),
                        streamer=streamer,
                    )
            except Exception as exc:
                state["error"] = exc
                # If generation fails before calling streamer.end(), the consumer
                # would block forever. Sending the stop signal unblocks iteration so
                # the exception can be re-raised on the request thread.
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
                    self.metrics.record_failure("stream")
                    raise error
                self.metrics.record_failure("stream")
                raise GenerationError("streaming model generation failed") from error
            latency_ms = (time.perf_counter() - start) * 1000
            result = self._completion_result(
                state["output_ids"][0],
                input_width,
                input_width,
                latency_ms,
                stopping.timed_out,
            )
            self.metrics.record_results("stream", [result], latency_ms)
            yield StreamEvent(event="done", result=result)
        finally:
            # Closing an HTTP stream sets this flag. The stopping criterion observes
            # it between tokens, giving cooperative client-disconnect cancellation.
            cancelled.set()
            if thread.is_alive():
                thread.join(timeout=5.0)

    def metrics_snapshot(self) -> dict[str, Any]:
        """Combine counters with static runtime facts for the metrics endpoint."""

        return {
            **self.metrics.snapshot(),
            "model": self.config.model_name,
            "device": self.device,
            "context_window": self.context_window,
            "kv_cache_enabled": self.config.use_kv_cache,
            "tokenization_cache": self.tokenization_cache.snapshot(),
        }
