"""Validated HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .sampler import SamplingParams


class GenerationOptions(BaseModel):
    """Flat JSON decoding fields shared by single and batch endpoints."""

    # Field constraints let FastAPI reject malformed requests before they enter a
    # scheduler queue or allocate model tensors.
    max_new_tokens: int = Field(32, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = True
    seed: int | None = None

    def to_sampling_params(self) -> SamplingParams:
        """Translate the HTTP model into the engine's transport-free dataclass."""

        return SamplingParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            seed=self.seed,
        )


class GenerateRequest(GenerationOptions):
    """One prompt plus the common generation options."""

    prompt: str = Field(..., min_length=1)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        # min_length rejects "", while strip also rejects whitespace-only input.
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class BatchGenerateRequest(GenerationOptions):
    """An explicit prompt batch whose rows share one decoding configuration."""

    prompts: list[str] = Field(..., min_length=1, max_length=64)

    @field_validator("prompts")
    @classmethod
    def prompts_must_not_be_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("prompts must not contain blank strings")
        return values


class GenerateResponse(BaseModel):
    """Wire representation of the engine's GenerationResult dataclass."""

    # from_attributes lets model_validate read dataclass attributes without first
    # converting the result to an untyped dictionary.
    model_config = ConfigDict(from_attributes=True)

    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    finish_reason: Literal["stop", "length", "timeout"]


class BatchGenerateResponse(BaseModel):
    """Keep batch output order identical to input prompt order."""

    outputs: list[GenerateResponse]


class ErrorResponse(BaseModel):
    """Native API error shape; /v1 endpoints use the OpenAI error envelope."""

    error: str
    detail: str
    request_id: str | None = None


class OpenAICompletionRequest(BaseModel):
    """Supported subset of the legacy OpenAI text completions contract."""

    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(16, ge=1, le=512)
    temperature: float = Field(1.0, ge=0.0, le=5.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def openai_prompt_must_not_be_blank(
        cls, value: str | list[str]
    ) -> str | list[str]:
        # Normalizing temporarily to a list keeps validation identical for the
        # single-prompt and multi-prompt forms allowed by the compatibility API.
        prompts = [value] if isinstance(value, str) else value
        if not prompts:
            raise ValueError("prompt list must not be empty")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompt must not be blank")
        return value

    def to_sampling_params(self) -> SamplingParams:
        """Map OpenAI's temperature=0 convention to our explicit greedy flag."""

        do_sample = self.temperature > 0
        return SamplingParams(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if do_sample else 1.0,
            top_k=0,
            top_p=self.top_p,
            do_sample=do_sample,
            seed=self.seed,
        )


class OpenAICompletionChoice(BaseModel):
    """One indexed completion in an OpenAI-compatible response."""

    text: str
    index: int
    logprobs: None = None
    finish_reason: Literal["stop", "length"]


class OpenAIUsage(BaseModel):
    """Logical token usage, excluding artificial batch padding."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAICompletionResponse(BaseModel):
    """Non-streaming OpenAI-compatible text completion envelope."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[OpenAICompletionChoice]
    usage: OpenAIUsage
