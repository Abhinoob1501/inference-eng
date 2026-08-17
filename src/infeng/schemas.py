"""Validated HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .sampler import SamplingParams


class GenerationOptions(BaseModel):
    max_new_tokens: int = Field(32, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = True
    seed: int | None = None

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            seed=self.seed,
        )


class GenerateRequest(GenerationOptions):
    prompt: str = Field(..., min_length=1)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class BatchGenerateRequest(GenerationOptions):
    prompts: list[str] = Field(..., min_length=1, max_length=64)

    @field_validator("prompts")
    @classmethod
    def prompts_must_not_be_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("prompts must not contain blank strings")
        return values


class GenerateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    finish_reason: Literal["stop", "length", "timeout"]


class BatchGenerateResponse(BaseModel):
    outputs: list[GenerateResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
