from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EngineConfig
from .sampler import SamplingParams


class InferenceEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.device = self._resolve_device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @staticmethod
    def validate_sampling_params(params: SamplingParams) -> None:
        if params.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if params.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < params.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if not params.do_sample and params.temperature != 1.0:
            raise ValueError("temperature must be 1.0 when do_sample=False (greedy)")
        if params.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not params.do_sample and params.top_k != 0:
            raise ValueError("top_k must be 0 when do_sampling = False")
        if not params.do_sample and params.top_p != 1.0:
            raise ValueError("top_p must be 1 when do_sampling = False")


    def generate(self, prompt: str, params: SamplingParams) -> dict:
        
        """Generate text given a prompt.
        
        Args:
            prompt: Input text to extend.
            params: Sampling parameters (temperature, top_k, seed, etc.).
            
            Returns:
                dict with keys:
                - text: Full decoded output (prompt + generated).
                - prompt_tokens: Number of tokens in input.
                - generated_tokens: Number of new generated tokens.
                - total_tokens: Total tokens in output.
                - model: Model name used.
                - latency_ms: Wall-clock generation time.
            """
        self.validate_sampling_params(params)

        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "text": decoded,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(generated_ids.shape[0]),
            "total_tokens": int(output_ids[0].shape[0]),
            "model": self.config.model_name,
            "latency_ms": round(latency_ms, 2),
        }
    
    def generate_batch(self, prompts: list[str], params:SamplingParams) -> list[dict]:
        self.validate_sampling_params(params)

        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True
        ).to(self.device)
        start = time.perf_counter()


        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                pad_tokens=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        results = []

        for i, output in enumerate(output_ids):
            prompt_tokens = int(inputs["attention_mask"][i].sum())
            generated_tokens = int(output.shape[0] - prompt_tokens)

            decoded = self.tokenizer.decode(
                output,
                skip_special_tokens = True
            )

            results.append(
                {
                    "text" : decoded,
                    "prompt_tokens" : prompt_tokens,
                    "generated_tokens" : generated_tokens,
                    "total_tokens" : int(output.shape[0]),
                    "model" : self.config.model_name,
                    "latency_ms" : round(latency_ms, 2),
                }
            )

            return results