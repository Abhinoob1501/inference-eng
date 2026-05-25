from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams
import pytest

def test_engine_smoke_generation() -> None:
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    output = engine.generate("Hello", SamplingParams(max_new_tokens=8, top_k=0, do_sample=False))

    assert isinstance(output["text"], str)
    assert output["prompt_tokens"] > 0
    assert output["total_tokens"] >= output["prompt_tokens"]


def test_deterministic_generation():
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    params = SamplingParams(
        max_new_tokens=16,
        temperature=1.0,
        do_sample=True,
        seed=42
    )

    output1 = engine.generate("Hello World", params)
    output2 = engine.generate("Hello World", params)

    assert output1["text"] == output2["text"], "Fixed seed should produce identical output"

def test_greedy_ignores_seed():
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2",device="cpu"))
    params_a = SamplingParams(
        max_new_tokens = 8,
        do_sample = False,
        top_k=0,
        seed = 67
    )
    params_b = SamplingParams(
        max_new_tokens = 8,
        do_sample = False,
        top_k=0,
        seed = 420
    )

    output_a = engine.generate("Hello", params_a)
    output_b = engine.generate("Hello", params_b)
    assert output_a["text"] == output_b["text"]

def test_invalid_sampling_params_raise():
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    with pytest.raises(ValueError):
        engine.generate("Hello", SamplingParams(max_new_tokens=0))

def test_batch_generation():
    engine = InferenceEngine(EngineConfig(model_name = "sshleifer/tiny-gpt2", device="cpu"))

    outputs = engine.generate_batch(
        ["Hi", "Hello there"], 
        SamplingParams(max_new_tokens=4, do_sample=False, top_k = 0)
    )

    assert len(outputs) == 2
    assert all("text" in o for o in outputs)
    assert all(o["prompt_tokens"] > 0 for o in outputs)