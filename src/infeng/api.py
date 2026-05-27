from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .engine import InferenceEngine
from .sampler import SamplingParams
from .schemas import GenerateRequest, GenerateResponse

app = FastAPI(title="InfEng", version="0.1.0")
engine = InferenceEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": engine.config.model_name, "device": engine.device}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    params = SamplingParams(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        do_sample=req.do_sample,
        seed=req.seed,
    )
    result = engine.generate(req.prompt, params)
    return GenerateResponse(**result)

@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    params = SamplingParams(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        do_sample=req.do_sample,
        seed=req.seed,
    )

    def token_generated():
        for chunk in engine.stream_generate(req.prompt, params):
            yield chunk

    return StreamingResponse(
        token_generated(),
        media_type="text/plain"
    )

