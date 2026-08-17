# 2-Week Implementation Plan (Daily)

> Status update: the correctness/core milestone now includes typed results, static
> batch generation, SSE streaming, limits, API lifecycle handling, structured
> errors, tests, and a reproducible benchmark harness. Continuous batching and the
> performance experiments below remain future work.

## Week 1 — Correctness + Core Engine

### Day 1: Setup and Baseline
- Create venv, install dependencies, run smoke test.
- Start FastAPI app and validate `/health`.
- Record baseline latency for one prompt (CPU first).

### Day 2: Engine Contract
- Stabilize `InferenceEngine.generate()` return schema.
- Add request/response validation for generation params.
- Add deterministic test using fixed `seed`.

### Day 3: Decoding Modes
- Add explicit support for greedy (`do_sample=False`) and sampling.
- Validate behavior differences with tests.
- Add basic parameter guardrails and helpful errors.

### Day 4: Batch Inference (List of prompts)
- Add `generate_batch(prompts, params)` in engine.
- Handle variable prompt lengths with tokenizer padding.
- Add batch smoke test and compare throughput vs single requests.

### Day 5: Streaming Tokens (Server-side)
- Implement token streaming endpoint (SSE or chunked response).
- Return partial tokens plus final metadata.
- Add simple client script for stream consumption.

### Day 6: Benchmark Harness
- Add benchmark script for p50/p95 latency and tokens/sec.
- Test combinations of `max_new_tokens` and batch sizes.
- Save benchmark outputs in `benchmarks/`.

### Day 7: Refactor + Cleanup
- Split modules cleanly (`engine`, `sampler`, `schemas`, `api`).
- Improve logging with request IDs.
- Update README usage and known limitations.

## Week 2 — Performance + Advanced Features

### Day 8: KV Cache Awareness
- Validate and document cache behavior in autoregressive decode.
- Ensure no regressions in outputs.
- Measure decode-step performance impact.

### Day 9: Quantization Experiment
- Try `torch.int8`/bitsandbytes path (if GPU available).
- Compare memory + latency + output quality.
- Document tradeoffs in `docs/experiments.md`.

### Day 10: Continuous Batching (Basic)
- Implement queue-based scheduler window (e.g., 10–30ms).
- Merge nearby requests into one batch.
- Add bounded queue capacity, cancellation, fairness, and overload tests.
- Measure throughput improvements under concurrent load.

### Day 11: Prefix Caching
- Cache tokenized/encoded shared prompt prefixes.
- Reuse cached work for repeated prefixes.
- Add cache hit/miss metrics.

### Day 12: Reliability and Limits
- Add timeout handling and max prompt token limits.
- Add standardized error responses.
- Add structured logs for failures and slow requests.

### Day 13: API Compatibility Layer
- Add optional OpenAI-like `/v1/completions` shape.
- Keep internal engine contract unchanged.
- Add contract tests for both API modes.

### Day 14: Final Review + Packaging
- Run full tests and benchmark suite.
- Produce final report: metrics, architecture, next roadmap.
- Tag `v0.1.0-learning` locally.

## Definition of Done (v0.1)
- Single + batch generation works reliably.
- Basic streaming endpoint available.
- Benchmarks available and repeatable.
- Test suite runs green.
- Documentation reflects architecture + usage.
