# InfEng

A small, correctness-first inference server for learning LLM serving concepts with
`sshleifer/tiny-gpt2`.

For a guided explanation of how requests, batching, streaming, caching, and metrics
fit together, read [the architecture walkthrough](docs/architecture.md).

## Current capabilities

- Completion-only single and static-batch generation
- Bounded admission-window dynamic batching for concurrent requests
- Greedy and seeded sampling modes
- Exact logical token accounting for variable-length batches
- Server-sent event (SSE) streaming with final metadata
- Prompt, output, batch, concurrency, queue, and generation limits
- Liveness/readiness probes and structured API errors
- Engine and scheduler metrics plus an OpenAI-compatible completions endpoint
- Bounded exact-prompt tokenization cache with hit/miss/eviction metrics
- Repeatable latency and throughput benchmark reports

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the API:

```bash
uvicorn infeng.api:app --reload
```

Useful endpoints:

- `GET /live` — process liveness
- `GET /ready` or `GET /health` — model readiness
- `POST /generate` — one completion
- `POST /generate/batch` — a static batch of completions
- `POST /generate/stream` — SSE completion chunks plus final metadata
- `POST /v1/completions` — OpenAI-compatible text completions and streaming
- `GET /metrics` — engine token/latency and scheduler queue/batch metrics
- `GET /docs` — interactive OpenAPI documentation

Greedy request example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "content-type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":8,"do_sample":false}'
```

`text` is always the completion only; it does not repeat the prompt. Sampling-only
fields such as `temperature`, `top_k`, and `top_p` are ignored in greedy mode.

Run the bundled streaming client after starting the server:

```bash
python scripts/stream_client.py
```

## Configuration

The server reads these optional environment variables:

| Variable | Default |
| --- | ---: |
| `INFENG_MODEL_NAME` | `sshleifer/tiny-gpt2` |
| `INFENG_MODEL_REVISION` | unset |
| `INFENG_DEVICE` | `auto` |
| `INFENG_MAX_PROMPT_TOKENS` | `512` |
| `INFENG_MAX_NEW_TOKENS` | `512` |
| `INFENG_MAX_BATCH_SIZE` | `8` |
| `INFENG_MAX_CONCURRENT_REQUESTS` | `1` |
| `INFENG_SCHEDULER_BATCH_WINDOW_MS` | `20` |
| `INFENG_SCHEDULER_QUEUE_CAPACITY` | `64` |
| `INFENG_TOKENIZATION_CACHE_CAPACITY` | `256` (`0` disables it) |
| `INFENG_QUEUE_TIMEOUT_SECONDS` | `30` |
| `INFENG_GENERATION_TIMEOUT_SECONDS` | `60` |
| `INFENG_USE_KV_CACHE` | `true` |

Seeded sampling requests are isolated from the process-wide PyTorch RNG. Sampling
requests are serialized to avoid cross-request RNG interference.

The tokenization cache only reuses exact prompt preprocessing. It does not cache
model KV tensors and should not be interpreted as prefix-KV caching.

## Tests and benchmarks

```bash
python -m pytest -q
python scripts/quick_benchmark.py --device cpu --iterations 5 --concurrency 8
```

The benchmark performs warm-ups and measures single, static-batch, concurrent
dynamic-batch, and streaming paths. It writes a timestamped JSON report under
`benchmarks/`. Reports include p50/p95 latency, throughput, scheduler batch metrics,
time to first streamed chunk, hardware, dependency versions, and engine
configuration. Use `--disable-kv-cache` to measure the cache tradeoff. Performance
claims should always cite one of these reports.

## Known limitations and next roadmap

- Dynamic batching combines requests only before decode starts; it is not a paged
  scheduler that inserts requests into an already-running decode batch.
- Streaming chunks are decoded text fragments, not guaranteed one-token events.
- Client cancellation is cooperative and is checked between decoding steps.
- Quantization and prefix caching are not implemented.
- The learning model is intentionally tiny and is not suitable for production use.

The next performance milestone is a custom paged decode loop that can insert and
remove requests between token steps. That decode layer is also the prerequisite for
safe prefix-KV reuse. Hardware-specific FP16/int8 quantization experiments follow.
