# InfEng

A small, correctness-first inference server for learning LLM serving concepts with
`sshleifer/tiny-gpt2`.

## Current capabilities

- Completion-only single and static-batch generation
- Greedy and seeded sampling modes
- Exact logical token accounting for variable-length batches
- Server-sent event (SSE) streaming with final metadata
- Prompt, output, batch, concurrency, queue, and generation limits
- Liveness/readiness probes and structured API errors
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
| `INFENG_QUEUE_TIMEOUT_SECONDS` | `30` |
| `INFENG_GENERATION_TIMEOUT_SECONDS` | `60` |

Seeded sampling requests are isolated from the process-wide PyTorch RNG. Sampling
requests are serialized to avoid cross-request RNG interference.

## Tests and benchmarks

```bash
python -m pytest -q
python scripts/quick_benchmark.py --device cpu --iterations 5
```

The benchmark performs warm-ups, measures single, batch, and streaming paths, and
writes a timestamped JSON report under `benchmarks/`. Reports include p50/p95
latency, throughput, time to first streamed chunk, hardware, dependency versions,
and the engine configuration. Performance claims should always cite one of these
reports.

## Known limitations and next roadmap

- Static batching is explicit; there is no queue-based continuous batch scheduler.
- Streaming chunks are decoded text fragments, not guaranteed one-token events.
- Client cancellation is cooperative and is checked between decoding steps.
- Quantization and prefix caching are not implemented.
- The learning model is intentionally tiny and is not suitable for production use.

The next performance milestone is continuous batching with cancellation and
fairness, followed by KV-cache metrics, bounded prefix caching, and quantization
experiments.
