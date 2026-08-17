"""Reproducible latency and throughput benchmark for InfEng."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import transformers

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams
from infeng.scheduler import DynamicBatchScheduler

PROMPTS = [
    "Write one sentence about machine learning.",
    "Explain inference engines in simple terms.",
    "What is token sampling?",
]


def synchronize(device: str) -> None:
    """Wait for queued CUDA kernels so wall-clock measurements are honest."""

    # CPU operations are synchronous; CUDA launches are normally asynchronous.
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float]:
    """Return the compact latency fields written for every benchmark mode."""

    return {
        "mean_ms": round(mean(values), 2),
        "p50_ms": round(median(values), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
    }


def timed_generate(engine: InferenceEngine, prompt: str, params: SamplingParams):
    """Measure one direct engine call including tokenization/cache lookup."""

    synchronize(engine.device)
    started = time.perf_counter()
    result = engine.generate(prompt, params)
    synchronize(engine.device)
    return result, (time.perf_counter() - started) * 1000


def timed_scheduled_generate(
    scheduler: DynamicBatchScheduler, prompt: str, params: SamplingParams
):
    """Measure end-to-end scheduler time, including its admission window."""

    started = time.perf_counter()
    result = scheduler.generate(prompt, params)
    return result, (time.perf_counter() - started) * 1000


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run comparable single, static, dynamic, and streaming workloads."""

    config = EngineConfig(
        model_name=args.model,
        device=args.device,
        scheduler_batch_window_ms=args.batch_window_ms,
        use_kv_cache=not args.disable_kv_cache,
    )
    engine = InferenceEngine(config)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, do_sample=False)

    # Warm-ups absorb one-time model/kernel/tokenizer initialization so measured
    # iterations represent steady-state serving rather than startup.
    for index in range(args.warmups):
        engine.generate(PROMPTS[index % len(PROMPTS)], params)

    single_latencies: list[float] = []
    single_generated = 0
    # Single mode sends every prompt as an independent sequential model call.
    single_started = time.perf_counter()
    for _ in range(args.iterations):
        for prompt in PROMPTS:
            result, latency = timed_generate(engine, prompt, params)
            single_latencies.append(latency)
            single_generated += result.generated_tokens
    synchronize(engine.device)
    single_wall_seconds = time.perf_counter() - single_started

    # Static batch mode sends the same prompts in one tensor operation. Its timer is
    # intentionally reset here; the original benchmark accidentally included the
    # preceding single-request duration in its batch result.
    batch_latencies: list[float] = []
    batch_generated = 0
    batch_started = time.perf_counter()
    for _ in range(args.iterations):
        synchronize(engine.device)
        started = time.perf_counter()
        results = engine.generate_batch(PROMPTS, params)
        synchronize(engine.device)
        batch_latencies.append((time.perf_counter() - started) * 1000)
        batch_generated += sum(result.generated_tokens for result in results)
    batch_wall_seconds = time.perf_counter() - batch_started

    # Dynamic mode starts with independent concurrent callers. The scheduler must
    # discover compatibility and construct physical batches itself.
    scheduler = DynamicBatchScheduler(engine)
    scheduler.start()
    dynamic_latencies: list[float] = []
    dynamic_generated = 0
    dynamic_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for _ in range(args.iterations):
                futures = [
                    executor.submit(
                        timed_scheduled_generate,
                        scheduler,
                        PROMPTS[index % len(PROMPTS)],
                        params,
                    )
                    for index in range(args.concurrency)
                ]
                for future in futures:
                    result, latency = future.result()
                    dynamic_generated += result.generated_tokens
                    dynamic_latencies.append(latency)
        synchronize(engine.device)
        dynamic_wall_seconds = time.perf_counter() - dynamic_started
        scheduler_metrics = scheduler.snapshot()
    finally:
        scheduler.close()

    # Streaming measures user-perceived time to first decoded fragment separately
    # from total completion latency.
    first_chunk_latencies: list[float] = []
    stream_latencies: list[float] = []
    for _ in range(args.iterations):
        synchronize(engine.device)
        started = time.perf_counter()
        first_chunk = None
        for event in engine.stream_generate(PROMPTS[0], params):
            if event.event == "chunk" and first_chunk is None:
                synchronize(engine.device)
                first_chunk = (time.perf_counter() - started) * 1000
        synchronize(engine.device)
        stream_latencies.append((time.perf_counter() - started) * 1000)
        if first_chunk is not None:
            first_chunk_latencies.append(first_chunk)

    # Store environment/config beside results so numbers can be reproduced and
    # performance claims are not detached from hardware or dependency versions.
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "config": asdict(config),
        "benchmark": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "prompt_count": len(PROMPTS),
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency,
        },
        "results": {
            "single": {
                **latency_summary(single_latencies),
                "generated_tokens_per_second": round(
                    single_generated / single_wall_seconds, 2
                ),
            },
            "batch": {
                **latency_summary(batch_latencies),
                "generated_tokens_per_second": round(
                    batch_generated / batch_wall_seconds, 2
                ),
            },
            "dynamic_batch": {
                **latency_summary(dynamic_latencies),
                "generated_tokens_per_second": round(
                    dynamic_generated / dynamic_wall_seconds, 2
                ),
                "scheduler": scheduler_metrics,
            },
            "stream": {
                "time_to_first_chunk": latency_summary(first_chunk_latencies),
                "total_latency": latency_summary(stream_latencies),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    """Define command-line controls for repeatable benchmark scenarios."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    return parser.parse_args()


def main() -> None:
    """Run the benchmark, print a summary, and persist the complete JSON report."""

    args = parse_args()
    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output_dir / f"benchmark-{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))
    print(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
