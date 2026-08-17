"""Minimal SSE client for the streaming generation endpoint."""

from __future__ import annotations

import json

import httpx


def main() -> None:
    event_name = "message"
    with httpx.stream(
        "POST",
        "http://127.0.0.1:8000/generate/stream",
        json={
            "prompt": "Explain machine learning",
            "max_new_tokens": 32,
            "do_sample": False,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if event_name == "chunk":
                    print(payload["text"], end="", flush=True)
                elif event_name == "done":
                    metadata = payload["response"]
                    print(
                        f"\n[{metadata['finish_reason']}; "
                        f"{metadata['generated_tokens']} generated tokens; "
                        f"{metadata['latency_ms']} ms]"
                    )
                elif event_name == "error":
                    raise RuntimeError(payload["detail"])


if __name__ == "__main__":
    main()
