"""Minimal SSE client for the streaming generation endpoint."""

from __future__ import annotations

import json

import httpx


def main() -> None:
    """Consume the project's small named-event SSE protocol with plain httpx."""

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
            # One SSE event arrives as an ``event:`` line followed by a JSON
            # ``data:`` line and a blank separator. httpx exposes those lines as the
            # server flushes them, so partial text can be printed immediately.
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if event_name == "chunk":
                    print(payload["text"], end="", flush=True)
                elif event_name == "done":
                    # The final event repeats the canonical non-streaming metadata,
                    # making token counts and finish reason available to clients.
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
