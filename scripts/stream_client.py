import httpx

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/generate/stream",
    json={
        "prompt": "Explain machine learning",
        "max_new_tokens": 32,
        "do_sample": False,
    },
) as response:
    for chunk in response.iter_text():
        print(chunk, end="", flush=True)