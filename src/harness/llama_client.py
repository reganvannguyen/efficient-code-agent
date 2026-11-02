"""Small HTTP client for llama.cpp's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LlamaClientError(Exception):
    """Raised when a local model request fails or returns an invalid response."""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def chat(
    prompt: str,
    *,
    base_url: str = "http://127.0.0.1:9931",
    model: str = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    timeout: float = 120,
    max_tokens: int = 256,
) -> ChatResponse:
    """Send one non-streaming user message to a llama.cpp chat server."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be blank")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be positive")

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        raise LlamaClientError(f"model server returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LlamaClientError(f"could not reach model server: {reason}") from exc

    try:
        data = json.loads(body)
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise TypeError("message content is not text")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            prompt_tokens=_usage_count(usage, "prompt_tokens"),
            completion_tokens=_usage_count(usage, "completion_tokens"),
            total_tokens=_usage_count(usage, "total_tokens"),
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ) as exc:
        raise LlamaClientError(
            "model server returned a malformed chat response"
        ) from exc


def _usage_count(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
