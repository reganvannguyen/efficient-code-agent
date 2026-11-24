"""Small HTTP client for llama.cpp's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from harness.config import DEFAULT_BASE_URL


class LlamaClientError(Exception):
    """Raised when a local model request fails or returns an invalid response."""


class LlamaTransportError(LlamaClientError):
    """Raised when the server cannot be reached or the connection fails."""


class LlamaTimeoutError(LlamaTransportError):
    """Raised when a model request or response read times out."""


class LlamaResponseTooLargeError(LlamaClientError):
    """Raised when the server response exceeds the bounded body size."""


MAX_RESPONSE_BODY_BYTES = 1_048_576


@dataclass(frozen=True)
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def chat(
    prompt: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = "local-model",
    timeout: float = 120,
    max_tokens: int = 256,
    response_schema: dict[str, object] | None = None,
    system_prompt: str | None = None,
) -> ChatResponse:
    """Send a chat request, optionally constrained by a JSON response schema."""
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
    if response_schema is not None and not isinstance(response_schema, dict):
        raise ValueError("response_schema must be a dictionary or None")
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
    ):
        raise ValueError("system_prompt must be a non-blank string or None")

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    request_payload = {
        "model": model.strip() or "local-model",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_schema is not None:
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_action",
                "strict": True,
                "schema": response_schema,
            },
        }
    payload = json.dumps(request_payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BODY_BYTES + 1)
            if len(body) > MAX_RESPONSE_BODY_BYTES:
                raise LlamaResponseTooLargeError("model response body exceeded 1 MiB")
    except HTTPError as exc:
        raise LlamaClientError(f"model server returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
            raise LlamaTimeoutError(f"model request timed out: {reason}") from exc
        raise LlamaTransportError(f"could not reach model server: {reason}") from exc

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
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
        RecursionError,
    ) as exc:
        raise LlamaClientError(
            "model server returned a malformed chat response"
        ) from exc


def _usage_count(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def probe_server(
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 1.5,
) -> tuple[bool, list[str]]:
    """Probe an OpenAI-compatible local server via /v1/models with /health fallback."""
    root = base_url.rstrip("/")
    try:
        with urlopen(f"{root}/v1/models", timeout=timeout) as response:
            if response.status == 200:
                body = response.read(MAX_RESPONSE_BODY_BYTES)
                try:
                    data = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return True, []
                models: list[str] = []
                raw_items = data.get("data") if isinstance(data, dict) else None
                if isinstance(raw_items, list):
                    for item in raw_items:
                        if isinstance(item, dict):
                            mid = item.get("id")
                            if (
                                isinstance(mid, str)
                                and mid.strip()
                                and mid not in models
                            ):
                                models.append(mid.strip())
                return True, models
    except HTTPError:
        pass
    except (URLError, TimeoutError, OSError, HTTPException):
        return False, []

    try:
        with urlopen(f"{root}/health", timeout=timeout) as response:
            return response.status == 200, []
    except (HTTPError, URLError, TimeoutError, OSError, HTTPException):
        return False, []


def tokenize(
    content: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
) -> list[int]:
    """Tokenize text using the llama.cpp server's /tokenize endpoint."""
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")
    if not content:
        return []

    payload = json.dumps({"content": content}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BODY_BYTES + 1)
            if len(body) > MAX_RESPONSE_BODY_BYTES:
                raise LlamaResponseTooLargeError(
                    "tokenize response body exceeded 1 MiB"
                )
    except HTTPError as exc:
        raise LlamaClientError(f"model server returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
            raise LlamaTimeoutError(f"tokenize request timed out: {reason}") from exc
        raise LlamaTransportError(f"could not reach model server: {reason}") from exc

    try:
        data = json.loads(body)
        tokens = data["tokens"]
        if not isinstance(tokens, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in tokens
        ):
            raise TypeError("tokens must be a list of integers")
        return tokens
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
    ) as exc:
        raise LlamaClientError(
            "model server returned a malformed tokenize response"
        ) from exc
