import json
from urllib.error import HTTPError, URLError

import pytest

from harness import llama_client
from harness.llama_client import ChatResponse, LlamaClientError


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_chat_posts_expected_payload_and_parses_response(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.method
        seen["headers"] = request.headers
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeHTTPResponse(
            json.dumps(
                {
                    "choices": [{"message": {"content": "Hello"}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 2,
                        "total_tokens": 13,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(llama_client, "urlopen", fake_urlopen)
    response = llama_client.chat(
        "Say hello", base_url="http://localhost:9000/", model="test.gguf", timeout=4
    )

    assert response == ChatResponse("Hello", 11, 2, 13)
    assert seen == {
        "url": "http://localhost:9000/v1/chat/completions",
        "method": "POST",
        "headers": {"Content-type": "application/json"},
        "payload": {
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 256,
            "stream": False,
        },
        "timeout": 4,
    }


def test_chat_accepts_missing_or_unusable_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse(
            b'{"choices":[{"message":{"content":"ok"}}]}'
        ),
    )
    assert llama_client.chat("hello") == ChatResponse("ok")

    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse(
            b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":2.5}}'
        ),
    )
    assert llama_client.chat("hello") == ChatResponse("ok")


@pytest.mark.parametrize("prompt", ["", "   "])
def test_chat_rejects_blank_prompt(prompt) -> None:
    with pytest.raises(ValueError, match="prompt"):
        llama_client.chat(prompt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"max_tokens": 0},
        {"max_tokens": 1.5},
    ],
    ids=[
        "zero-timeout",
        "nan-timeout",
        "inf-timeout",
        "zero-tokens",
        "float-tokens",
    ],
)
def test_chat_rejects_invalid_limits(kwargs) -> None:
    with pytest.raises(ValueError):
        llama_client.chat("hello", **kwargs)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff",
        b"{}",
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":3}}]}',
    ],
)
def test_chat_reports_malformed_response(monkeypatch, body) -> None:
    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse(body),
    )

    with pytest.raises(LlamaClientError, match="malformed chat response"):
        llama_client.chat("hello")


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("http://localhost", 500, "broken", {}, None),
        URLError("connection refused"),
        TimeoutError("timed out"),
    ],
    ids=["http", "connection", "timeout"],
)
def test_chat_wraps_http_connection_and_timeout_errors(monkeypatch, error) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(llama_client, "urlopen", fail)

    with pytest.raises(LlamaClientError):
        llama_client.chat("hello")
