import json
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest

from harness import llama_client
from harness.llama_client import (
    ChatResponse,
    LlamaClientError,
    LlamaResponseTooLargeError,
    LlamaTimeoutError,
    LlamaTransportError,
)


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_size = size
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


def test_chat_forwards_optional_system_prompt_and_response_schema(monkeypatch) -> None:
    schema = {
        "type": "object",
        "required": ["kind"],
        "additionalProperties": False,
    }
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        return FakeHTTPResponse(b'{"choices":[{"message":{"content":"{}"}}]}')

    monkeypatch.setattr(llama_client, "urlopen", fake_urlopen)

    assert llama_client.chat(
        "Return JSON",
        response_schema=schema,
        system_prompt="System instruction",
    ) == ChatResponse("{}")
    assert seen["payload"]["messages"] == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Return JSON"},
    ]
    assert seen["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_action",
            "strict": True,
            "schema": schema,
        },
    }


def test_chat_rejects_invalid_response_schema_before_request(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(llama_client, "urlopen", lambda *_a, **_k: calls.append(1))

    with pytest.raises(ValueError, match="response_schema"):
        llama_client.chat("Return JSON", response_schema=["not", "an", "object"])

    assert calls == []


@pytest.mark.parametrize("system_prompt", ["", "  ", 3])
def test_chat_rejects_invalid_system_prompt_before_request(
    monkeypatch, system_prompt
) -> None:
    calls = []
    monkeypatch.setattr(llama_client, "urlopen", lambda *_a, **_k: calls.append(1))

    with pytest.raises(ValueError, match="system_prompt"):
        llama_client.chat("hello", system_prompt=system_prompt)

    assert calls == []


def test_chat_reads_at_most_one_mib_plus_sentinel(monkeypatch) -> None:
    response = FakeHTTPResponse(b"x" * (llama_client.MAX_RESPONSE_BODY_BYTES + 1))
    monkeypatch.setattr(llama_client, "urlopen", lambda *_a, **_k: response)
    with pytest.raises(LlamaResponseTooLargeError):
        llama_client.chat("hello")
    assert response.read_size == llama_client.MAX_RESPONSE_BODY_BYTES + 1


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

    expected = (
        LlamaTimeoutError if isinstance(error, TimeoutError) else LlamaClientError
    )
    with pytest.raises(expected):
        llama_client.chat("hello")


@pytest.mark.parametrize(
    "error,expected",
    [
        (URLError(TimeoutError("socket timed out")), LlamaTimeoutError),
        (URLError("connection refused"), LlamaTransportError),
    ],
)
def test_chat_classifies_wrapped_transport_timeouts(
    monkeypatch, error, expected
) -> None:
    monkeypatch.setattr(
        llama_client, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(error)
    )
    with pytest.raises(expected):
        llama_client.chat("hello")


def test_interrupted_http_body_read_is_transport_error(monkeypatch) -> None:
    class InterruptedResponse(FakeHTTPResponse):
        def read(self, size: int = -1) -> bytes:
            raise IncompleteRead(b"partial")

    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_a, **_k: InterruptedResponse(b""),
    )
    with pytest.raises(LlamaTransportError):
        llama_client.chat("hello")


def test_deeply_nested_http_json_is_a_client_error(monkeypatch) -> None:
    body = (
        b'{"choices":[{"message":{"content":"ok"}}],"usage":'
        + b"[" * 2000
        + b"]" * 2000
        + b"}"
    )
    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_a, **_k: FakeHTTPResponse(body),
    )
    with pytest.raises(LlamaClientError, match="malformed chat response"):
        llama_client.chat("hello")


def test_overlong_http_json_integer_is_a_client_error(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_client,
        "urlopen",
        lambda *_a, **_k: FakeHTTPResponse(b"{}"),
    )
    monkeypatch.setattr(
        llama_client.json,
        "loads",
        lambda _body: (_ for _ in ()).throw(ValueError("bad integer")),
    )
    with pytest.raises(LlamaClientError, match="malformed chat response"):
        llama_client.chat("hello")
