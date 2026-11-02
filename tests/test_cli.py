import subprocess
import sys

import pytest

from harness import cli
from harness.llama_client import ChatResponse, LlamaClientError


def test_module_entry_point_accepts_exit_without_loading_a_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "harness"],
        input="/exit\n",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "prompts are independent" in result.stdout
    assert result.stderr == ""


def test_interactive_prompts_are_sent_as_independent_calls(monkeypatch, capsys) -> None:
    prompts = iter(["first question", "second question", "/exit"])
    calls = []

    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    monkeypatch.setattr(
        cli,
        "chat",
        lambda prompt: calls.append(prompt) or ChatResponse(f"Answer: {prompt}"),
    )

    cli.main([])

    captured = capsys.readouterr()
    assert calls == ["first question", "second question"]
    assert "Answer: first question\nAnswer: second question\n" in captured.out


def test_interactive_ignores_blank_prompts_and_stops_on_eof(
    monkeypatch, capsys
) -> None:
    prompts = iter(["  \t", "\n"])
    calls = []

    def get_input(_prompt):
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", get_input)
    monkeypatch.setattr(cli, "chat", lambda prompt: calls.append(prompt))

    cli.interactive()

    assert calls == []
    assert "Interactive mode" in capsys.readouterr().out


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_interactive_input_termination_is_clean(monkeypatch, capsys, exception) -> None:
    def stop(_prompt):
        raise exception

    monkeypatch.setattr("builtins.input", stop)
    cli.interactive()

    captured = capsys.readouterr()
    assert "Interactive mode" in captured.out
    assert "Traceback" not in captured.err


def test_interactive_keyboard_interrupt_during_request_is_clean(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "hello")
    monkeypatch.setattr(
        cli, "chat", lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    cli.interactive()

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_interactive_model_error_allows_retry(monkeypatch, capsys) -> None:
    prompts = iter(["retry me", "try again", "/exit"])
    calls = []

    def get_input(_prompt):
        return next(prompts)

    def fake_chat(prompt):
        calls.append(prompt)
        if prompt == "retry me":
            raise LlamaClientError("server unavailable")
        return ChatResponse("Recovered", 3, 1, 4)

    monkeypatch.setattr("builtins.input", get_input)
    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.interactive()

    captured = capsys.readouterr()
    assert calls == ["retry me", "try again"]
    assert captured.out.endswith("Recovered\n")
    assert captured.err == (
        "harness: server unavailable\ntoken usage: prompt=3 completion=1 total=4\n"
    )


def test_chat_command_forwards_options_and_prints_response(monkeypatch, capsys) -> None:
    calls = []

    def fake_chat(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return ChatResponse("Hello", 10, 2, 12)

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(
        [
            "chat",
            "say hello",
            "--base-url",
            "http://localhost:9000",
            "--model",
            "test.gguf",
            "--timeout",
            "3.5",
            "--max-tokens",
            "32",
        ]
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            "say hello",
            {
                "base_url": "http://localhost:9000",
                "model": "test.gguf",
                "timeout": 3.5,
                "max_tokens": 32,
            },
        )
    ]
    assert captured.out == "Hello\n"
    assert captured.err == "token usage: prompt=10 completion=2 total=12\n"


def test_chat_command_omits_unknown_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "chat", lambda *_args, **_kwargs: ChatResponse("Hi"))

    cli.main(["chat", "hello"])

    captured = capsys.readouterr()
    assert captured.out == "Hi\n"
    assert captured.err == ""


def test_chat_command_reports_client_error_without_traceback(
    monkeypatch, capsys
) -> None:
    def fail(*_args, **_kwargs):
        raise LlamaClientError("server unavailable")

    monkeypatch.setattr(cli, "chat", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["chat", "hello"])

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert captured.out == ""
    assert captured.err == "harness: server unavailable\n"


@pytest.mark.parametrize(
    "args",
    [
        ["chat", "hello", "--timeout", "nope"],
        ["chat", "   "],
        ["chat", "hello", "--timeout", "0"],
        ["chat", "hello", "--timeout", "nan"],
        ["chat", "hello", "--max-tokens", "0"],
    ],
)
def test_chat_command_invalid_arguments_exit_two(args) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(args)

    assert error.value.code == 2
