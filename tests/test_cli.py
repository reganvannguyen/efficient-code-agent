import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness import cli
from harness.agent.runtime import (
    DEFAULT_MAX_PROMPT_BYTES,
    READ_ONLY_ACTION_SCHEMA,
    READ_ONLY_SYSTEM_PROMPT,
)
from harness.llama_client import (
    ChatResponse,
    LlamaClientError,
    LlamaTimeoutError,
    LlamaTransportError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "read_only_repo"


def copy_fixture(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    return repo


def action(kind, **fields):
    return ChatResponse(json.dumps({"kind": kind, **fields}))


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


def test_run_command_defaults_forward_model_configuration(
    monkeypatch, tmp_path, capsys
):
    repo = copy_fixture(tmp_path)
    requests = []

    def fake_chat(prompt, **kwargs):
        requests.append((prompt, kwargs))
        return action(
            "complete", summary="A greeting is formed from the prefix and name."
        )

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(["run", "Explain how greeting is formed", "--repo", str(repo)])

    captured = capsys.readouterr()
    assert captured.out == "A greeting is formed from the prefix and name.\n"
    assert (
        "status=completed stop_reason=model_complete model_calls=1 steps=1"
        in captured.err
    )
    assert "unverified model completion" in captured.err
    system_prompt = requests[0][1]["system_prompt"]
    assert system_prompt == READ_ONLY_SYSTEM_PROMPT
    assert (
        len(requests[0][0].encode("utf-8")) + len(system_prompt.encode("utf-8"))
        <= DEFAULT_MAX_PROMPT_BYTES
    )
    assert requests[0][1] == {
        "base_url": "http://127.0.0.1:9931",
        "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "max_tokens": 256,
        "timeout": pytest.approx(120),
        "response_schema": READ_ONLY_ACTION_SCHEMA,
        "system_prompt": READ_ONLY_SYSTEM_PROMPT,
    }


def test_run_command_forwards_explicit_options(monkeypatch, tmp_path, capsys):
    options = []
    run_options = []
    from harness.agent.runtime import run_agent as real_run_agent

    def spy_run_agent(task, repo, **kwargs):
        run_options.append(kwargs.copy())
        return real_run_agent(task, repo, **kwargs)

    def fake_chat(_prompt, **kwargs):
        options.append(kwargs)
        return action("complete", summary="done")

    monkeypatch.setattr(cli, "chat", fake_chat)
    monkeypatch.setattr(cli, "run_agent", spy_run_agent)
    cli.main(
        [
            "run",
            "task",
            "--repo",
            str(copy_fixture(tmp_path)),
            "--base-url",
            "http://localhost:9000",
            "--model",
            "custom.gguf",
            "--max-tokens",
            "77",
            "--timeout",
            "4.5",
            "--run-timeout",
            "12",
            "--max-model-calls",
            "3",
            "--max-steps",
            "5",
        ]
    )
    capsys.readouterr()
    assert options == [
        {
            "base_url": "http://localhost:9000",
            "model": "custom.gguf",
            "max_tokens": 77,
            "timeout": pytest.approx(4.5),
            "response_schema": READ_ONLY_ACTION_SCHEMA,
            "system_prompt": READ_ONLY_SYSTEM_PROMPT,
        }
    ]
    assert run_options[0]["request_timeout"] == pytest.approx(4.5)
    assert run_options[0]["run_timeout"] == pytest.approx(12)
    assert run_options[0]["max_model_calls"] == 3
    assert run_options[0]["max_steps"] == 5
    assert run_options[0]["max_prompt_bytes"] == DEFAULT_MAX_PROMPT_BYTES - len(
        READ_ONLY_SYSTEM_PROMPT.encode("utf-8")
    )


def test_run_repository_defaults_to_current_directory(monkeypatch, tmp_path, capsys):
    repo = copy_fixture(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        cli,
        "chat",
        lambda *_a, **_k: action("complete", summary="answer"),
    )

    cli.main(["run", "task"])

    assert capsys.readouterr().out == "answer\n"


def test_run_real_runtime_reads_fixture_files_and_preserves_them(
    monkeypatch, tmp_path, capsys
):
    repo = copy_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()}
    calls = []
    from harness.tools.dispatcher import execute_tool as real_execute_tool

    def spy_execute_tool(tool_action, workspace):
        calls.append((tool_action.name, tool_action.arguments.copy()))
        return real_execute_tool(tool_action, workspace)

    monkeypatch.setattr("harness.tools.dispatcher.execute_tool", spy_execute_tool)
    prompts = []
    responses = iter(
        [
            action("tool", name="list_files", arguments={}),
            action("tool", name="read_file", arguments={"path": "settings.py"}),
            action("tool", name="read_file", arguments={"path": "greeting.py"}),
            action(
                "complete",
                summary="greeting.py imports GREETING_PREFIX from settings.py and combines it with the name.",
            ),
        ]
    )

    def fake_chat(prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(
        [
            "run",
            "Explain how greeting.py combines the prefix and name",
            "--repo",
            str(repo),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("greeting.py imports GREETING_PREFIX")
    assert "status=completed" in captured.err
    assert calls == [
        ("list_files", {}),
        ("read_file", {"path": "settings.py"}),
        ("read_file", {"path": "greeting.py"}),
    ]
    assert "GREETING_PREFIX" in prompts[2] and "Hello" in prompts[2]
    assert "from settings import GREETING_PREFIX" in prompts[3]
    assert {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()} == before


def test_coding_only_options_are_rejected_without_coding(monkeypatch, tmp_path):
    from harness import cli

    monkeypatch.setattr(
        cli,
        "chat",
        lambda *_args, **_kwargs: pytest.fail("inference must not start"),
    )
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["run", "task", "--repo", str(tmp_path), "--test-timeout", "60"])
    assert exit_error.value.code == 2


def test_coding_auto_approval_uses_coding_schema(monkeypatch, tmp_path, capsys):
    from harness import cli

    seen = {}

    def fake_chat(prompt, **kwargs):
        seen.update(kwargs)
        return ChatResponse('{"kind":"complete","summary":"done"}')

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(["run", "task", "--repo", str(tmp_path), "--coding", "--auto-approve"])
    assert seen["response_schema"] is not None
    names = [
        variant["properties"]["name"].get("const")
        for variant in seen["response_schema"]["oneOf"]
        if "name" in variant["properties"]
    ]
    assert "edit_file" in names
    assert "run_tests" in names
    assert "done" in capsys.readouterr().out


def test_cli_coding_read_approve_edit_test_complete(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.txt").write_text("old\n")
    (repo / "test_sample.py").write_text("def test_pass():\n    assert True\n")
    import re

    requests = []
    monkeypatch.setattr(
        cli, "_approve_in_terminal", lambda request: requests.append(request) or True
    )
    calls = 0

    def fake_chat(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return action("tool", name="read_file", arguments={"path": "source.txt"})
        if calls == 2:
            match = re.search(r'"sha256":"([0-9a-f]{64})"', prompt)
            assert match is not None
            return action(
                "tool",
                name="edit_file",
                arguments={
                    "path": "source.txt",
                    "expected_sha256": match.group(1),
                    "old_text": "old",
                    "new_text": "new",
                },
            )
        if calls == 3:
            return action(
                "tool",
                name="run_tests",
                arguments={"preset": "pytest", "target": "test_sample.py"},
            )
        if calls == 4:
            return action("tool", name="read_file", arguments={"path": "source.txt"})
        return action("complete", summary="updated source")

    monkeypatch.setattr(cli, "chat", fake_chat)

    cli.main(
        [
            "run",
            "edit and test",
            "--repo",
            str(repo),
            "--coding",
            "--test-target",
            "test_sample.py",
        ]
    )

    output = capsys.readouterr()
    assert output.out == "updated source\n"
    assert (repo / "source.txt").read_text() == "new\n"
    assert [request.tool for request in requests] == ["edit_file", "run_tests"]
    assert "changed_paths=source.txt" in output.err
    assert "last_test_exit_code=0 outcome=passed" in output.err


def test_approval_prompt_uses_stderr_and_denies_without_tty(monkeypatch, capsys):
    import io

    request = cli.ApprovalRequest(
        "run_tests",
        Path("/workspace"),
        preset="pytest",
        target="tests/test_one.py",
        executable="/python",
        argv=("/python", "-m", "pytest", "-q", "--", "tests/test_one.py"),
    )
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("yes\n"))
    assert not cli._approve_in_terminal(request)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_approval_prompt_eof_denies_and_preview_stays_on_stderr(monkeypatch, capsys):
    import io

    class TTYInput(io.StringIO):
        def isatty(self):
            return True

    request = cli.ApprovalRequest(
        "edit_file",
        Path("/workspace"),
        path="src/one.py",
        expected_sha256="a" * 64,
        old_text="before",
        new_text="after",
        diff="-before\n+after\n",
    )
    monkeypatch.setattr(cli.sys, "stdin", TTYInput(""))
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True)
    assert not cli._approve_in_terminal(request)
    output = capsys.readouterr()
    assert output.out == ""
    assert 'old_text="before"' in output.err
    assert 'new_text="after"' in output.err


def test_run_searches_then_reads_matching_file(monkeypatch, tmp_path, capsys):
    repo = copy_fixture(tmp_path)
    prompts = []
    responses = iter(
        [
            action("tool", name="search_code", arguments={"query": "GREETING_PREFIX"}),
            action("tool", name="read_file", arguments={"path": "settings.py"}),
            action("complete", summary="The prefix is Hello."),
        ]
    )

    def fake_chat(prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(["run", "Find the greeting prefix", "--repo", str(repo)])

    captured = capsys.readouterr()
    assert captured.out == "The prefix is Hello.\n"
    assert "status=completed" in captured.err
    assert "settings.py" in prompts[1]
    assert "GREETING_PREFIX" in prompts[2] and "Hello" in prompts[2]


@pytest.mark.parametrize(
    "args",
    [
        ["run", "   "],
        ["run", "task", "--max-tokens", "0"],
        ["run", "task", "--max-model-calls", "-1"],
        ["run", "task", "--max-steps", "0"],
        ["run", "task", "--timeout", "nan"],
        ["run", "task", "--run-timeout", "0"],
        ["run", "task", "--repo", "/missing/repo/path"],
    ],
)
def test_run_invalid_arguments_exit_two_before_model(monkeypatch, args):
    monkeypatch.setattr(cli, "chat", lambda *_a, **_k: pytest.fail("model called"))
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_run_transport_retry_is_bounded_and_reported(monkeypatch, tmp_path, capsys):
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append(1)
        raise LlamaTransportError("offline")

    monkeypatch.setattr(cli, "chat", unavailable)
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "task", "--repo", str(copy_fixture(tmp_path))])
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert len(calls) == 2
    assert captured.out == ""
    assert "model server unavailable" in captured.err
    assert "stop_reason=model_error model_calls=2 steps=0" in captured.err


@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (LlamaTimeoutError("late"), "model_timeout"),
        (LlamaClientError("malformed"), "model_error"),
    ],
)
def test_run_model_failures_are_reported_without_traceback(
    monkeypatch, tmp_path, capsys, exception, reason
):
    monkeypatch.setattr(cli, "chat", lambda *_a, **_k: (_ for _ in ()).throw(exception))
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "task", "--repo", str(copy_fixture(tmp_path))])
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert captured.out == ""
    assert f"stop_reason={reason}" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("responses", "expected_status", "expected_reason"),
    [
        ([ChatResponse("not json")] * 3, "limit_reached", "invalid_action_limit"),
        (
            [action("tool", name="list_files", arguments={})],
            "limit_reached",
            "model_call_limit",
        ),
        (
            [action("tool", name="list_files", arguments={})],
            "limit_reached",
            "step_limit",
        ),
    ],
)
def test_run_limits_report_status_and_counts(
    monkeypatch, tmp_path, capsys, responses, expected_status, expected_reason
):
    calls = iter(responses)
    monkeypatch.setattr(cli, "chat", lambda *_a, **_k: next(calls))
    args = ["run", "task", "--repo", str(copy_fixture(tmp_path))]
    if expected_reason == "model_call_limit":
        args += ["--max-model-calls", "1"]
    elif expected_reason == "step_limit":
        args += ["--max-steps", "1"]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert f"status={expected_status} stop_reason={expected_reason}" in captured.err
    assert captured.out == ""


def test_run_keyboard_interrupt_returns_cancelled_exit_code(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        cli, "chat", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "task", "--repo", str(copy_fixture(tmp_path))])
    captured = capsys.readouterr()
    assert error.value.code == 130
    assert captured.out == ""
    assert "status=cancelled stop_reason=cancelled" in captured.err


def test_run_deadline_reports_limit_after_slow_model(monkeypatch, tmp_path, capsys):
    import time

    def slow_model(*_args, **_kwargs):
        time.sleep(0.02)
        return action("complete", summary="late")

    monkeypatch.setattr(cli, "chat", slow_model)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "run",
                "task",
                "--repo",
                str(copy_fixture(tmp_path)),
                "--run-timeout",
                "0.005",
            ]
        )
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert captured.out == ""
    assert "status=limit_reached stop_reason=run_timeout" in captured.err


def test_module_run_help_lists_runtime_options():
    result = subprocess.run(
        [sys.executable, "-m", "harness", "run", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--repo" in result.stdout
    assert "--run-timeout" in result.stdout
    assert "--max-model-calls" in result.stdout
    assert "--max-steps" in result.stdout


@pytest.mark.skipif(
    os.getenv("HARNESS_LIVE_SMOKE") != "1", reason="opt-in local model smoke"
)
def test_live_run_smoke_reads_both_fixture_files(monkeypatch, tmp_path, capsys):
    repo = copy_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()}
    read_paths = set()
    from harness.tools.dispatcher import execute_tool as real_execute_tool

    def spy_execute_tool(tool_action, workspace):
        result = real_execute_tool(tool_action, workspace)
        if tool_action.name == "read_file" and result.ok:
            read_paths.add(result.data["path"])
        return result

    monkeypatch.setattr("harness.tools.dispatcher.execute_tool", spy_execute_tool)
    cli.main(
        [
            "run",
            "Read settings.py and greeting.py, then explain how the prefix and name are combined.",
            "--repo",
            str(repo),
        ]
    )
    captured = capsys.readouterr()
    assert "status=completed" in captured.err
    assert captured.out.strip()
    assert {"settings.py", "greeting.py"} <= read_paths
    assert {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()} == before


def test_cli_reports_verified_completion_and_diff(monkeypatch, tmp_path, capsys):
    calc = tmp_path / "calc.py"
    calc.write_text("x = 1\n")
    test_file = tmp_path / "test_calc.py"
    test_file.write_text("from calc import x\ndef test_calc(): assert x == 2\n")
    digest = hashlib.sha256(b"x = 1\n").hexdigest()

    actions = iter(
        [
            action(
                "tool",
                name="edit_file",
                arguments={
                    "path": "calc.py",
                    "expected_sha256": digest,
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                },
            ),
            action(
                "tool",
                name="run_tests",
                arguments={"preset": "pytest", "target": "test_calc.py"},
            ),
            action("complete", summary="Bug fixed and verified"),
        ]
    )

    monkeypatch.setattr(cli, "chat", lambda _p, **_kw: next(actions))
    cli.main(
        [
            "run",
            "Fix test",
            "--repo",
            str(tmp_path),
            "--coding",
            "--auto-approve",
            "--test-target",
            "test_calc.py",
        ]
    )
    captured = capsys.readouterr()
    assert "status=completed" in captured.err
    assert "verification=verified" in captured.err
    assert "(verified completion)" in captured.err
    assert "candidate_diff:" in captured.err
    assert "tests_executed=1" in captured.err
    assert captured.out.strip() == "Bug fixed and verified"


def test_session_command_reuses_working_memory_across_turns(
    monkeypatch, tmp_path, capsys
):
    from harness import cli

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
    memory_dir = tmp_path / ".harness" / "runs"

    turns = iter(["Find VALUE in app.py", "What was VALUE?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(turns))

    prompts: list[str] = []
    responses = iter(
        [
            ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "update_context",
                        "arguments": {
                            "operation": "replace",
                            "target": "working.md",
                            "content": "# Working Context\n- VALUE is 42 in app.py:1\n",
                        },
                    }
                )
            ),
            ChatResponse(
                json.dumps({"kind": "complete", "summary": "Saved VALUE=42."})
            ),
            ChatResponse(json.dumps({"kind": "complete", "summary": "VALUE is 42."})),
        ]
    )

    def fake_chat(prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(cli, "chat", fake_chat)
    cli.main(
        [
            "session",
            "--repo",
            str(repo),
            "--session",
            "chat-demo",
            "--memory-dir",
            str(memory_dir),
            "--no-db",
        ]
    )
    captured = capsys.readouterr()
    assert "Saved VALUE=42." in captured.out
    assert "VALUE is 42." in captured.out
    assert "VALUE is 42 in app.py:1" in prompts[-1]
    assert (memory_dir / "chat-demo" / "working.md").exists()
