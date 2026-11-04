import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.agent import RunStatus, ToolAction, run_agent, runtime
from harness.agent.runtime import _NEXT_ACTION, READ_ONLY_ACTION_SCHEMA, _build_prompt
from harness.llama_client import (
    ChatResponse,
    LlamaClientError,
    LlamaResponseTooLargeError,
    LlamaTimeoutError,
    LlamaTransportError,
)
from harness.tools import ToolResult, execute_tool


def test_read_only_action_schema_has_exact_tool_and_completion_variants() -> None:
    list_files, read_file, completion = READ_ONLY_ACTION_SCHEMA["oneOf"]

    assert READ_ONLY_ACTION_SCHEMA["type"] == "object"
    for tool in (list_files, read_file):
        assert tool["required"] == ["kind", "name", "arguments"]
        assert tool["properties"]["kind"]["const"] == "tool"
        assert set(tool["properties"]) == {"kind", "name", "arguments"}
        assert tool["additionalProperties"] is False
    assert list_files["properties"]["name"]["const"] == "list_files"
    list_arguments = list_files["properties"]["arguments"]
    assert list_arguments["type"] == "object"
    assert set(list_arguments["properties"]) == {"directory", "pattern"}
    assert "required" not in list_arguments
    assert list_arguments["properties"]["directory"] == {
        "type": "string",
        "minLength": 1,
    }
    assert list_arguments["properties"]["pattern"] == {
        "type": "string",
        "minLength": 1,
    }
    assert list_arguments["additionalProperties"] is False
    assert read_file["properties"]["name"]["const"] == "read_file"
    read_arguments = read_file["properties"]["arguments"]
    assert read_arguments["type"] == "object"
    assert set(read_arguments["properties"]) == {
        "path",
        "start_line",
        "end_line",
    }
    assert read_arguments["required"] == ["path"]
    assert read_arguments["properties"]["path"] == {
        "type": "string",
        "minLength": 1,
    }
    for line_field in ("start_line", "end_line"):
        assert read_arguments["properties"][line_field] == {
            "type": "integer",
            "minimum": 1,
        }
        assert "maximum" not in read_arguments["properties"][line_field]
    assert read_arguments["additionalProperties"] is False
    assert completion["required"] == ["kind", "summary"]
    assert set(completion["properties"]) == {"kind", "summary"}
    assert completion["properties"]["kind"]["const"] == "complete"
    assert completion["properties"]["summary"]["type"] == "string"
    assert completion["properties"]["summary"]["minLength"] == 1
    assert completion["additionalProperties"] is False


def test_run_reads_multiple_files_and_completes(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("first file", encoding="utf-8")
    (tmp_path / "two.txt").write_text("second file", encoding="utf-8")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    outputs = iter(
        [
            '{"kind":"tool","name":"list_files","arguments":{}}',
            '{"kind":"tool","name":"read_file","arguments":{"path":"one.txt"}}',
            '{"kind":"tool","name":"read_file","arguments":{"path":"two.txt"}}',
            '{"kind":"complete","summary":"Both files say what they contain."}',
        ]
    )
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Inspect the files", tmp_path, model=model)

    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == "model_complete"
    assert run.final_response == "Both files say what they contain."
    assert (run.model_calls, run.steps) == (4, 4)
    assert '"tool":"list_files","ok":true' in prompts[1]
    assert '"paths":["one.txt","two.txt"]' in prompts[1]
    assert "first file" in prompts[2]
    assert '"path":"two.txt"' in prompts[3]
    assert "second file" in prompts[3]
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_list_files_recovers_from_regex_like_pattern_with_separate_globs(
    tmp_path: Path,
) -> None:
    (tmp_path / "settings.py").write_text('PREFIX = "Hello"\n', encoding="utf-8")
    (tmp_path / "greeting.py").write_text(
        "from settings import PREFIX\n", encoding="utf-8"
    )
    outputs = iter(
        [
            '{"kind":"tool","name":"list_files","arguments":{"pattern":"settings.py|greeting.py"}}',
            '{"kind":"tool","name":"list_files","arguments":{"pattern":"settings.py"}}',
            '{"kind":"tool","name":"list_files","arguments":{"pattern":"greeting.py"}}',
            '{"kind":"complete","summary":"Found settings.py and greeting.py."}',
        ]
    )
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("List the greeting and settings file paths", tmp_path, model=model)

    assert run.status is RunStatus.COMPLETED
    assert run.model_calls == run.steps == 4
    assert run.final_response == "Found settings.py and greeting.py."
    assert "Use list_files for path-only questions" in prompts[0]
    assert "Python `Path.match` glob syntax, not regular expressions" in prompts[0]
    assert "An empty filtered result does not prove a path is absent" in prompts[0]
    assert '"paths":[]' in prompts[1]
    assert '"pattern":"settings.py"' in prompts[2]
    assert '"pattern":"greeting.py"' in prompts[3]
    assert '"paths":["settings.py"]' in prompts[3]
    assert '"paths":["greeting.py"]' in prompts[3]


def test_invalid_action_feedback_guides_a_real_file_read(tmp_path: Path) -> None:
    (tmp_path / "settings.py").write_text('ANSWER = "verified"\n', encoding="utf-8")
    outputs = iter(
        [
            '```json\n{"kind":"read_file","arguments":{"path":"settings.py"}}\n```',
            '{"kind":"read_file","arguments":{"path":"settings.py"}}',
            '{"kind":"tool","name":"read_file","arguments":{"path":"settings.py"}}',
            '{"kind":"complete","summary":"settings.py defines ANSWER as verified."}',
        ]
    )
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Explain ANSWER in settings.py", tmp_path, model=model)

    assert run.status is RunStatus.COMPLETED
    assert run.final_response == "settings.py defines ANSWER as verified."
    assert run.model_calls == run.steps == 4
    assert "Only `kind` values `tool` and `complete` are supported." in prompts[0]
    assert (
        '{"kind":"tool","name":"read_file","arguments":{"path":"relative/path"}}'
        in prompts[0]
    )
    assert "no prose or Markdown fences" in prompts[0]
    history_label = "Action and observation history (JSON exchanges, oldest first):\n"

    def last_exchange(prompt: str) -> dict[str, object]:
        history = prompt.split(history_label, 1)[1].split("\n\nNext action:", 1)[0]
        return json.loads(history.splitlines()[-1])

    first_feedback = last_exchange(prompts[1])
    second_feedback = last_exchange(prompts[2])
    for exchange in (first_feedback, second_feedback):
        error = exchange["observation"]["error"]
        assert 'use {"kind":"tool","name":"read_file",' in error
        assert '"arguments":{"path":"relative/path"}}' in error
        assert "Put the tool name in 'name', not 'kind'" in error
    assert "```json" in first_feedback["observation"]["excerpt"]
    assert '"kind":"read_file"' in second_feedback["observation"]["excerpt"]
    assert "ANSWER" in prompts[3] and "verified" in prompts[3]


def test_independent_runs_do_not_share_history_counters_or_answers(
    tmp_path: Path,
) -> None:
    first_prompts: list[str] = []
    second_prompts: list[str] = []

    def first_model(prompt: str, *, timeout: float) -> ChatResponse:
        first_prompts.append(prompt)
        if len(first_prompts) == 1:
            return ChatResponse('{"kind":"tool","name":"list_files","arguments":{}}')
        return ChatResponse('{"kind":"complete","summary":"first answer"}')

    def second_model(prompt: str, *, timeout: float) -> ChatResponse:
        second_prompts.append(prompt)
        return ChatResponse('{"kind":"complete","summary":"second answer"}')

    first = run_agent("first private task", tmp_path, model=first_model)
    second = run_agent("second private task", tmp_path, model=second_model)

    assert first.final_response == "first answer"
    assert second.final_response == "second answer"
    assert (first.model_calls, first.steps) == (2, 2)
    assert (second.model_calls, second.steps) == (1, 1)
    assert "first private task" in first_prompts[0]
    assert "second private task" not in first_prompts[0]
    assert "second private task" in second_prompts[0]
    assert "first private task" not in second_prompts[0]
    assert "Action and observation history" not in second_prompts[0]


@pytest.mark.parametrize(
    "case",
    [
        ('{"kind":"tool","name":"missing","arguments":{}}', "unknown_tool"),
        (
            '{"kind":"tool","name":"read_file","arguments":{"path":"../secret"}}',
            "path_denied",
        ),
        (
            '{"kind":"tool","name":"read_file","arguments":{"path":"file.txt","start_line":true}}',
            "invalid_arguments",
        ),
        ("not json", "Invalid response"),
    ],
)
def test_invalid_tool_or_response_is_observed_and_can_recover(
    tmp_path: Path, case: tuple[str, str]
) -> None:
    answer, expected_feedback = case
    outputs = iter([answer, '{"kind":"complete","summary":"Done"}'])
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Inspect safely", tmp_path, model=model)

    assert run.status is RunStatus.COMPLETED
    assert run.steps == run.model_calls == 2
    assert expected_feedback in prompts[1]


@pytest.mark.parametrize("max_calls", [1, 2])
def test_completion_on_last_allowed_call_succeeds(
    tmp_path: Path, max_calls: int
) -> None:
    calls = 0

    def model(_: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        if calls < max_calls:
            return ChatResponse('{"kind":"tool","name":"list_files","arguments":{}}')
        return ChatResponse('{"kind":"complete","summary":"Done"}')

    run = run_agent("Finish", tmp_path, model=model, max_model_calls=max_calls)

    assert run.status is RunStatus.COMPLETED
    assert run.model_calls == max_calls


@pytest.mark.parametrize(
    "answer",
    [
        '{"kind":"tool","name":"list_files","arguments":{}}',
        "invalid",
    ],
)
def test_noncompleting_model_stops_at_call_limit(tmp_path: Path, answer: str) -> None:
    run = run_agent(
        "Keep going",
        tmp_path,
        model=lambda _, *, timeout: ChatResponse(answer),
        max_model_calls=3,
    )

    assert run.status is RunStatus.LIMIT_REACHED
    if answer == "invalid":
        assert run.stop_reason == "invalid_action_limit"
        assert run.model_calls == run.steps == 3
    else:
        assert run.stop_reason == "model_call_limit"
        assert run.model_calls == run.steps == 3


def test_model_failure_counts_attempt_and_fails_without_step(tmp_path: Path) -> None:
    def model(_: str, *, timeout: float) -> ChatResponse:
        raise LlamaClientError("offline")

    run = run_agent("Inspect", tmp_path, model=model)

    assert run.status is RunStatus.FAILED
    assert run.stop_reason == "model_error"
    assert run.model_calls == 1
    assert run.steps == 0
    assert run.final_response is None


def test_prompt_budget_keeps_task_and_newest_whole_exchange() -> None:
    history = [
        '{"action":{"i":0},"observation":{"text":"oldest"}}',
        '{"action":{"i":1},"observation":{"text":"middle"}}',
        '{"action":{"i":2},"observation":{"text":"newest"}}',
    ]
    task = "Read this task"
    full_prompt = _build_prompt(task, history.copy(), 100_000)
    newest_two_prompt = _build_prompt(task, history[1:].copy(), 100_000)
    assert full_prompt is not None and newest_two_prompt is not None
    byte_limit = len(newest_two_prompt.encode("utf-8"))
    prompt = _build_prompt(task, history, byte_limit)

    assert prompt is not None
    assert "You are a read-only repository assistant." in prompt
    assert "Original task (immutable):\nRead this task" in prompt
    assert '"i":0' not in prompt
    assert prompt.index('"i":1') < prompt.index('"i":2')
    assert prompt.endswith(_NEXT_ACTION)
    assert len(prompt.encode("utf-8")) <= byte_limit
    assert history == [
        '{"action":{"i":1},"observation":{"text":"middle"}}',
        '{"action":{"i":2},"observation":{"text":"newest"}}',
    ]


def test_oversized_prompt_stops_without_inference(tmp_path: Path) -> None:
    baseline = _build_prompt("Answer", [], 100_000)
    assert baseline is not None
    byte_limit = len(baseline.encode("utf-8")) + 100
    calls = 0

    def model(_: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse(
            '{"kind":"tool","name":"read_file","arguments":{"path":"one"}}'
        )

    run = run_agent("Answer", tmp_path, model=model, max_prompt_bytes=byte_limit)

    assert calls == 1
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "prompt_budget"
    assert run.model_calls == run.steps == 1


def test_oversized_task_stops_before_inference(tmp_path: Path) -> None:
    calls = 0

    def model(_: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse('{"kind":"complete","summary":"Done"}')

    run = run_agent("Answer", tmp_path, model=model, max_prompt_bytes=1)

    assert calls == 0
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "prompt_budget"
    assert run.model_calls == run.steps == 0


def test_large_read_observation_stops_before_overbudget_next_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text("x" * 4000, encoding="utf-8")
    task = "Read the large file"
    base_prompt = _build_prompt(task, [], 100_000)
    assert base_prompt is not None
    byte_limit = len(base_prompt.encode("utf-8")) + 500
    observation = execute_tool(
        ToolAction("read_file", {"path": "large.txt"}), tmp_path
    ).to_dict()
    assert len(json.dumps(observation).encode("utf-8")) > 500
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(
            '{"kind":"tool","name":"read_file","arguments":{"path":"large.txt"}}'
        )

    run = run_agent(task, tmp_path, model=model, max_prompt_bytes=byte_limit)

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "prompt_budget"
    assert run.model_calls == run.steps == len(prompts) == 1
    assert all(len(prompt.encode("utf-8")) <= byte_limit for prompt in prompts)


def test_prompt_budget_counts_utf8_bytes() -> None:
    task = "é" * 100
    full_prompt = _build_prompt(task, [], 10_000)
    assert full_prompt is not None
    byte_count = len(full_prompt.encode("utf-8"))

    assert full_prompt.endswith(_NEXT_ACTION)
    assert _build_prompt(task, [], byte_count) == full_prompt
    assert _build_prompt(task, [], byte_count - 1) is None


@pytest.mark.parametrize(
    "answer",
    [
        "invalid \ud800 response",
        '{"kind":"tool","name":"read_file","arguments":{"path":"\\ud800"}}',
    ],
)
def test_lone_surrogates_are_observed_without_crashing(
    tmp_path: Path, answer: str
) -> None:
    outputs = iter([answer, '{"kind":"complete","summary":"Recovered"}'])
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Recover", tmp_path, model=model)

    assert run.status is RunStatus.COMPLETED
    assert run.model_calls == run.steps == 2
    assert any(
        marker in prompts[1]
        for marker in ("invalid Unicode", "Invalid response", "path_denied")
    )


def test_invalid_response_excerpt_is_utf8_safe_and_at_most_1kib(
    tmp_path: Path,
) -> None:
    outputs = iter(["é" * 2000, '{"kind":"complete","summary":"Recovered"}'])
    prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Recover from invalid output", tmp_path, model=model)
    history_label = "Action and observation history (JSON exchanges, oldest first):\n"
    history = prompts[1].split(history_label, 1)[1].split("\n\nNext action:", 1)[0]
    exchange = history.splitlines()[0]
    excerpt = json.loads(exchange)["observation"]["excerpt"]

    assert run.status is RunStatus.COMPLETED
    assert len(excerpt.encode("utf-8")) <= 1024
    assert excerpt == "é" * 512


def test_step_limit_is_independent_and_accepts_completion_on_last_step(
    tmp_path: Path,
) -> None:
    outputs = iter(
        [
            '{"kind":"tool","name":"list_files","arguments":{}}',
            '{"kind":"complete","summary":"Done"}',
        ]
    )
    run = run_agent(
        "Inspect",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(next(outputs)),
        max_steps=2,
    )
    assert (run.status, run.stop_reason, run.steps, run.model_calls) == (
        RunStatus.COMPLETED,
        "model_complete",
        2,
        2,
    )

    run = run_agent(
        "Inspect",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(
            '{"kind":"tool","name":"list_files","arguments":{}}'
        ),
        max_steps=1,
    )
    assert run.stop_reason == "step_limit"
    assert run.steps == run.model_calls == 1


def test_invalid_action_streak_stops_and_valid_action_resets(tmp_path: Path) -> None:
    outputs = iter(
        [
            "bad",
            '{"kind":"tool","name":"list_files","arguments":{}}',
            "bad",
            "bad",
            "bad",
        ]
    )
    run = run_agent(
        "Inspect",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(next(outputs)),
        max_consecutive_invalid_actions=3,
    )
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "invalid_action_limit"
    assert run.steps == run.model_calls == 5


@pytest.mark.parametrize(
    "answer",
    [
        '{"kind":"tool","name":"unknown","arguments":{}}',
        '{"kind":"tool","name":"read_file","arguments":{"path":"x","extra":1}}',
        '{"kind":"tool","name":"read_file","arguments":{"path":"../x"}}',
    ],
)
def test_invalid_tool_error_codes_advance_invalid_streak(
    tmp_path: Path, answer: str
) -> None:
    run = run_agent(
        "Inspect", tmp_path, model=lambda _p, *, timeout: ChatResponse(answer)
    )
    assert run.stop_reason == "invalid_action_limit"
    assert run.steps == run.model_calls == 3


def test_ordinary_tool_error_is_observed_and_resets_invalid_streak(
    tmp_path: Path, monkeypatch
) -> None:
    outputs = iter(
        [
            "bad",
            '{"kind":"tool","name":"read_file","arguments":{"path":"x"}}',
            "bad",
            "bad",
            "bad",
        ]
    )
    monkeypatch.setattr(
        "harness.tools.dispatcher.execute_tool",
        lambda action, _root: ToolResult(
            action.name, False, {}, error_code="io_error", message="read failed"
        ),
    )
    prompts = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Recover", tmp_path, model=model)
    assert run.stop_reason == "invalid_action_limit"
    assert run.model_calls == run.steps == 5
    assert '"error_code":"io_error"' in prompts[2]


@pytest.mark.parametrize(
    "answer",
    [
        '{"kind":"complete","summary":"too late"}',
        '{"kind":"tool","name":"list_files","arguments":{}}',
    ],
)
def test_cancellation_after_model_response_prevents_tool_and_completion(
    tmp_path: Path, answer: str, monkeypatch
) -> None:
    cancelled = False
    calls = 0
    tool_calls = 0
    original = __import__(
        "harness.tools.dispatcher", fromlist=["execute_tool"]
    ).execute_tool

    def execute(action, root):
        nonlocal tool_calls
        tool_calls += 1
        return original(action, root)

    monkeypatch.setattr("harness.tools.dispatcher.execute_tool", execute)

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal cancelled, calls
        calls += 1
        cancelled = True
        return ChatResponse(answer)

    run = run_agent("Stop", tmp_path, model=model, cancelled=lambda: cancelled)
    assert run.status is RunStatus.CANCELLED
    assert run.stop_reason == "cancelled"
    assert run.final_response is None
    assert run.steps == 0 and calls == 1 and tool_calls == 0


def test_cancellation_before_model_and_keyboard_interrupt_are_clean(
    tmp_path: Path,
) -> None:
    run = run_agent(
        "Stop",
        tmp_path,
        model=lambda *_a, **_k: pytest.fail("called"),
        cancelled=lambda: True,
    )
    assert run.stop_reason == "cancelled" and run.model_calls == 0
    run = run_agent(
        "Stop",
        tmp_path,
        model=lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert run.status is RunStatus.CANCELLED
    assert run.stop_reason == "cancelled" and run.model_calls == 1


def test_cancellation_after_tool_execution_stops_run(
    tmp_path: Path, monkeypatch
) -> None:
    cancelled = False
    original = __import__(
        "harness.tools.dispatcher", fromlist=["execute_tool"]
    ).execute_tool

    def execute(action, root):
        nonlocal cancelled
        result = original(action, root)
        cancelled = True
        return result

    monkeypatch.setattr("harness.tools.dispatcher.execute_tool", execute)
    run = run_agent(
        "Stop",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(
            '{"kind":"tool","name":"list_files","arguments":{}}'
        ),
        cancelled=lambda: cancelled,
    )
    assert run.status is RunStatus.CANCELLED and run.stop_reason == "cancelled"
    assert run.model_calls == run.steps == 1


def test_request_timeout_is_forwarded_and_retry_recomputes_remaining_budget(
    tmp_path: Path, monkeypatch
) -> None:
    now = 0.0
    monkeypatch.setattr(runtime, "monotonic", lambda: now)
    timeouts = []

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal now
        timeouts.append(timeout)
        if len(timeouts) == 1:
            now = 4.0
            raise LlamaTransportError("disconnect")
        return ChatResponse('{"kind":"complete","summary":"ok"}')

    run = run_agent("Retry", tmp_path, model=model, request_timeout=30, run_timeout=20)
    assert run.status is RunStatus.COMPLETED
    assert timeouts == [20, 16]
    assert run.model_calls == 2


def test_retry_allowance_resets_after_successful_response(tmp_path: Path) -> None:
    attempts = 0

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal attempts
        attempts += 1
        if attempts in {1, 3}:
            raise LlamaTransportError("intermittent")
        if attempts == 2:
            return ChatResponse('{"kind":"tool","name":"list_files","arguments":{}}')
        return ChatResponse('{"kind":"complete","summary":"recovered"}')

    run = run_agent("Retry", tmp_path, model=model)
    assert run.status is RunStatus.COMPLETED
    assert run.model_calls == 4 and run.steps == 2


def test_retry_is_limited_by_call_budget_and_timeout_failure_classification(
    tmp_path: Path,
) -> None:
    calls = 0

    def transport(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        raise LlamaTransportError("offline")

    run = run_agent("Retry", tmp_path, model=transport, max_model_calls=1)
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "model_call_limit"
    assert run.model_calls == calls == 1

    def timeout(_prompt: str, *, timeout: float) -> ChatResponse:
        raise LlamaTimeoutError("late")

    run = run_agent("Retry", tmp_path, model=timeout)
    assert run.status is RunStatus.FAILED and run.stop_reason == "model_timeout"
    assert run.model_calls == 2


def test_malformed_client_response_is_not_retried_and_body_limit_has_no_step(
    tmp_path: Path,
) -> None:
    calls = 0

    def malformed(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        raise LlamaClientError("malformed chat response")

    run = run_agent("Fail", tmp_path, model=malformed)
    assert run.stop_reason == "model_error" and calls == run.model_calls == 1

    run = run_agent(
        "Fail",
        tmp_path,
        model=lambda _p, *, timeout: (_ for _ in ()).throw(
            LlamaResponseTooLargeError("large")
        ),
    )
    assert run.stop_reason == "model_output_limit"
    assert run.steps == 0


def test_cancellation_and_deadline_take_precedence_after_failed_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    cancelled = False

    def cancelled_model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal cancelled
        cancelled = True
        raise LlamaTransportError("disconnect")

    run = run_agent(
        "Cancel", tmp_path, model=cancelled_model, cancelled=lambda: cancelled
    )
    assert run.status is RunStatus.CANCELLED
    assert run.stop_reason == "cancelled"
    assert run.model_calls == 1

    now = 0.0
    monkeypatch.setattr(runtime, "monotonic", lambda: now)

    def late_model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal now
        now = 5.0
        raise LlamaTransportError("disconnect")

    run = run_agent("Deadline", tmp_path, model=late_model, run_timeout=4)
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "run_timeout"
    assert run.model_calls == 1


@pytest.mark.parametrize(
    "answer",
    [
        '{"kind":"complete","summary":"late"}',
        '{"kind":"tool","name":"list_files","arguments":{}}',
    ],
)
def test_success_after_deadline_prevents_completion_and_tool(
    tmp_path: Path, monkeypatch, answer: str
) -> None:
    now = 0.0
    tool_calls = 0
    original = __import__(
        "harness.tools.dispatcher", fromlist=["execute_tool"]
    ).execute_tool

    def execute(action, root):
        nonlocal tool_calls
        tool_calls += 1
        return original(action, root)

    monkeypatch.setattr(runtime, "monotonic", lambda: now)
    monkeypatch.setattr("harness.tools.dispatcher.execute_tool", execute)

    def late_model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal now
        now = 5.0
        return ChatResponse(answer)

    run = run_agent("Deadline", tmp_path, model=late_model, run_timeout=4)
    assert run.status is RunStatus.LIMIT_REACHED and run.stop_reason == "run_timeout"
    assert run.final_response is None and run.steps == 0 and tool_calls == 0


def test_cancellation_wins_when_deadline_is_also_exceeded(
    tmp_path: Path, monkeypatch
) -> None:
    now = 0.0
    cancelled = False
    monkeypatch.setattr(runtime, "monotonic", lambda: now)

    def late_model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal now, cancelled
        now = 5.0
        cancelled = True
        return ChatResponse('{"kind":"complete","summary":"late"}')

    run = run_agent(
        "Deadline",
        tmp_path,
        model=late_model,
        run_timeout=4,
        cancelled=lambda: cancelled,
    )
    assert run.status is RunStatus.CANCELLED and run.stop_reason == "cancelled"


def test_deadline_reached_between_boundary_check_and_timeout_calculation(
    tmp_path: Path, monkeypatch
) -> None:
    times = iter([0.0, 0.0, 0.0, 0.0, 4.0, 4.0])
    monkeypatch.setattr(runtime, "monotonic", lambda: next(times))
    calls = 0

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse('{"kind":"complete","summary":"bad"}')

    run = run_agent("Deadline", tmp_path, model=model, run_timeout=4)
    assert run.stop_reason == "run_timeout"
    assert run.model_calls == calls == 0


@pytest.mark.parametrize(
    "interrupt,expected",
    [("cancel", "cancelled"), ("deadline", "run_timeout")],
)
def test_cancellation_or_deadline_after_prompt_precedes_prompt_budget(
    tmp_path: Path, monkeypatch, interrupt: str, expected: str
) -> None:
    now = 0.0
    cancelled = False
    monkeypatch.setattr(runtime, "monotonic", lambda: now)

    def build_prompt(*_args):
        nonlocal now, cancelled
        if interrupt == "cancel":
            cancelled = True
        else:
            now = 5.0
        return None

    monkeypatch.setattr(runtime, "_build_prompt", build_prompt)
    run = run_agent(
        "Prompt",
        tmp_path,
        cancelled=lambda: cancelled,
        run_timeout=4,
        max_prompt_bytes=1,
    )
    assert run.stop_reason == expected


def test_oversized_unicode_model_text_stops_before_action_parse(tmp_path: Path) -> None:
    run = run_agent(
        "Inspect",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(
            '{"kind":"complete","summary":"' + "é" * 40 + '"}'
        ),
        max_model_response_bytes=64,
    )
    assert run.stop_reason == "model_output_limit"
    assert run.steps == 1 and run.final_response is None


def test_repeated_invalid_json_and_ordinary_tool_error_recover(tmp_path: Path) -> None:
    outputs = iter(
        [
            "[" * 2000 + "]" * 2000,
            '{"kind":"tool","name":"list_files","arguments":{}}',
            '{"kind":"complete","summary":"Recovered"}',
        ]
    )
    run = run_agent(
        "Recover", tmp_path, model=lambda _p, *, timeout: ChatResponse(next(outputs))
    )
    assert run.status is RunStatus.COMPLETED
    assert run.steps == run.model_calls == 3


@pytest.mark.parametrize(
    "task,workspace,kwargs",
    [
        (" ", Path("."), {}),
        ("task", Path("missing-workspace"), {}),
        ("task", Path("."), {"max_model_calls": True}),
        ("task", Path("."), {"max_prompt_bytes": 0}),
        ("task", Path("."), {"max_steps": True}),
        ("task", Path("."), {"max_consecutive_invalid_actions": 0}),
        ("task", Path("."), {"request_timeout": float("inf")}),
        ("task", Path("."), {"run_timeout": True}),
        ("task", Path("."), {"max_model_response_bytes": 0}),
        ("task", Path("."), {"cancelled": False}),
    ],
)
def test_invalid_invocation_raises_before_model(
    tmp_path: Path, task: str, workspace: Path, kwargs: dict[str, object]
) -> None:
    calls = 0

    def model(_: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse("")

    with pytest.raises(ValueError):
        run_agent(
            task,
            tmp_path if workspace == Path(".") else workspace,
            model=model,
            **kwargs,
        )
    assert calls == 0


def test_legacy_model_signature_is_rejected_before_inference(tmp_path: Path) -> None:
    calls = 0

    def model(_prompt: str) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse("ok")

    with pytest.raises(ValueError, match="timeout"):
        run_agent("Task", tmp_path, model=model)
    assert calls == 0


@pytest.mark.parametrize(
    "statement",
    [
        "from harness.tools import execute_tool",
        "from harness.agent import run_agent",
    ],
)
def test_public_imports_work_in_fresh_process(statement: str) -> None:
    subprocess.run([sys.executable, "-c", statement], check=True)
