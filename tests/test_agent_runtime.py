import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.agent import RunStatus, ToolAction, run_agent
from harness.agent.runtime import _build_prompt
from harness.llama_client import ChatResponse, LlamaClientError
from harness.tools import execute_tool


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

    def model(prompt: str) -> ChatResponse:
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


def test_independent_runs_do_not_share_history_counters_or_answers(
    tmp_path: Path,
) -> None:
    first_prompts: list[str] = []
    second_prompts: list[str] = []

    def first_model(prompt: str) -> ChatResponse:
        first_prompts.append(prompt)
        if len(first_prompts) == 1:
            return ChatResponse('{"kind":"tool","name":"list_files","arguments":{}}')
        return ChatResponse('{"kind":"complete","summary":"first answer"}')

    def second_model(prompt: str) -> ChatResponse:
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

    def model(prompt: str) -> ChatResponse:
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

    def model(_: str) -> ChatResponse:
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
        "Keep going", tmp_path, model=lambda _: ChatResponse(answer), max_model_calls=3
    )

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "model_call_limit"
    assert run.model_calls == run.steps == 3


def test_model_failure_counts_attempt_and_fails_without_step(tmp_path: Path) -> None:
    def model(_: str) -> ChatResponse:
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
    assert len(prompt.encode("utf-8")) <= byte_limit
    assert history == [
        '{"action":{"i":1},"observation":{"text":"middle"}}',
        '{"action":{"i":2},"observation":{"text":"newest"}}',
    ]


def test_oversized_prompt_stops_without_inference(tmp_path: Path) -> None:
    calls = 0

    def model(_: str) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse(
            '{"kind":"tool","name":"read_file","arguments":{"path":"one"}}'
        )

    run = run_agent("Answer", tmp_path, model=model, max_prompt_bytes=1400)

    assert calls == 1
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "prompt_budget"
    assert run.model_calls == run.steps == 1


def test_oversized_task_stops_before_inference(tmp_path: Path) -> None:
    calls = 0

    def model(_: str) -> ChatResponse:
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

    def model(prompt: str) -> ChatResponse:
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

    def model(prompt: str) -> ChatResponse:
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

    def model(prompt: str) -> ChatResponse:
        prompts.append(prompt)
        return ChatResponse(next(outputs))

    run = run_agent("Recover from invalid output", tmp_path, model=model)
    history_label = "Action and observation history (JSON exchanges, oldest first):\n"
    exchange = prompts[1].split(history_label, 1)[1]
    excerpt = json.loads(exchange)["observation"]["excerpt"]

    assert run.status is RunStatus.COMPLETED
    assert len(excerpt.encode("utf-8")) <= 1024
    assert excerpt == "é" * 512


@pytest.mark.parametrize(
    "task,workspace,kwargs",
    [
        (" ", Path("."), {}),
        ("task", Path("missing-workspace"), {}),
        ("task", Path("."), {"max_model_calls": True}),
        ("task", Path("."), {"max_prompt_bytes": 0}),
    ],
)
def test_invalid_invocation_raises_before_model(
    tmp_path: Path, task: str, workspace: Path, kwargs: dict[str, object]
) -> None:
    calls = 0

    def model(_: str) -> ChatResponse:
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


@pytest.mark.parametrize(
    "statement",
    [
        "from harness.tools import execute_tool",
        "from harness.agent import run_agent",
    ],
)
def test_public_imports_work_in_fresh_process(statement: str) -> None:
    subprocess.run([sys.executable, "-c", statement], check=True)
