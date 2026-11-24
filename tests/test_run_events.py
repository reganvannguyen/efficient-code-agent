"""Tests for structured run events, identifiers, and outcome taxonomy."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from harness.agent.events import BenchmarkOutcome, RunEvent, RunEventType
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.cli import main
from harness.llama_client import (
    ChatResponse,
    LlamaClientError,
    LlamaTimeoutError,
)
from harness.tools.policy import ToolMode, ToolPolicy


def _response(
    content: dict[str, object] | str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> ChatResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return ChatResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def test_run_event_create_and_serialization() -> None:
    run_id = str(uuid.uuid4())
    event = RunEvent.create(
        run_id=run_id,
        sequence=1,
        event_type=RunEventType.RUN_STARTED,
        step=0,
        duration_ms=12.5,
        data={"task": "inspect"},
    )
    assert event.run_id == run_id
    assert event.sequence == 1
    assert event.event_type is RunEventType.RUN_STARTED
    assert event.step == 0
    assert event.duration_ms == 12.5
    assert event.data == {"task": "inspect"}

    d = event.to_dict()
    assert d["run_id"] == run_id
    assert d["sequence"] == 1
    assert d["event_type"] == "run_started"
    assert d["step"] == 0
    assert d["duration_ms"] == 12.5
    assert d["data"] == {"task": "inspect"}
    assert isinstance(d["timestamp"], float)


def test_run_events_sequence_and_identifier(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world", encoding="utf-8")
    actions = iter(
        [
            {"kind": "tool", "name": "read_file", "arguments": {"path": "hello.txt"}},
            {"kind": "complete", "summary": "Found world."},
        ]
    )

    events_streamed: list[RunEvent] = []

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        return _response(
            next(actions), prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

    run = run_agent(
        "inspect file",
        tmp_path,
        model=model,
        on_event=events_streamed.append,
    )

    assert run.status is RunStatus.COMPLETED
    assert uuid.UUID(run.run_id)  # Valid UUID4
    assert run.start_time is not None
    assert run.end_time is not None
    assert run.duration_seconds is not None
    assert run.duration_seconds >= 0.0

    # Ensure events stored on run match streamed events
    assert run.events == events_streamed
    assert len(run.events) >= 6

    # Verify monotonic sequences
    for i, event in enumerate(run.events, start=1):
        assert event.sequence == i
        assert event.run_id == run.run_id

    # Verify event types order
    types = [e.event_type for e in run.events]
    assert types[0] is RunEventType.RUN_STARTED
    assert types[-1] is RunEventType.RUN_FINISHED
    assert RunEventType.MODEL_REQUEST in types
    assert RunEventType.MODEL_RESPONSE in types
    assert RunEventType.TOOL_REQUESTED in types
    assert RunEventType.TOOL_RESULT in types
    assert RunEventType.VERIFICATION_RESULT in types

    # Verify token totals
    assert run.total_prompt_tokens == 20
    assert run.total_completion_tokens == 10
    assert run.total_tokens == 30

    # Verify benchmark outcome default on completion
    assert run.benchmark_outcome is BenchmarkOutcome.NOT_EVALUATED

    # Verify finish event data
    finish_data = run.events[-1].data
    assert finish_data["status"] == "completed"
    assert finish_data["stop_reason"] == "model_complete"
    assert finish_data["benchmark_outcome"] == "not_evaluated"
    assert finish_data["total_prompt_tokens"] == 20
    assert finish_data["total_completion_tokens"] == 10
    assert finish_data["total_tokens"] == 30


def test_token_counts_remain_none_when_unreported(tmp_path: Path) -> None:
    actions = iter([{"kind": "complete", "summary": "done"}])

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        return _response(next(actions))  # No tokens supplied

    run = run_agent("test task", tmp_path, model=model)
    assert run.total_prompt_tokens is None
    assert run.total_completion_tokens is None
    assert run.total_tokens is None

    # Model response event data
    response_events = [
        e for e in run.events if e.event_type is RunEventType.MODEL_RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data["prompt_tokens"] is None
    assert response_events[0].data["completion_tokens"] is None
    assert response_events[0].data["total_tokens"] is None


def test_benchmark_outcome_on_limits_and_errors(tmp_path: Path) -> None:
    # 1. Step limit -> UNRESOLVED
    run1 = run_agent(
        "test limit",
        tmp_path,
        model=lambda _p, *, timeout: _response(
            {"kind": "tool", "name": "list_files", "arguments": {}}
        ),
        max_steps=2,
    )
    assert run1.status is RunStatus.LIMIT_REACHED
    assert run1.benchmark_outcome is BenchmarkOutcome.UNRESOLVED
    assert any(e.event_type is RunEventType.BUDGET_EXCEEDED for e in run1.events)

    # 2. Run timeout -> TIMEOUT
    import time

    def slow_model(_p: str, *, timeout: float) -> ChatResponse:
        time.sleep(0.02)
        return _response({"kind": "complete", "summary": "done"})

    run2 = run_agent(
        "test timeout",
        tmp_path,
        model=slow_model,
        run_timeout=0.005,
    )
    assert run2.status is RunStatus.LIMIT_REACHED
    assert run2.benchmark_outcome is BenchmarkOutcome.TIMEOUT

    # 3. Model timeout -> TIMEOUT
    call_count = 0

    def timeout_model(_p: str, *, timeout: float) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        raise LlamaTimeoutError("timed out")

    run3 = run_agent("test model timeout", tmp_path, model=timeout_model)
    assert run3.status is RunStatus.FAILED
    assert run3.benchmark_outcome is BenchmarkOutcome.AGENT_ERROR
    error_events = [e for e in run3.events if e.event_type is RunEventType.MODEL_ERROR]
    assert len(error_events) >= 1
    assert error_events[0].data["error_type"] == "LlamaTimeoutError"

    # 4. Model client error -> AGENT_ERROR
    def client_err_model(_p: str, *, timeout: float) -> ChatResponse:
        raise LlamaClientError("bad status")

    run4 = run_agent("test client error", tmp_path, model=client_err_model)
    assert run4.status is RunStatus.FAILED
    assert run4.benchmark_outcome is BenchmarkOutcome.AGENT_ERROR

    # 5. Cancelled -> UNRESOLVED
    run5 = run_agent(
        "test cancelled",
        tmp_path,
        model=lambda _p, *, timeout: _response({"kind": "complete", "summary": "done"}),
        cancelled=lambda: True,
    )
    assert run5.status is RunStatus.CANCELLED
    assert run5.benchmark_outcome is BenchmarkOutcome.UNRESOLVED
    assert any(e.event_type is RunEventType.CANCELLED for e in run5.events)


def test_on_event_callback_exception_tolerance(tmp_path: Path) -> None:
    def buggy_callback(event: RunEvent) -> None:
        raise RuntimeError("callback explosion")

    run = run_agent(
        "inspect",
        tmp_path,
        model=lambda _p, *, timeout: _response({"kind": "complete", "summary": "ok"}),
        on_event=buggy_callback,
    )
    assert run.status is RunStatus.COMPLETED
    assert len(run.events) >= 3


def test_on_event_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="on_event must be callable or None"):
        run_agent("task", tmp_path, on_event="not_a_function")  # type: ignore[arg-type]


def test_coding_approval_events(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("print('old')\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(b"print('old')\n").hexdigest()
    actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "code.py",
                    "expected_sha256": digest,
                    "old_text": "print('old')\n",
                    "new_text": "print('new')\n",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_sample.py"},
            },
            {"kind": "complete", "summary": "done"},
        ]
    )

    policy = ToolPolicy(
        edit_file=ToolMode.ASK,
        run_tests=ToolMode.AUTO,
        test_targets=("./test_sample.py/",),
    )
    approve_mock = MagicMock(return_value=True)

    run = run_agent(
        "edit code",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
        approve=approve_mock,
    )

    types = [e.event_type for e in run.events]
    assert RunEventType.APPROVAL_REQUESTED in types
    assert RunEventType.APPROVAL_RESOLVED in types

    req_event = next(
        e for e in run.events if e.event_type is RunEventType.APPROVAL_REQUESTED
    )
    assert req_event.data["tool"] == "edit_file"
    assert "code.py" in str(req_event.data["target"])

    res_event = next(
        e for e in run.events if e.event_type is RunEventType.APPROVAL_RESOLVED
    )
    assert res_event.data["approved"] is True


def test_cli_outputs_run_id_tokens_and_benchmark_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "harness.cli.chat",
        lambda *_a, **_k: _response(
            {"kind": "complete", "summary": "cli done"},
            prompt_tokens=42,
            completion_tokens=18,
            total_tokens=60,
        ),
    )

    main(["run", "test task", "--repo", str(tmp_path)])
    captured = capsys.readouterr()

    assert "cli done\n" in captured.out
    assert "harness: run_id=" in captured.err
    assert "harness: benchmark_outcome=not_evaluated" in captured.err
    assert "harness: tokens=prompt:42,completion:18,total:60" in captured.err
