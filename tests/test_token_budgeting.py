"""Tests for model-compatible token budgets, deterministic pruning, and output capping."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from harness.agent.events import RunEventType
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.context import (
    ContextInput,
    EditableContextStrategy,
    LlamaTokenCounter,
    TokenBudget,
    estimate_tokens,
)
from harness.llama_client import ChatResponse, LlamaClientError, tokenize


def _word_tokens(text: str) -> int:
    return len(text.split())


def test_token_budget_computes_input_budget_and_validates() -> None:
    budget = TokenBudget(
        context_window=8192,
        output_reserve=1024,
        overhead_tokens=512,
        safety_margin=128,
    )
    assert budget.input_budget == 6528

    with pytest.raises(ValueError, match="context_window"):
        TokenBudget(context_window=0)
    with pytest.raises(ValueError, match="non-negative"):
        TokenBudget(context_window=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_budget must be positive"):
        TokenBudget(
            context_window=1000,
            output_reserve=600,
            overhead_tokens=300,
            safety_margin=100,
        )


def test_tokenize_endpoint_parses_tokens_and_handles_errors(monkeypatch) -> None:
    assert tokenize("") == []

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self, _limit: int = -1) -> bytes:
            return self._payload

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "harness.llama_client.urlopen",
        lambda _req, timeout: _FakeResponse(b'{"tokens":[101, 202, 303]}'),
    )
    assert tokenize("hello world") == [101, 202, 303]

    def _raise_http(_req, timeout):
        raise HTTPError("http://127.0.0.1:9931/tokenize", 500, "err", {}, io.BytesIO())

    monkeypatch.setattr("harness.llama_client.urlopen", _raise_http)
    with pytest.raises(LlamaClientError, match="HTTP 500"):
        tokenize("hello")

    counter = LlamaTokenCounter(fallback=estimate_tokens)
    assert counter.count_tokens("hello world") == estimate_tokens("hello world")


def test_editable_context_pruning_order_is_deterministic_and_protects_task() -> None:
    strategy = EditableContextStrategy()

    def build_once() -> tuple[str | None, dict[str, object]]:
        ctx = ContextInput(
            instructions="IMMUTABLE_RULES",
            directive="NEXT_STEP",
            task="IMMUTABLE_TASK",
            history=["old_1 " * 20, "old_2 " * 20, "newest_exchange"],
            byte_limit=10_000,
            token_limit=35,
            token_counter=_word_tokens,
            task_notes="## Confirmed Findings\nline1\nline2\nline3\nline4\n"
            + ("extra " * 30),
            repo_excerpts=("def foo(): pass\n" * 10,),
            memory_index=("working.md", "task.md", "findings/a.md"),
        )
        prompt, report = strategy.build_prompt_with_report(ctx)
        return prompt, report.to_dict()

    prompt_a, report_a = build_once()
    prompt_b, report_b = build_once()
    assert prompt_a == prompt_b
    assert report_a == report_b
    assert prompt_a is not None
    assert _word_tokens(prompt_a) <= 35
    assert "IMMUTABLE_RULES" in prompt_a
    assert "IMMUTABLE_TASK" in prompt_a
    assert "NEXT_STEP" in prompt_a
    assert "newest_exchange" in prompt_a
    assert report_a["pruned"] is True
    assert report_a["dropped_history_count"] == 2
    assert report_a["capped_excerpts_count"] >= 1
    assert report_a["omitted_findings_count"] == 3
    assert report_a["trimmed_notes"] is True


def test_protected_content_overflow_fails_with_context_budget_exceeded(
    tmp_path: Path,
) -> None:
    calls = 0

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal calls
        calls += 1
        return ChatResponse('{"kind":"complete","summary":"done"}')

    run = run_agent(
        "word " * 50,
        tmp_path,
        model=model,
        context_strategy="editable_context",
        token_budget=TokenBudget(
            context_window=100,
            output_reserve=40,
            overhead_tokens=30,
            safety_margin=10,
        ),
        token_counter=_word_tokens,
    )
    assert calls == 0
    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "context_budget_exceeded"


def test_immense_tool_output_is_capped_with_artifact_and_pruning_logged_separately(
    tmp_path: Path,
) -> None:
    (tmp_path / "huge.txt").write_text("alpha " * 1500, encoding="utf-8")
    artifacts_dir = tmp_path / ".artifacts"

    responses = iter(
        [
            ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "read_file",
                        "arguments": {"path": "huge.txt"},
                    }
                )
            ),
            ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "list_files",
                        "arguments": {},
                    }
                )
            ),
            ChatResponse(
                json.dumps({"kind": "complete", "summary": "Inspected repo."})
            ),
        ]
    )
    seen_prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        seen_prompts.append(prompt)
        return next(responses)

    run = run_agent(
        "Inspect huge.txt",
        tmp_path,
        model=model,
        context_strategy="editable_context",
        token_budget=TokenBudget(
            context_window=2048,
            output_reserve=256,
            overhead_tokens=128,
            safety_margin=64,
        ),
        token_counter=_word_tokens,
        max_tool_output_bytes=512,
        tool_artifacts_dir=artifacts_dir,
    )

    assert run.status is RunStatus.COMPLETED
    assert len(seen_prompts) == 3
    assert all(_word_tokens(p) <= 1600 for p in seen_prompts)
    assert '"truncated":true' in seen_prompts[1]
    artifact_files = list(artifacts_dir.glob("step_0001_read_file.json"))
    assert len(artifact_files) == 1
    assert "alpha" in artifact_files[0].read_text(encoding="utf-8")

    # Forced pruning events are distinct from model-initiated CONTEXT_EDIT events
    context_edits = [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT]
    assert context_edits == []
