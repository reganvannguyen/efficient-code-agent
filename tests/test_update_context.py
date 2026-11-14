"""Deterministic tests for model-initiated update_context / context_edit (#30)."""

from __future__ import annotations

import json
from pathlib import Path

from harness.agent.events import RunEventType
from harness.agent.protocol import ContextEditAction, InvalidAction, parse_action
from harness.agent.runtime import action_schema, run_agent
from harness.agent.state import RunStatus
from harness.context import DEFAULT_MAX_NOTE_BYTES, WorkingMemoryStore
from harness.llama_client import ChatResponse


def _resp(payload: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(payload))


def test_parse_action_normalizes_update_context_and_rejects_malformed() -> None:
    parsed = parse_action(
        json.dumps(
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "# Working Context\nConfirmed.",
                },
            }
        )
    )
    assert parsed == ContextEditAction(
        operation="replace",
        target="working.md",
        content="# Working Context\nConfirmed.",
    )

    direct = parse_action(
        json.dumps(
            {
                "kind": "context_edit",
                "operation": "append",
                "target": "working.md",
                "content": "- Extra note",
                "run_id": "run-123",
            }
        )
    )
    assert direct == ContextEditAction(
        operation="append",
        target="working.md",
        content="- Extra note",
        run_id="run-123",
    )

    bad = parse_action(
        json.dumps(
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {"operation": "replace", "target": "working.md"},
            }
        )
    )
    assert isinstance(bad, InvalidAction)


def test_model_replaces_hypothesis_and_sees_update_on_next_call(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"
    prompts: list[str] = []

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "# Working Context\n\n## Current Hypothesis\nBug is in cache.\n",
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "# Working Context\n\n## Confirmed Findings\n- Bug is in parser.py:12.\n",
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "append",
                    "target": "working.md",
                    "content": "## Next Action\nFix parser.py:12.",
                },
            },
            {"kind": "complete", "summary": "Confirmed parser.py:12 root cause."},
        ]
    )

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return _resp(next(actions))

    run = run_agent(
        "Find the root cause",
        repo,
        model=model,
        context_strategy="editable_context",
        memory_dir=memory_dir,
    )
    assert run.status == RunStatus.COMPLETED
    assert "Bug is in cache." in prompts[1]
    assert "Bug is in cache." not in prompts[2]
    assert "Bug is in parser.py:12." in prompts[2]
    assert "Fix parser.py:12." in prompts[3]

    edit_events = [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT]
    assert len(edit_events) == 3
    assert all(e.data["accepted"] is True for e in edit_events)
    assert edit_events[-1].data["note_bytes"] > 0

    store = WorkingMemoryStore(memory_dir, run.run_id)
    assert store.read_metadata()["original_task"] == "Find the root cause"


def test_rejected_context_edits_leave_prior_notes_untouched(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "# Working Context\nSafe note.",
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "delete",
                    "target": "working.md",
                    "content": "",
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "metadata.json",
                    "content": '{"original_task":"hijacked"}',
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "hi",
                    "run_id": "other-run-id",
                },
            },
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "working.md",
                    "content": "x" * (DEFAULT_MAX_NOTE_BYTES + 10),
                },
            },
            {"kind": "complete", "summary": "Done safely."},
        ]
    )

    run = run_agent(
        "Original goal",
        repo,
        model=lambda _p, *, timeout: _resp(next(actions)),
        context_strategy="editable_context",
        memory_dir=memory_dir,
        max_consecutive_invalid_actions=6,
    )
    assert run.status == RunStatus.COMPLETED
    store = WorkingMemoryStore(memory_dir, run.run_id)
    assert store.read_note("working.md") == "# Working Context\nSafe note."
    assert store.read_metadata()["original_task"] == "Original goal"

    edit_events = [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT]
    assert [e.data["accepted"] for e in edit_events] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert [e.data["error_code"] for e in edit_events[1:]] == [
        "unsupported_operation",
        "invalid_target",
        "scope_mismatch",
        "note_limit_exceeded",
    ]


def test_run_without_context_edits_completes_and_schema_includes_when_enabled(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

    actions = iter(
        [
            {"kind": "tool", "name": "list_files", "arguments": {}},
            {"kind": "complete", "summary": "Listed files without editing notes."},
        ]
    )
    run = run_agent(
        "List files",
        repo,
        model=lambda _p, *, timeout: _resp(next(actions)),
        context_strategy="editable_context",
    )
    assert run.status == RunStatus.COMPLETED
    assert not [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT]

    default_names = {
        v["properties"]["name"]["const"]
        for v in action_schema()["oneOf"]
        if "name" in v["properties"]
    }
    editable_names = {
        v["properties"]["name"]["const"]
        for v in action_schema(include_context_edit=True)["oneOf"]
        if "name" in v["properties"]
    }
    assert "update_context" not in default_names
    assert "update_context" in editable_names


def test_update_context_can_create_findings_and_surfaces_memory_index(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"
    prompts: list[str] = []

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "findings/auth.md",
                    "content": "# Auth Finding\n- Token leeway is 45s.\n",
                },
            },
            {"kind": "complete", "summary": "Saved findings/auth.md."},
        ]
    )

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return _resp(next(actions))

    run = run_agent(
        "Save auth finding",
        repo,
        model=model,
        context_strategy="editable_context",
        memory_dir=memory_dir,
    )
    assert run.status == RunStatus.COMPLETED
    assert (
        "Available memory notes (read with read_memory): working.md, task.md"
        in prompts[0]
    )
    assert "findings/auth.md" in prompts[1]
    store = WorkingMemoryStore(memory_dir, run.run_id)
    assert "Token leeway is 45s." in store.read_note("findings/auth.md")
