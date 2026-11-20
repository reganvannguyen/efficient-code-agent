"""Deterministic tests for optional bounded read_memory (#31)."""

from __future__ import annotations

import json
from pathlib import Path

from harness.agent.runtime import action_schema, run_agent
from harness.agent.state import RunStatus
from harness.context import WorkingMemoryStore
from harness.llama_client import ChatResponse


def _resp(payload: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(payload))


def test_read_memory_returns_bounded_finding_and_keeps_default_prompt_compact(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"

    store = WorkingMemoryStore(memory_dir, "sess-1")
    store.initialize("Debug auth", run_id="sess-1")
    store.write_finding(
        "auth-trace",
        title="Auth Trace",
        body="Token expiry mismatch at line 42. " + ("detail " * 80),
        source="src/auth.py:40-48",
        observed_at="2026-11-14T16:00:00Z",
    )
    store.write_finding(
        "db-notes",
        title="DB Notes",
        body="Unrelated database pool notes.",
        source="src/db.py:10",
    )

    prompts: list[str] = []
    actions = iter(
        [
            {"kind": "tool", "name": "read_memory", "arguments": {}},
            {
                "kind": "tool",
                "name": "read_memory",
                "arguments": {"target": "findings/auth-trace.md", "max_bytes": 120},
            },
            {"kind": "complete", "summary": "Read bounded auth-trace finding."},
        ]
    )

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return _resp(next(actions))

    run = run_agent(
        "Check auth findings",
        repo,
        model=model,
        context_strategy="editable_context",
        session_id="sess-1",
        memory_dir=memory_dir,
    )
    assert run.status == RunStatus.COMPLETED
    # Default prompt does not stuff optional findings/*.md automatically
    assert "Unrelated database pool notes." not in prompts[0]
    assert "Token expiry mismatch" not in prompts[0]
    # Listing returned available findings
    assert "findings/auth-trace.md" in prompts[1]
    assert "findings/db-notes.md" in prompts[1]
    # Bounded read returned truncated content with source metadata
    assert "src/auth.py:40-48" in prompts[2]
    assert '"truncated":true' in prompts[2]


def test_read_memory_rejects_other_runs_escapes_and_missing_notes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"

    other_store = WorkingMemoryStore(memory_dir, "other-run")
    other_store.initialize("Secret task", run_id="other-run")
    other_store.write_finding(
        "secret",
        title="Secret",
        body="Should never leak",
        source="secret.py:1",
    )

    prompts: list[str] = []
    actions = iter(
        [
            {
                "kind": "tool",
                "name": "read_memory",
                "arguments": {"target": "findings/missing.md"},
            },
            {
                "kind": "tool",
                "name": "read_memory",
                "arguments": {"target": "../other-run/findings/secret.md"},
            },
            {
                "kind": "tool",
                "name": "read_memory",
                "arguments": {"target": "working.md", "run_id": "other-run"},
            },
            {
                "kind": "tool",
                "name": "read_memory",
                "arguments": {"target": "metadata.json"},
            },
            {"kind": "complete", "summary": "Handled missing and denied reads."},
        ]
    )

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        prompts.append(prompt)
        return _resp(next(actions))

    run = run_agent(
        "Test boundaries",
        repo,
        model=model,
        context_strategy="editable_context",
        session_id="current-run",
        memory_dir=memory_dir,
        max_consecutive_invalid_actions=5,
    )
    assert run.status == RunStatus.COMPLETED
    joined = "\n".join(prompts)
    assert "Should never leak" not in joined
    assert '"error_code":"memory_not_found"' in prompts[1]
    assert '"error_code":"path_denied"' in prompts[2]
    assert '"error_code":"scope_mismatch"' in prompts[3]
    assert '"error_code":"path_denied"' in prompts[4]


def test_read_memory_schema_gating() -> None:
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
    assert "read_memory" not in default_names
    assert "read_memory" in editable_names


def test_read_memory_omitted_from_schema_until_findings_exist(tmp_path: Path) -> None:
    from functools import partial

    repo = tmp_path / "repo"
    repo.mkdir()
    memory_dir = tmp_path / ".harness" / "runs"

    seen_tools: list[set[str]] = []
    actions = iter(
        [
            {
                "kind": "tool",
                "name": "update_context",
                "arguments": {
                    "operation": "replace",
                    "target": "findings/note.md",
                    "content": "Confirmed bug at line 10.",
                },
            },
            {"kind": "complete", "summary": "Done."},
        ]
    )

    def model(
        prompt: str, *, timeout: float, response_schema: object = None
    ) -> ChatResponse:
        assert isinstance(response_schema, dict)
        names = {
            v["properties"]["name"]["const"]
            for v in response_schema.get("oneOf", [])
            if "name" in v.get("properties", {})
        }
        seen_tools.append(names)
        return _resp(next(actions))

    wrapped = partial(model, response_schema=action_schema(include_context_edit=True))
    run = run_agent(
        "Check findings gating",
        repo,
        model=wrapped,
        context_strategy="editable_context",
        memory_dir=memory_dir,
    )
    assert run.status == RunStatus.COMPLETED
    assert len(seen_tools) == 2
    assert "read_memory" not in seen_tools[0]
    assert "read_memory" in seen_tools[1]
