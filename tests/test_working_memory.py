"""Deterministic tests for task- and session-scoped Markdown working memory (#29)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.context import (
    MemoryStorageError,
    WorkingMemoryStore,
)
from harness.evaluation.workspace import capture_workspace_patch
from harness.llama_client import ChatResponse


def _complete(summary: str) -> ChatResponse:
    return ChatResponse(json.dumps({"kind": "complete", "summary": summary}))


def test_two_runs_are_isolated_and_findings_are_lazy(tmp_path: Path) -> None:
    base = tmp_path / ".harness" / "runs"
    store_a = WorkingMemoryStore(base, "run-a")
    store_b = WorkingMemoryStore(base, "run-b")

    store_a.initialize("Fix bug A", run_id="run-a")
    store_b.initialize("Fix bug B", run_id="run-b")

    assert not store_a.findings_dir.exists()
    assert not store_b.findings_dir.exists()

    store_a.write_note("working.md", "# Working Context\n\n## Goal\nFix bug A\n")
    assert "Fix bug A" in store_a.read_note("working.md")
    assert "Fix bug A" not in store_b.read_note("working.md")
    assert "Fix bug B" in store_b.read_note("working.md")

    store_a.write_finding(
        "auth-check",
        title="Auth Expiry",
        body="Token expiry uses naive datetime.",
        source="src/auth.py:40-48",
    )
    assert store_a.list_findings() == ["findings/auth-check.md"]
    assert store_b.list_findings() == []
    assert "Source: src/auth.py:40-48" in store_a.read_note("findings/auth-check.md")


def test_failed_or_oversized_writes_preserve_existing_notes(tmp_path: Path) -> None:
    store = WorkingMemoryStore(tmp_path / "runs", "run-1", max_note_bytes=256)
    store.initialize("Task 1", run_id="run-1")
    original = "# Working Context\n\nKeep me intact.\n"
    store.write_note("working.md", original)

    with pytest.raises(MemoryStorageError, match="exceeds byte limit"):
        store.write_note("working.md", "x" * 300)
    assert store.read_note("working.md") == original

    with patch("harness.context.memory.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            store.write_note("working.md", "# Working Context\n\nPartial\n")
    assert store.read_note("working.md") == original
    leftover_temps = [
        p for p in store.run_dir.iterdir() if p.name.startswith(".harness-mem-")
    ]
    assert leftover_temps == []


def test_immutable_metadata_and_path_escapes_are_rejected(tmp_path: Path) -> None:
    store = WorkingMemoryStore(tmp_path / "runs", "run-1")
    store.initialize(
        "Immutable task", run_id="run-1", policy_summary={"read_only": True}
    )
    before_meta = store.read_metadata()

    for bad_target in (
        "metadata.json",
        "../other/working.md",
        "/etc/passwd",
        "findings/../../escape.md",
        "src/main.py",
        "working.txt",
    ):
        with pytest.raises(MemoryStorageError):
            store.write_note(bad_target, "hacked")

    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (store.run_dir / "findings").mkdir()
    (store.run_dir / "findings" / "link.md").symlink_to(outside)
    with pytest.raises(MemoryStorageError, match="Symlinked"):
        store.write_note("findings/link.md", "overwrite")

    assert store.read_metadata() == before_meta
    assert outside.read_text(encoding="utf-8") == "secret"


def test_session_id_reuses_working_memory_across_turns(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    memory_dir = tmp_path / ".harness" / "runs"

    run1 = run_agent(
        "Turn 1: inspect app.py",
        repo,
        model=lambda _p, *, timeout: _complete("Inspected app.py."),
        context_strategy="editable_context",
        session_id="chat-1",
        memory_dir=memory_dir,
    )
    assert run1.status == RunStatus.COMPLETED

    store = WorkingMemoryStore(memory_dir, "chat-1")
    store.append_note("working.md", "\n- Confirmed: x = 1 in app.py:1\n")

    seen_prompts: list[str] = []

    def capture_model(prompt: str, *, timeout: float) -> ChatResponse:
        seen_prompts.append(prompt)
        return _complete("Turn 2 done.")

    run2 = run_agent(
        "Turn 2: follow up",
        repo,
        model=capture_model,
        context_strategy="editable_context",
        session_id="chat-1",
        memory_dir=memory_dir,
    )
    assert run2.status == RunStatus.COMPLETED
    assert seen_prompts
    assert "Confirmed: x = 1 in app.py:1" in seen_prompts[0]
    assert store.read_metadata()["original_task"] == "Turn 1: inspect app.py"


def test_harness_memory_excluded_from_candidate_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "main.py").write_text("print(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )

    store = WorkingMemoryStore(repo / ".harness" / "runs", "run-diff")
    store.initialize("Edit main.py", run_id="run-diff")
    (repo / "main.py").write_text("print(2)\n", encoding="utf-8")

    patch_text = capture_workspace_patch(repo)
    assert "main.py" in patch_text
    assert ".harness" not in patch_text
    assert "working.md" not in patch_text
