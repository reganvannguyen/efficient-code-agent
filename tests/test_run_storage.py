"""Tests for SQLite trace persistence, run artifacts, and configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agent.runtime import run_agent
from harness.agent.state import AgentRun, RunStatus
from harness.cli import main
from harness.llama_client import ChatResponse
from harness.storage.artifacts import (
    MAX_VERIFICATION_LOG_BYTES,
    sanitize_manifest,
    write_run_artifacts,
)
from harness.storage.database import (
    SqliteRunRecorder,
    load_run_record,
)
from harness.storage.recorder import LoggingPolicy, RunRecorder, StorageError


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


def test_sqlite_run_recorder_creates_schema_and_reconstructs_run(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "traces.db"
    artifacts_dir = tmp_path / "artifacts"
    recorder = SqliteRunRecorder(db_path=db_path, artifacts_dir=artifacts_dir)
    assert isinstance(recorder, RunRecorder)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello world\n", encoding="utf-8")

    responses = [
        _response(
            {"kind": "tool", "name": "read_file", "arguments": {"path": "hello.txt"}},
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
        _response(
            {"kind": "complete", "summary": "Read file successfully"},
            prompt_tokens=35,
            completion_tokens=8,
            total_tokens=43,
        ),
    ]

    def fake_model(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return responses.pop(0)

    manifest = {"model": "test-model", "env": "pytest"}
    run = run_agent(
        "Read hello.txt",
        repo,
        model=fake_model,
        recorder=recorder,
        manifest=manifest,
    )

    assert run.status is RunStatus.COMPLETED

    record = load_run_record(db_path, run.run_id)
    run_row = record["run"]
    events = record["events"]

    assert run_row["run_id"] == run.run_id
    assert run_row["task"] == "Read hello.txt"
    assert run_row["status"] == "completed"
    assert run_row["stop_reason"] == "model_complete"
    assert run_row["benchmark_outcome"] == "not_evaluated"
    assert run_row["steps"] == run.steps
    assert run_row["model_calls"] == 2
    assert run_row["prompt_tokens"] == 55
    assert run_row["completion_tokens"] == 18
    assert run_row["total_tokens"] == 73
    assert run_row["artifacts_dir"] == str(artifacts_dir / run.run_id)

    # Check event sequence
    assert len(events) >= 4
    event_types = [e["event_type"] for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    for i, e in enumerate(events, start=1):
        assert e["sequence"] == i
        assert e["run_id"] == run.run_id

    # Verify disk artifacts
    run_artifact_dir = artifacts_dir / run.run_id
    assert (run_artifact_dir / "manifest.json").is_file()
    manifest_on_disk = json.loads(
        (run_artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_on_disk["run_id"] == run.run_id
    assert manifest_on_disk["config"]["model"] == "test-model"


def test_sqlite_handles_missing_usage_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "traces.db"
    recorder = SqliteRunRecorder(db_path=db_path)

    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_model(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return _response({"kind": "complete", "summary": "Done"})

    run = run_agent(
        "Simple task",
        repo,
        model=fake_model,
        recorder=recorder,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.total_tokens is None

    record = load_run_record(db_path, run.run_id)
    run_row = record["run"]
    assert run_row["prompt_tokens"] is None
    assert run_row["completion_tokens"] is None
    assert run_row["total_tokens"] is None


def test_multi_run_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "traces.db"
    artifacts_dir = tmp_path / "artifacts"
    recorder = SqliteRunRecorder(db_path=db_path, artifacts_dir=artifacts_dir)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "f.txt").write_text("data\n", encoding="utf-8")

    def model_1(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return _response({"kind": "complete", "summary": "First run answer"})

    def model_2(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return _response({"kind": "complete", "summary": "Second run answer"})

    run1 = run_agent("Task 1", repo, model=model_1, recorder=recorder)
    run2 = run_agent("Task 2", repo, model=model_2, recorder=recorder)

    assert run1.run_id != run2.run_id

    # Check separate directories
    dir1 = artifacts_dir / run1.run_id
    dir2 = artifacts_dir / run2.run_id
    assert dir1.is_dir()
    assert dir2.is_dir()

    manifest1 = json.loads((dir1 / "manifest.json").read_text(encoding="utf-8"))
    manifest2 = json.loads((dir2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest1["task"] == "Task 1"
    assert manifest2["task"] == "Task 2"

    rec1 = load_run_record(db_path, run1.run_id)
    rec2 = load_run_record(db_path, run2.run_id)
    assert rec1["run"]["task"] == "Task 1"
    assert rec2["run"]["task"] == "Task 2"


def test_artifact_patch_and_verification_log(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    run = AgentRun(task="Fix bug", workspace=tmp_path)
    run.candidate_diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    run.test_invocations = [
        {
            "target": "tests/test_f.py",
            "exit_code": 0,
            "stdout": "1 passed in 0.05s",
            "stderr": "",
        }
    ]

    artifacts = write_run_artifacts(artifacts_dir / run.run_id, run)
    assert artifacts.patch_path is not None
    assert artifacts.patch_path.is_file()
    assert "old" in artifacts.patch_path.read_text(encoding="utf-8")

    assert artifacts.verification_log_path is not None
    assert artifacts.verification_log_path.is_file()
    assert "1 passed in 0.05s" in artifacts.verification_log_path.read_text(
        encoding="utf-8"
    )


def test_credential_sanitization(tmp_path: Path) -> None:
    raw = {
        "model": "qwen",
        "api_key": "sk-secret-key-123",
        "nested": {
            "auth_token": "bearer-xyz",
            "safe_param": "visible",
            "password_hash": "pwd123",
        },
        "credentials_list": ["token_val"],
    }
    sanitized = sanitize_manifest(raw)
    assert sanitized["model"] == "qwen"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["auth_token"] == "[REDACTED]"
    assert sanitized["nested"]["safe_param"] == "visible"
    assert sanitized["nested"]["password_hash"] == "[REDACTED]"


def test_degraded_logging_policy_on_sqlite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "traces.db"
    recorder = SqliteRunRecorder(db_path=db_path, policy=LoggingPolicy.DEGRADED)

    repo = tmp_path / "repo"
    repo.mkdir()

    # Simulate an error on SQLite execution by making get_connection raise
    def fail_conn() -> None:
        raise OSError("Disk quota exceeded")

    monkeypatch.setattr(recorder, "_get_connection", fail_conn)

    def fake_model(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return _response({"kind": "complete", "summary": "Finished despite db error"})

    # Run should NOT raise with DEGRADED policy
    run = run_agent(
        "Tolerate storage failure", repo, model=fake_model, recorder=recorder
    )
    assert run.status is RunStatus.COMPLETED
    assert recorder.is_degraded is True
    assert any("sqlite_logging_degraded" in issue for issue in run.unresolved_issues)


def test_fail_logging_policy_on_sqlite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "traces.db"
    recorder = SqliteRunRecorder(db_path=db_path, policy=LoggingPolicy.FAIL)

    repo = tmp_path / "repo"
    repo.mkdir()

    def fail_conn() -> None:
        raise OSError("Read-only filesystem")

    monkeypatch.setattr(recorder, "_get_connection", fail_conn)

    def fake_model(prompt: str, *, timeout: float = 120.0) -> ChatResponse:
        return _response({"kind": "complete", "summary": "Done"})

    # Run SHOULD raise StorageError with FAIL policy
    with pytest.raises(StorageError):
        run_agent("Fail on storage failure", repo, model=fake_model, recorder=recorder)


def test_large_verification_log_truncated(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    run = AgentRun(task="Huge logs", workspace=tmp_path)
    run.test_invocations = [
        {
            "target": "tests/",
            "exit_code": 1,
            "stdout": "A" * (MAX_VERIFICATION_LOG_BYTES + 500),
            "stderr": "",
        }
    ]
    artifacts = write_run_artifacts(artifacts_dir / run.run_id, run)
    assert artifacts.verification_log_path is not None
    content = artifacts.verification_log_path.read_text(encoding="utf-8")
    assert "[TRUNCATED]" in content
    assert len(content.encode("utf-8")) <= MAX_VERIFICATION_LOG_BYTES + 100


def test_cli_run_creates_traces_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
    db_file = tmp_path / "custom.db"

    responses = [
        _response(
            {"kind": "complete", "summary": "Found variable x"},
            prompt_tokens=15,
            completion_tokens=5,
            total_tokens=20,
        )
    ]

    def fake_chat(prompt: str, **kwargs: object) -> ChatResponse:
        return responses.pop(0)

    monkeypatch.setattr("harness.cli.chat", fake_chat)

    main(["run", "Find variable", "--repo", str(repo), "--db-path", str(db_file)])

    captured = capsys.readouterr()
    assert "Found variable x" in captured.out
    assert f"harness: db={db_file}" in captured.err

    assert db_file.is_file()
    # Ensure run record can be loaded
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM runs")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1


def test_cli_run_with_no_db_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    db_file = tmp_path / "never_created.db"

    responses = [_response({"kind": "complete", "summary": "Done"})]

    def fake_chat(prompt: str, **kwargs: object) -> ChatResponse:
        return responses.pop(0)

    monkeypatch.setattr("harness.cli.chat", fake_chat)

    main(["run", "Task", "--repo", str(repo), "--db-path", str(db_file), "--no-db"])

    captured = capsys.readouterr()
    assert "Done" in captured.out
    assert "harness: db=" not in captured.err
    assert not db_file.exists()
