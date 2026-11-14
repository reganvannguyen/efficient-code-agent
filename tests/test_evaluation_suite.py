"""Deterministic tests for sequential suite evaluation and summary reporting."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from harness.agent.events import BenchmarkOutcome
from harness.evaluation import (
    ManifestError,
    OracleTier,
    TaskManifest,
    evaluate_suite,
    load_suite_dir,
)
from harness.llama_client import ChatResponse
from harness.storage.database import load_run_record


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "calc.py").write_text(
        "def inc(x: int) -> int:\n    return x\n", encoding="utf-8"
    )
    (repo / "test_calc.py").write_text(
        "from calc import inc\n\n\ndef test_inc() -> None:\n    assert inc(1) == 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    commit = _git(repo, "rev-parse", "HEAD")
    sha = hashlib.sha256((repo / "calc.py").read_bytes()).hexdigest()
    return repo, commit, sha


def test_load_suite_dir_sorts_and_validates(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_suite_dir(tmp_path / "missing")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ManifestError):
        load_suite_dir(empty_dir)

    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    for tid in ("02-second", "01-first"):
        (suite_dir / f"{tid}.json").write_text(
            json.dumps(
                {
                    "task_id": tid,
                    "repo": "/tmp/repo",
                    "base_commit": "abc123",
                    "issue_text": f"Task {tid}",
                    "oracle_tests": ["test_calc.py"],
                }
            ),
            encoding="utf-8",
        )
    loaded = load_suite_dir(suite_dir)
    assert [m.task_id for m in loaded] == ["01-first", "02-second"]


def test_evaluate_suite_records_mixed_outcomes_sqlite_and_summary(
    tmp_path: Path,
) -> None:
    repo, commit, sha = _init_repo(tmp_path, "repo")

    t1 = TaskManifest(
        task_id="task-01-pass",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix inc(x) to return x + 1.",
        oracle_tests=("test_calc.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
        agent_test_targets=("test_calc.py",),
    )
    t2 = TaskManifest(
        task_id="task-02-fail",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix inc(x) to return x + 1.",
        oracle_tests=("test_calc.py",),
        oracle_tier=OracleTier.LOCAL_FULL,
        agent_test_targets=("test_calc.py",),
    )
    t3 = TaskManifest(
        task_id="task-03-setup-err",
        repo=str(repo),
        base_commit="deadbeef00000000000000000000000000000000",
        issue_text="Broken commit.",
        oracle_tests=("test_calc.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    responses = iter(
        [
            # Task 1: passes
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "calc.py",
                    "expected_sha256": sha,
                    "old_text": "    return x",
                    "new_text": "    return x + 1",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_calc.py"},
            },
            {"kind": "complete", "summary": "Fixed inc."},
            # Task 2: wrong edit -> unresolved
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "calc.py",
                    "expected_sha256": sha,
                    "old_text": "    return x",
                    "new_text": "    return x + 2",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_calc.py"},
            },
            {"kind": "complete", "summary": "Tried inc."},
        ]
    )

    artifacts_root = tmp_path / "artifacts"
    db_path = tmp_path / "traces.sqlite3"
    summary = evaluate_suite(
        [t1, t2, t3],
        work_root=tmp_path / "work",
        model=lambda _p, *, timeout: ChatResponse(json.dumps(next(responses))),
        artifacts_root=artifacts_root,
        db_path=db_path,
        max_model_calls=3,
    )

    assert summary.resolved_count == 1
    assert summary.outcome_counts == {
        BenchmarkOutcome.RESOLVED.value: 1,
        BenchmarkOutcome.SETUP_ERROR.value: 1,
        BenchmarkOutcome.UNRESOLVED.value: 1,
    }
    assert summary.oracle_tier_counts == {
        OracleTier.LOCAL_FULL.value: 1,
        OracleTier.LOCAL_TARGETED.value: 2,
    }

    assert (artifacts_root / "suite_summary.json").exists()
    md = (artifacts_root / "suite_summary.md").read_text(encoding="utf-8")
    assert "task-01-pass" in md
    assert "task-02-fail" in md
    assert "task-03-setup-err" in md

    # Verify SQLite recorded runs 1 and 2
    rec1 = load_run_record(db_path, summary.results[0].run_id or "")
    rec2 = load_run_record(db_path, summary.results[1].run_id or "")
    assert rec1 is not None
    assert rec2 is not None
