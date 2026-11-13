"""Deterministic tests for clean-checkout fixture evaluation and independent oracle."""

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
    evaluate_task,
    prepare_clean_checkout,
)
from harness.llama_client import ChatResponse


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "upstream_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    (repo / "math_ops.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b\n",
        encoding="utf-8",
    )
    (repo / "test_math_ops.py").write_text(
        "from math_ops import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial broken fixture")
    base_commit = _git(repo, "rev-parse", "HEAD")
    return repo, base_commit


def _response(payload: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(payload))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_validation_and_private_metadata_isolation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "task.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": "demo-001",
                "repo": "/tmp/repo",
                "base_commit": "abc123",
                "issue_text": "Fix add() in math_ops.py.",
                "oracle_tests": ["test_math_ops.py"],
                "oracle_tier": "local_targeted",
                "reference_patch": "SECRET REFERENCE PATCH",
                "metadata": {"private_note": "do not leak"},
            }
        ),
        encoding="utf-8",
    )
    manifest = TaskManifest.from_json_file(manifest_path)
    assert manifest.task_id == "demo-001"
    assert manifest.oracle_tier == OracleTier.LOCAL_TARGETED
    assert manifest.agent_prompt() == "Fix add() in math_ops.py."
    assert "SECRET REFERENCE PATCH" not in manifest.agent_prompt()
    assert "do not leak" not in manifest.agent_prompt()

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"task_id": "only-id"}), encoding="utf-8")
    with pytest.raises(ManifestError):
        TaskManifest.from_json_file(bad_path)


def test_same_manifest_recreates_identical_clean_snapshot(tmp_path: Path) -> None:
    repo, base_commit = _init_fixture_repo(tmp_path)
    snap_a = prepare_clean_checkout(repo.as_posix(), base_commit, tmp_path / "snap_a")
    (snap_a / "math_ops.py").write_text("# dirty edit\n", encoding="utf-8")
    (snap_a / "notes.md").write_text("stale working note\n", encoding="utf-8")

    snap_b = prepare_clean_checkout(repo.as_posix(), base_commit, tmp_path / "snap_a")
    assert not (snap_b / "notes.md").exists()
    assert _file_sha(snap_b / "math_ops.py") == _file_sha(repo / "math_ops.py")
    assert _file_sha(snap_b / "test_math_ops.py") == _file_sha(
        repo / "test_math_ops.py"
    )


def test_valid_patch_is_graded_in_separate_checkout_and_resolves(
    tmp_path: Path,
) -> None:
    repo, base_commit = _init_fixture_repo(tmp_path)
    math_sha = _file_sha(repo / "math_ops.py")
    manifest = TaskManifest(
        task_id="fix-add-001",
        repo=str(repo),
        base_commit=base_commit,
        issue_text="Fix add(a, b) in math_ops.py so it returns a + b.",
        oracle_tests=("test_math_ops.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
        agent_test_targets=("test_math_ops.py",),
        reference_patch="SECRET PATCH",
    )

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "math_ops.py",
                    "expected_sha256": math_sha,
                    "old_text": "    return a - b",
                    "new_text": "    return a + b",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_math_ops.py"},
            },
            {
                "kind": "complete",
                "summary": "Updated add() to return a + b.",
            },
        ]
    )

    def fake_model(prompt: str, *, timeout: float) -> ChatResponse:
        assert "SECRET PATCH" not in prompt
        return _response(next(actions))

    result = evaluate_task(
        manifest,
        work_root=tmp_path / "work",
        model=fake_model,
        artifacts_root=tmp_path / "artifacts",
    )
    assert result.outcome == BenchmarkOutcome.RESOLVED
    assert result.oracle_tier == OracleTier.LOCAL_TARGETED
    assert result.oracle_passed is True
    assert "+    return a + b" in result.candidate_patch
    assert result.artifacts is not None
    assert result.artifacts.patch_path is not None
    assert result.artifacts.patch_path.exists()
    assert result.artifacts.manifest_path.exists()


def test_agent_cannot_cheat_by_editing_oracle_tests(tmp_path: Path) -> None:
    repo, base_commit = _init_fixture_repo(tmp_path)
    test_sha = _file_sha(repo / "test_math_ops.py")
    manifest = TaskManifest(
        task_id="cheat-test-001",
        repo=str(repo),
        base_commit=base_commit,
        issue_text="Fix add(a, b) in math_ops.py.",
        oracle_tests=("test_math_ops.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
        agent_test_targets=("test_math_ops.py",),
    )

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "test_math_ops.py",
                    "expected_sha256": test_sha,
                    "old_text": "    assert add(2, 3) == 5",
                    "new_text": "    assert True",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_math_ops.py"},
            },
            {
                "kind": "complete",
                "summary": "All tests pass now.",
            },
        ]
    )

    def fake_model(prompt: str, *, timeout: float) -> ChatResponse:
        return _response(next(actions))

    result = evaluate_task(
        manifest,
        work_root=tmp_path / "work",
        model=fake_model,
    )
    assert result.outcome == BenchmarkOutcome.UNRESOLVED
    assert result.oracle_passed is False
    assert result.oracle_tier == OracleTier.LOCAL_TARGETED


def test_setup_failure_is_reported_as_setup_error(tmp_path: Path) -> None:
    repo, _ = _init_fixture_repo(tmp_path)
    bad_commit_manifest = TaskManifest(
        task_id="bad-commit-001",
        repo=str(repo),
        base_commit="deadbeef00000000000000000000000000000000",
        issue_text="Fix add().",
        oracle_tests=("test_math_ops.py",),
    )

    def unused_model(prompt: str, *, timeout: float) -> ChatResponse:
        raise AssertionError("Model must not be called when setup fails.")

    result = evaluate_task(
        bad_commit_manifest,
        work_root=tmp_path / "work",
        model=unused_model,
    )
    assert result.outcome == BenchmarkOutcome.SETUP_ERROR
    assert result.run_id is None
    assert result.oracle_passed is False


def test_reruns_do_not_leak_prior_edits_or_notes(tmp_path: Path) -> None:
    repo, base_commit = _init_fixture_repo(tmp_path)
    manifest = TaskManifest(
        task_id="rerun-isolation-001",
        repo=str(repo),
        base_commit=base_commit,
        issue_text="Fix add().",
        oracle_tests=("test_math_ops.py",),
        agent_test_targets=("test_math_ops.py",),
    )
    math_sha = _file_sha(repo / "math_ops.py")

    # Run 1: makes a wrong edit
    run1_actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "math_ops.py",
                    "expected_sha256": math_sha,
                    "old_text": "    return a - b",
                    "new_text": "    return 0",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_math_ops.py"},
            },
            {"kind": "complete", "summary": "done"},
        ]
    )
    res1 = evaluate_task(
        manifest,
        work_root=tmp_path / "work",
        model=lambda prompt, *, timeout: _response(next(run1_actions)),
        max_model_calls=3,
    )
    assert res1.outcome == BenchmarkOutcome.UNRESOLVED

    # Run 2: still sees the original `return a - b` (exact original SHA), not `return 0`
    run2_actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "math_ops.py",
                    "expected_sha256": math_sha,
                    "old_text": "    return a - b",
                    "new_text": "    return a + b",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_math_ops.py"},
            },
            {"kind": "complete", "summary": "done"},
        ]
    )
    res2 = evaluate_task(
        manifest,
        work_root=tmp_path / "work",
        model=lambda prompt, *, timeout: _response(next(run2_actions)),
    )
    assert res2.outcome == BenchmarkOutcome.RESOLVED
