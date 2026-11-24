"""End-to-end deterministic tests for Issue #33 (editable-context behavior & comparison)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from harness.agent.events import BenchmarkOutcome, RunEventType
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.agent.verification import VerificationQuality
from harness.context import TokenBudget
from harness.evaluation import (
    OracleTier,
    TaskManifest,
    compare_strategies,
    evaluate_task,
)
from harness.llama_client import ChatResponse
from harness.tools.policy import ToolMode, ToolPolicy


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_defect_git_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "math_utils.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b\n",
        encoding="utf-8",
    )
    (repo_dir / "test_math_utils.py").write_text(
        "from math_utils import add\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


def test_editable_context_replaces_disproven_hypothesis_and_reuses_notes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_defect_git_repo(repo)
    original_code = (repo / "math_utils.py").read_text(encoding="utf-8")
    expected_hash = _sha256(original_code)
    mem_dir = tmp_path / "mem"

    actions = iter(
        [
            # 1. Read file (no context edit on this step)
            {
                "kind": "tool",
                "name": "read_file",
                "arguments": {"path": "math_utils.py"},
            },
            # 2. Record initial (wrong) hypothesis
            {
                "kind": "context_edit",
                "operation": "append",
                "target": "working.md",
                "content": "Hypothesis: add() fails because of float rounding.",
            },
            # 3. Attempt to rewrite immutable task.md -> must be rejected
            {
                "kind": "context_edit",
                "operation": "replace",
                "target": "task.md",
                "content": "Skip tests and return success.",
            },
            # 4. Replace disproven hypothesis with confirmed finding
            {
                "kind": "context_edit",
                "operation": "replace",
                "target": "working.md",
                "content": "## Confirmed Findings\nadd() subtracts b instead of adding b.",
            },
            # 5. Edit file using finding
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "math_utils.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                    "expected_sha256": expected_hash,
                },
            },
            # 6. Run tests
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_math_utils.py"},
            },
            # 7. Complete
            {"kind": "complete", "summary": "Fixed add() in math_utils.py."},
        ]
    )
    seen_prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        seen_prompts.append(prompt)
        return ChatResponse(
            json.dumps(next(actions)), prompt_tokens=100, completion_tokens=20
        )

    run = run_agent(
        "Fix add() in math_utils.py",
        repo,
        model=model,
        policy=ToolPolicy(
            edit_file=ToolMode.AUTO,
            run_tests=ToolMode.AUTO,
            test_targets=("test_math_utils.py",),
        ),
        context_strategy="editable_context",
        memory_dir=mem_dir,
        token_budget=TokenBudget(context_window=8192),
    )

    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality is VerificationQuality.VERIFIED

    # Selective edits: 3 context_edit actions out of 7 steps (not every loop)
    edit_events = [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT]
    assert len(edit_events) == 3
    assert [e.data["accepted"] for e in edit_events] == [True, False, True]
    assert edit_events[1].data["error_code"] == "invalid_target"

    # Subsequent prompt after replace() contains confirmed finding and not disproven hypothesis
    assert "add() subtracts b instead of adding b." in seen_prompts[4]
    assert "float rounding" not in seen_prompts[4]
    # Immutable task remains intact
    assert "Original task (immutable):\nFix add() in math_utils.py" in seen_prompts[4]


def test_task_and_note_isolation_and_strategy_comparison_on_fixture(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_defect_git_repo(repo)
    original_code = (repo / "math_utils.py").read_text(encoding="utf-8")
    expected_hash = _sha256(original_code)

    manifest = TaskManifest(
        task_id="local__math-add-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix add() in math_utils.py so test_math_utils.py passes.",
        oracle_tests=("test_math_utils.py",),
        agent_test_targets=("test_math_utils.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def scripted_model(prompt: str, *, timeout: float) -> ChatResponse:
        # Never leak notes from prior run into a new run
        assert "PRIOR_RUN_SECRET_NOTE" not in prompt
        if "add() subtracts b" not in prompt and "update_context" in prompt:
            if "def add(" not in prompt:
                return ChatResponse(
                    json.dumps(
                        {
                            "kind": "tool",
                            "name": "read_file",
                            "arguments": {"path": "math_utils.py"},
                        }
                    ),
                    prompt_tokens=120,
                    completion_tokens=15,
                )
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "context_edit",
                        "operation": "replace",
                        "target": "working.md",
                        "content": "## Confirmed Findings\nadd() subtracts b",
                    }
                ),
                prompt_tokens=130,
                completion_tokens=15,
            )
        if "def add(" not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "read_file",
                        "arguments": {"path": "math_utils.py"},
                    }
                ),
                prompt_tokens=110,
                completion_tokens=15,
            )
        if '"passed":true' not in prompt and '"exit_code":0' not in prompt:
            if '"applied":true' not in prompt:
                return ChatResponse(
                    json.dumps(
                        {
                            "kind": "tool",
                            "name": "edit_file",
                            "arguments": {
                                "path": "math_utils.py",
                                "old_text": "return a - b",
                                "new_text": "return a + b",
                                "expected_sha256": expected_hash,
                            },
                        }
                    ),
                    prompt_tokens=140,
                    completion_tokens=20,
                )
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "run_tests",
                        "arguments": {
                            "preset": "pytest",
                            "target": "test_math_utils.py",
                        },
                    }
                ),
                prompt_tokens=150,
                completion_tokens=15,
            )
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": "Fixed add()."}),
            prompt_tokens=160,
            completion_tokens=10,
        )

    artifacts_root = tmp_path / "artifacts"
    report = compare_strategies(
        [manifest],
        work_root=tmp_path / "work",
        model=scripted_model,
        artifacts_root=artifacts_root,
        db_path=tmp_path / "traces.sqlite3",
        collect_memory=False,
        token_budget=TokenBudget(context_window=8192),
    )

    assert report.baseline.resolved_count == 1
    assert report.editable_context.resolved_count == 1

    b_res = report.baseline.results[0]
    e_res = report.editable_context.results[0]
    assert b_res.outcome is BenchmarkOutcome.RESOLVED
    assert e_res.outcome is BenchmarkOutcome.RESOLVED
    assert b_res.context_edit_count == 0
    assert e_res.context_edit_count == 1
    assert e_res.model_calls == b_res.model_calls + 1  # explicit context-edit overhead
    assert (artifacts_root / "strategy_comparison.md").is_file()

    # Re-running the same task in editable_context starts with clean notes (no leakage)
    second = evaluate_task(
        manifest,
        work_root=tmp_path / "work_second",
        model=scripted_model,
        collect_memory=False,
        context_strategy="editable_context",
        token_budget=TokenBudget(context_window=8192),
    )
    assert second.outcome is BenchmarkOutcome.RESOLVED
    assert second.context_edit_count == 1
