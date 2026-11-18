"""End-to-end deterministic tests for Issue #38 (code-aware retrieval & comparison)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from harness.agent.events import BenchmarkOutcome, RunEventType
from harness.agent.protocol import ToolAction
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.agent.verification import VerificationQuality
from harness.context import TokenBudget
from harness.evaluation import (
    OracleTier,
    TaskManifest,
    compare_retrieval_modes,
)
from harness.llama_client import ChatResponse
from harness.retrieval import CodeAwareRetrievalBackend
from harness.tools.dispatcher import execute_tool
from harness.tools.policy import ToolMode, ToolPolicy


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_multifile_git_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "notes.py").write_text(
        "# compute_discount is defined in pricing.py\n"
        "# TODO: verify compute_discount rounding\n",
        encoding="utf-8",
    )
    (repo_dir / "pricing.py").write_text(
        "def compute_discount(price: float, pct: float) -> float:\n"
        "    return round(price + (price * pct), 2)\n",
        encoding="utf-8",
    )
    (repo_dir / "test_pricing.py").write_text(
        "from pricing import compute_discount\n\n"
        "def test_compute_discount() -> None:\n"
        "    assert compute_discount(100.0, 0.2) == 80.0\n",
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
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_code_aware_agent_finds_definition_invalidates_on_edit_and_handles_syntax_errors(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_multifile_git_repo(repo)
    (repo / "broken.py").write_text(
        "def broken_fn(:\n    compute_discount(1, 0)\n", encoding="utf-8"
    )
    pricing_text = (repo / "pricing.py").read_text(encoding="utf-8")
    pricing_hash = _sha256(pricing_text)

    backend = CodeAwareRetrievalBackend()

    # Unparseable Python still permits useful lexical search in code_aware mode
    broken_search = execute_tool(
        ToolAction("search_code", {"query": "broken_fn"}),
        repo,
        retrieval_backend=backend,
    )
    assert broken_search.ok
    assert any(m["path"] == "broken.py" for m in broken_search.data["matches"])

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "search_code",
                "arguments": {"query": "compute_discount"},
            },
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "pricing.py",
                    "old_text": "return round(price + (price * pct), 2)",
                    "new_text": "return round(price - (price * pct), 2)",
                    "expected_sha256": pricing_hash,
                },
            },
            # Search again after edit_file: index must be invalidated and show new code
            {
                "kind": "tool",
                "name": "search_code",
                "arguments": {"query": "compute_discount"},
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_pricing.py"},
            },
            {"kind": "complete", "summary": "Fixed compute_discount in pricing.py."},
        ]
    )
    seen_prompts: list[str] = []

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        seen_prompts.append(prompt)
        return ChatResponse(
            json.dumps(next(actions)), prompt_tokens=100, completion_tokens=20
        )

    run = run_agent(
        "Fix compute_discount in pricing.py",
        repo,
        model=model,
        policy=ToolPolicy(
            edit_file=ToolMode.AUTO,
            run_tests=ToolMode.AUTO,
            test_targets=("test_pricing.py",),
        ),
        context_strategy="bounded_history",
        retrieval_backend=backend,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality is VerificationQuality.VERIFIED
    # Must leave CLM context editing disabled
    assert [e for e in run.events if e.event_type == RunEventType.CONTEXT_EDIT] == []
    # Post-edit search prompt reflects updated line 2
    assert "price - (price * pct)" in seen_prompts[3]

    # Deleting broken.py removes it from future search results
    (repo / "broken.py").unlink()
    after_delete = execute_tool(
        ToolAction("search_code", {"query": "broken_fn"}),
        repo,
        retrieval_backend=backend,
    )
    assert after_delete.ok
    assert after_delete.data["matches"] == []


def test_compare_retrieval_modes_holds_bounded_history_constant(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_multifile_git_repo(repo)
    pricing_hash = _sha256((repo / "pricing.py").read_text(encoding="utf-8"))

    manifest = TaskManifest(
        task_id="local__pricing-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount in pricing.py so test_pricing.py passes.",
        oracle_tests=("test_pricing.py",),
        agent_test_targets=("test_pricing.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def scripted_model(prompt: str, *, timeout: float) -> ChatResponse:
        assert "update_context" not in prompt
        if "compute_discount(price: float" not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "search_code",
                        "arguments": {"query": "compute_discount"},
                    }
                ),
                prompt_tokens=100,
                completion_tokens=15,
            )
        if '"applied":true' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "edit_file",
                        "arguments": {
                            "path": "pricing.py",
                            "old_text": "return round(price + (price * pct), 2)",
                            "new_text": "return round(price - (price * pct), 2)",
                            "expected_sha256": pricing_hash,
                        },
                    }
                ),
                prompt_tokens=120,
                completion_tokens=20,
            )
        if '"passed":true' not in prompt and '"exit_code":0' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "run_tests",
                        "arguments": {"preset": "pytest", "target": "test_pricing.py"},
                    }
                ),
                prompt_tokens=130,
                completion_tokens=15,
            )
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": "Fixed compute_discount."}),
            prompt_tokens=140,
            completion_tokens=10,
        )

    artifacts_root = tmp_path / "artifacts"
    report = compare_retrieval_modes(
        [manifest],
        work_root=tmp_path / "work",
        model=scripted_model,
        artifacts_root=artifacts_root,
        db_path=tmp_path / "traces.sqlite3",
        collect_memory=False,
        token_budget=TokenBudget(context_window=8192),
    )

    assert report.lexical_only.resolved_count == 1
    assert report.code_aware.resolved_count == 1
    l_res = report.lexical_only.results[0]
    c_res = report.code_aware.results[0]
    assert l_res.outcome is BenchmarkOutcome.RESOLVED
    assert c_res.outcome is BenchmarkOutcome.RESOLVED
    assert l_res.retrieval_parse_count == 0
    assert c_res.retrieval_parse_count >= 1
    assert (artifacts_root / "retrieval_comparison.md").is_file()
