"""Deterministic tests for controlled four-way internal ablations (Issue #40)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.agent.state import RunStatus
from harness.evaluation import (
    FOUR_WAY_CONDITIONS,
    FrozenEvaluationProtocol,
    OracleTier,
    TaskManifest,
    run_four_way_ablation,
)
from harness.llama_client import ChatResponse


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
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


def test_four_way_ablation_isolates_axes_and_records_all_metrics(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_repo(repo)
    pricing_hash = _sha256((repo / "pricing.py").read_text(encoding="utf-8"))

    manifest = TaskManifest(
        task_id="local__ablation-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount in pricing.py.",
        oracle_tests=("test_pricing.py",),
        agent_test_targets=("test_pricing.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def scripted_model(prompt: str, *, timeout: float) -> ChatResponse:
        # Never leak notes from prior runs
        assert "LEAKED_NOTE" not in prompt
        if "update_context" in prompt and "compute_discount subtracts" not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "context_edit",
                        "operation": "replace",
                        "target": "working.md",
                        "content": "## Confirmed Findings\ncompute_discount subtracts pct",
                    }
                ),
                prompt_tokens=90,
                completion_tokens=15,
            )
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
        if '"tool":"edit_file","ok":true' not in prompt:
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
                prompt_tokens=110,
                completion_tokens=20,
            )
        if '"tool":"run_tests"' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "run_tests",
                        "arguments": {"preset": "pytest", "target": "test_pricing.py"},
                    }
                ),
                prompt_tokens=120,
                completion_tokens=15,
            )
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": "Fixed compute_discount."}),
            prompt_tokens=130,
            completion_tokens=10,
        )

    protocol = FrozenEvaluationProtocol.from_json_file(Path("benchmarks/protocol.json"))
    artifacts_root = tmp_path / "artifacts"
    report = run_four_way_ablation(
        [manifest],
        work_root=tmp_path / "work",
        model=scripted_model,
        artifacts_root=artifacts_root,
        db_path=tmp_path / "ablation.sqlite3",
        protocol=protocol,
        collect_memory=False,
    )

    assert tuple(report.conditions) == tuple(c.name for c in FOUR_WAY_CONDITIONS)
    for cond in FOUR_WAY_CONDITIONS:
        res = report.conditions[cond.name].results[0]
        assert res.task_id == "local__ablation-01"
        assert res.outcome is BenchmarkOutcome.RESOLVED

    assert report.conditions["baseline"].results[0].context_edit_count == 0
    assert report.conditions["baseline"].results[0].retrieval_parse_count == 0

    assert report.conditions["retrieval_only"].results[0].context_edit_count == 0
    assert report.conditions["retrieval_only"].results[0].retrieval_parse_count >= 1

    assert report.conditions["clm_only"].results[0].context_edit_count == 1
    assert report.conditions["clm_only"].results[0].retrieval_parse_count == 0

    assert report.conditions["combined"].results[0].context_edit_count == 1
    assert report.conditions["combined"].results[0].retrieval_parse_count >= 1

    assert (artifacts_root / "four_way_ablation.json").is_file()
    md = (artifacts_root / "four_way_ablation.md").read_text(encoding="utf-8")
    assert "`baseline`" in md
    assert "`retrieval_only`" in md
    assert "`clm_only`" in md
    assert "`combined`" in md


def test_completed_runtime_without_oracle_pass_is_unresolved(tmp_path: Path) -> None:
    repo = tmp_path / "source_repo"
    _init_repo(repo)
    (repo / "test_smoke.py").write_text(
        "from pricing import compute_discount\n\n"
        "def test_smoke() -> None:\n"
        "    assert isinstance(compute_discount(100.0, 0.2), float)\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "test_smoke.py"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add smoke test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pricing_hash = _sha256((repo / "pricing.py").read_text(encoding="utf-8"))

    manifest = TaskManifest(
        task_id="local__bad-fix-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount in pricing.py.",
        oracle_tests=("test_pricing.py",),
        agent_test_targets=("test_smoke.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def bad_patch_model(prompt: str, *, timeout: float) -> ChatResponse:
        if '"tool":"edit_file","ok":true' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "edit_file",
                        "arguments": {
                            "path": "pricing.py",
                            "old_text": "return round(price + (price * pct), 2)",
                            "new_text": "return round(price * 999, 2)",
                            "expected_sha256": pricing_hash,
                        },
                    }
                )
            )
        if '"tool":"run_tests"' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "run_tests",
                        "arguments": {"preset": "pytest", "target": "test_smoke.py"},
                    }
                )
            )
        return ChatResponse(json.dumps({"kind": "complete", "summary": "Done"}))

    report = run_four_way_ablation(
        [manifest],
        work_root=tmp_path / "work",
        model=bad_patch_model,
        collect_memory=False,
    )
    for cond in FOUR_WAY_CONDITIONS:
        res = report.conditions[cond.name].results[0]
        assert res.agent_status is RunStatus.COMPLETED
        assert res.outcome is BenchmarkOutcome.UNRESOLVED
        assert res.oracle_passed is False
