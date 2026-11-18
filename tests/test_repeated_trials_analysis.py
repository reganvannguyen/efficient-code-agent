"""Tests for repeated trials, paired task comparisons, and uncertainty reporting (Issue #42)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from harness.evaluation import (
    FourWayAblationReport,
    OracleTier,
    SuiteSummary,
    TaskManifest,
    aggregate_trial_reports,
    run_repeated_ablations,
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


def test_repeated_ablations_aggregates_metrics_paired_deltas_and_uncertainty(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_repo(repo)
    pricing_hash = _sha256((repo / "pricing.py").read_text(encoding="utf-8"))

    manifest = TaskManifest(
        task_id="local__trial-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount in pricing.py.",
        oracle_tests=("test_pricing.py",),
        agent_test_targets=("test_pricing.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def model(prompt: str, *, timeout: float) -> ChatResponse:
        # Make retrieval_only & combined succeed, while baseline & clm_only fail to patch
        if (
            "exact_symbol_definition" in prompt
            or "code_aware" in prompt
            or "pricing.py:1-2" in prompt
        ):
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
                    completion_tokens=15,
                )
            if '"tool":"run_tests"' not in prompt:
                return ChatResponse(
                    json.dumps(
                        {
                            "kind": "tool",
                            "name": "run_tests",
                            "arguments": {
                                "preset": "pytest",
                                "target": "test_pricing.py",
                            },
                        }
                    ),
                    prompt_tokens=120,
                    completion_tokens=15,
                )
            return ChatResponse(
                json.dumps({"kind": "complete", "summary": "Fixed."}),
                prompt_tokens=130,
                completion_tokens=10,
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
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": "No changes."}),
            prompt_tokens=100,
            completion_tokens=10,
        )

    artifacts = tmp_path / "artifacts"
    reports, analysis = run_repeated_ablations(
        [manifest],
        trials=2,
        seeds=(11, 22),
        work_root=tmp_path / "work",
        model=model,
        artifacts_root=artifacts,
        collect_memory=False,
    )

    assert len(reports) == 2
    assert analysis.trials == 2
    assert analysis.seeds == (11, 22)
    assert analysis.per_task_pp_step == 100.0

    assert analysis.conditions["baseline"].resolved_runs == 0
    assert analysis.conditions["retrieval_only"].resolved_runs == 2
    assert analysis.conditions["retrieval_only"].resolution_rate_total == 1.0
    assert analysis.conditions["retrieval_only"].mean_rss_after_mib is None

    assert analysis.paired_vs_baseline["retrieval_only"].improved_tasks == (
        "local__trial-01",
    )
    assert analysis.paired_vs_baseline["clm_only"].unchanged_unresolved == (
        "local__trial-01",
    )

    md = (artifacts / "multi_trial_analysis.md").read_text(encoding="utf-8")
    assert (
        "Fewer prompt tokens do not automatically reduce allocated KV-cache RAM" in md
    )
    assert (artifacts / "multi_trial_analysis.json").is_file()


def test_aggregate_trial_reports_rejects_mismatched_suite_digest() -> None:
    empty = SuiteSummary.from_results([])
    r1 = FourWayAblationReport(suite_digest="aaa", conditions={"baseline": empty})
    r2 = FourWayAblationReport(suite_digest="bbb", conditions={"baseline": empty})
    with pytest.raises(ValueError, match="suite_digest"):
        aggregate_trial_reports([r1, r2])
