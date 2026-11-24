"""Tests for patch re-validation, traceability, and final benchmark reporting (Issue #43)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.agent.events import BenchmarkOutcome
from harness.evaluation import (
    FourWayAblationReport,
    FrozenEvaluationProtocol,
    OracleTier,
    SuiteSummary,
    TaskManifest,
    aggregate_trial_reports,
    build_final_benchmark_report,
    recheck_saved_patch,
    validate_report_traceability,
)
from harness.evaluation.runner import EvaluationResult


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


def test_recheck_saved_patch_validates_good_and_rejects_empty_patch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_repo(repo)
    manifest = TaskManifest(
        task_id="local__recheck-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount.",
        oracle_tests=("test_pricing.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    empty = recheck_saved_patch(manifest, "", work_root=tmp_path / "work_empty")
    assert empty.outcome is BenchmarkOutcome.UNRESOLVED

    good_patch = (
        "diff --git a/pricing.py b/pricing.py\n"
        "--- a/pricing.py\n"
        "+++ b/pricing.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def compute_discount(price: float, pct: float) -> float:\n"
        "-    return round(price + (price * pct), 2)\n"
        "+    return round(price - (price * pct), 2)\n"
    )
    rechecked = recheck_saved_patch(
        manifest, good_patch, work_root=tmp_path / "work_good"
    )
    assert rechecked.outcome is BenchmarkOutcome.RESOLVED
    assert rechecked.oracle_passed is True


def test_final_report_traces_counts_and_separates_tiers() -> None:
    protocol = FrozenEvaluationProtocol.from_json_file(Path("benchmarks/protocol.json"))
    r_ok = EvaluationResult(
        task_id="t1",
        run_id="run1",
        outcome=BenchmarkOutcome.RESOLVED,
        oracle_tier=OracleTier.LOCAL_TARGETED,
        oracle_passed=True,
        candidate_patch="diff",
        model_calls=5,
        prompt_tokens=500,
    )
    r_fail = EvaluationResult(
        task_id="t1",
        run_id="run2",
        outcome=BenchmarkOutcome.UNRESOLVED,
        oracle_tier=OracleTier.LOCAL_TARGETED,
        oracle_passed=False,
        candidate_patch="",
        model_calls=6,
        prompt_tokens=600,
    )
    rep = FourWayAblationReport(
        suite_digest="digest123",
        conditions={
            "baseline": SuiteSummary.from_results([r_fail]),
            "retrieval_only": SuiteSummary.from_results([r_ok]),
            "clm_only": SuiteSummary.from_results([r_ok]),
            "combined": SuiteSummary.from_results([r_ok]),
        },
        protocol=protocol,
    )
    analysis = aggregate_trial_reports([rep], seeds=[0])
    tiers = validate_report_traceability(analysis, [rep])
    assert tiers["baseline"] == {"local_targeted": 1}
    assert tiers["combined"] == {"local_targeted": 1}

    md = build_final_benchmark_report(
        analysis,
        protocol=protocol,
        reports=[rep],
        exclusions=["sympy__sympy-12096: setup timeout"],
    )
    assert "not** official SWE-bench Verified scores" in md
    assert "Oracle tier breakdown" in md
    assert "sympy__sympy-12096" in md

    with pytest.raises(ValueError, match="Expected 1 trial reports"):
        validate_report_traceability(analysis, [])
