"""Deterministic tests for Issue #41 external harness comparison (mini-SWE-agent)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.evaluation import (
    OracleTier,
    TaskManifest,
    compare_with_external_harness,
    default_mini_swe_agent_spec,
    evaluate_external_task,
    evaluate_suite,
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


def test_shared_fixture_evaluated_by_same_oracle_and_annotated_external(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_repo(repo)
    pricing_hash = _sha256((repo / "pricing.py").read_text(encoding="utf-8"))

    manifest = TaskManifest(
        task_id="local__ext-01",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount in pricing.py.",
        oracle_tests=("test_pricing.py",),
        agent_test_targets=("test_pricing.py",),
        oracle_tier=OracleTier.LOCAL_TARGETED,
    )

    def internal_model(prompt: str, *, timeout: float) -> ChatResponse:
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
                )
            )
        if '"tool":"run_tests"' not in prompt:
            return ChatResponse(
                json.dumps(
                    {
                        "kind": "tool",
                        "name": "run_tests",
                        "arguments": {"preset": "pytest", "target": "test_pricing.py"},
                    }
                )
            )
        return ChatResponse(json.dumps({"kind": "complete", "summary": "Done"}))

    internal_summary = evaluate_suite(
        [manifest],
        work_root=tmp_path / "int_work",
        model=internal_model,
        collect_memory=False,
    )

    def fake_external_runner(
        _manifest: TaskManifest, workspace: Path, _timeout: float
    ) -> tuple[int, str]:
        target = workspace / "pricing.py"
        target.write_text(
            target.read_text(encoding="utf-8").replace("+", "-"),
            encoding="utf-8",
        )
        return 0, "patched"

    spec = default_mini_swe_agent_spec()
    artifacts = tmp_path / "ext_artifacts"
    report = compare_with_external_harness(
        [manifest],
        internal_label="baseline",
        internal_summary=internal_summary,
        external_spec=spec,
        work_root=tmp_path / "ext_work",
        artifacts_root=artifacts,
        runner=fake_external_runner,
        limitations=("bash permissions differ from ToolPolicy",),
    )

    assert report.internal_summary.resolved_count == 1
    assert report.external_summary.resolved_count == 1
    md = (artifacts / "external_comparison.md").read_text(encoding="utf-8")
    assert (
        "External harness comparison (not a controlled retrieval/context ablation)"
        in md
    )
    assert "Unmatched parameters:" in md
    payload = json.loads(
        (artifacts / "external_comparison.json").read_text(encoding="utf-8")
    )
    assert payload["comparison_kind"] == "external_harness_comparison"
    assert payload["external_spec"]["pinned_version"] == "2.4.6"


def test_missing_external_binary_is_recorded_as_setup_error(tmp_path: Path) -> None:
    repo = tmp_path / "source_repo"
    commit = _init_repo(repo)
    manifest = TaskManifest(
        task_id="local__ext-missing",
        repo=str(repo),
        base_commit=commit,
        issue_text="Fix compute_discount.",
        oracle_tests=("test_pricing.py",),
    )
    res = evaluate_external_task(
        manifest,
        work_root=tmp_path / "work",
        spec=default_mini_swe_agent_spec(),
        command_template=("definitely-nonexistent-mini-swe-binary", "--task", "{task}"),
    )
    assert res.outcome is BenchmarkOutcome.SETUP_ERROR
    assert "not found" in (res.error_detail or "").lower()
