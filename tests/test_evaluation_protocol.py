"""Tests for frozen Phase 7 evaluation protocol, suite isolation, and prompt hygiene."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.evaluation import (
    FrozenEvaluationProtocol,
    ManifestError,
    OracleTier,
    TaskManifest,
    compute_suite_digest,
    load_suite_dir,
    verify_agent_prompt_isolation,
    verify_suite_partition_isolation,
)


def test_frozen_protocol_loads_and_validates_pinned_json() -> None:
    protocol_path = Path("benchmarks/protocol.json")
    protocol = FrozenEvaluationProtocol.from_json_file(protocol_path)
    assert protocol.protocol_version == "phase7-v1"
    assert protocol.gguf_filename == "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    assert protocol.context_window == 8192
    assert protocol.max_model_calls == 30
    assert protocol.max_steps == 30
    assert protocol.max_consecutive_invalid_actions == 6
    assert protocol.cleanup_workspaces is True


def test_smoke_manifests_are_isolated_and_deterministic() -> None:
    smoke = load_suite_dir(Path("benchmarks/smoke"))
    assert len(smoke) == 5
    for manifest in smoke:
        verify_agent_prompt_isolation(manifest)
        assert manifest.oracle_tier is OracleTier.LOCAL_TARGETED

    digests_a = verify_suite_partition_isolation(smoke=smoke)
    digests_b = verify_suite_partition_isolation(smoke=list(reversed(smoke)))
    assert digests_a["smoke_digest"] == digests_b["smoke_digest"]
    assert compute_suite_digest(smoke) == digests_a["smoke_digest"]


def test_partition_isolation_rejects_overlap_and_prompt_leakage() -> None:
    m1 = TaskManifest(
        task_id="t1",
        repo="https://example.com/repo.git",
        base_commit="abc1234",
        issue_text="Fix bug",
        oracle_tests=("tests/test_a.py",),
        reference_patch="SECRET_PATCH",
    )
    with pytest.raises(ManifestError, match="appears in both"):
        verify_suite_partition_isolation(dev=[m1], held_out=[m1])

    leaked = TaskManifest(
        task_id="t2",
        repo="https://example.com/repo.git",
        base_commit="abc1234",
        issue_text="Fix bug SECRET_PATCH",
        oracle_tests=("tests/test_a.py",),
        reference_patch="SECRET_PATCH",
    )
    with pytest.raises(ManifestError, match="leaked reference_patch"):
        verify_agent_prompt_isolation(leaked)
