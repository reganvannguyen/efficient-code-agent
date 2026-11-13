"""Tests for pinned experiment configuration and macOS memory observation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.evaluation import (
    ExperimentConfig,
    ModelSpec,
    TaskManifest,
    compute_file_sha256,
    evaluate_task,
    parse_memory_pressure,
    parse_ps_rss_kib,
    parse_sysctl_swapusage,
    sample_macos_memory,
)
from harness.evaluation import resources as resources_mod
from harness.llama_client import ChatResponse


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def test_compute_file_sha256_and_experiment_config(tmp_path: Path) -> None:
    fake_gguf = tmp_path / "qwen2.5-coder-7b-q4_k_m.gguf"
    payload = b"gguf-header-and-weights"
    fake_gguf.write_bytes(payload)

    digest = compute_file_sha256(fake_gguf)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert compute_file_sha256(tmp_path / "missing.gguf") is None

    spec = ModelSpec(
        gguf_filename=fake_gguf.name,
        gguf_sha256=digest,
        parameter_size="7B",
        quantization="Q4_K_M",
        llama_cpp_version="b4000",
        server_args=("--ctx-size", "8192"),
        temperature=0.0,
        seed=42,
        tool_format="json_schema",
    )
    assert spec.context_window == 8192

    config = ExperimentConfig.capture(
        spec,
        background_notes="only terminal running",
    )
    serialized = config.to_dict()
    assert serialized["model"]["gguf_filename"] == fake_gguf.name
    assert serialized["model"]["gguf_sha256"] == digest
    assert serialized["model"]["context_window"] == 8192
    assert serialized["restart_policy"] == "single_model_sequential"
    assert serialized["background_notes"] == "only terminal running"


def test_macos_output_parsers_and_missing_values() -> None:
    used, total = parse_sysctl_swapusage(
        "vm.swapusage: total = 2048.00M  used = 512.50M  free = 1535.50M  (encrypted)"
    )
    assert total == 2048 * 1024 * 1024
    assert used == int(512.50 * 1024 * 1024)

    assert parse_sysctl_swapusage("unparseable") == (None, None)

    assert (
        parse_memory_pressure("The system is under normal memory pressure.") == "normal"
    )
    assert parse_memory_pressure("System memory pressure: WARN") == "warn"
    assert parse_memory_pressure("CRITICAL") == "critical"
    assert parse_memory_pressure("unknown") is None

    assert parse_ps_rss_kib("  123456\n") == 123456 * 1024
    assert parse_ps_rss_kib("") is None
    assert parse_ps_rss_kib("not-a-number") is None


def test_sample_macos_memory_never_estimates_unavailable_metrics(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(resources_mod, "_run_probe", lambda cmd, *, timeout: None)  # type: ignore[attr-defined]
    snap = sample_macos_memory(server_pid=99999)
    assert snap.swap_used_bytes is None
    assert snap.swap_total_bytes is None
    assert snap.memory_pressure_level is None
    assert snap.process_rss_bytes is None
    assert snap.notes is not None
    assert "unavailable" in snap.notes


def test_evaluate_task_records_experiment_config_and_memory_in_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import VALUE\n\n\ndef test_val() -> None:\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    base_commit = _git(repo, "rev-parse", "HEAD")

    app_sha = hashlib.sha256((repo / "app.py").read_bytes()).hexdigest()
    manifest = TaskManifest(
        task_id="cfg-001",
        repo=str(repo),
        base_commit=base_commit,
        issue_text="Set VALUE = 2 in app.py.",
        oracle_tests=("test_app.py",),
        agent_test_targets=("test_app.py",),
    )
    config = ExperimentConfig(
        model=ModelSpec(
            gguf_filename="test-model.gguf",
            gguf_sha256="abc123",
            quantization="Q4_K_M",
            context_window=8192,
        ),
        harness_commit="deadbeef",
    )

    actions = iter(
        [
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "app.py",
                    "expected_sha256": app_sha,
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "test_app.py"},
            },
            {"kind": "complete", "summary": "Updated VALUE to 2."},
        ]
    )

    result = evaluate_task(
        manifest,
        work_root=tmp_path / "work",
        model=lambda _p, *, timeout: ChatResponse(json.dumps(next(actions))),
        artifacts_root=tmp_path / "artifacts",
        experiment_config=config,
    )
    assert result.outcome == BenchmarkOutcome.RESOLVED
    assert result.experiment_config == config
    assert result.memory_before is not None
    assert result.memory_after is not None
    assert result.artifacts is not None

    saved_manifest = json.loads(
        result.artifacts.manifest_path.read_text(encoding="utf-8")
    )
    saved_config = saved_manifest["config"]
    assert saved_config["experiment_config"]["model"]["gguf_filename"] == (
        "test-model.gguf"
    )
    assert "memory_before" in saved_config["resources"]
    assert "memory_after" in saved_config["resources"]
