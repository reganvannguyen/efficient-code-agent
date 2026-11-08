"""Run-scoped artifact persistence and sensitive key sanitization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.storage.recorder import RunArtifacts

if TYPE_CHECKING:
    from harness.agent.state import AgentRun

_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "token",
    "secret",
    "password",
    "auth",
    "credential",
)
MAX_VERIFICATION_LOG_BYTES = 1_048_576  # 1 MiB ceiling for test logs


def sanitize_value(key: str, value: Any) -> Any:
    """Mask string values associated with sensitive keys."""
    lower_key = key.lower()
    if any(sub in lower_key for sub in _SENSITIVE_KEY_SUBSTRINGS):
        if isinstance(value, str):
            return "[REDACTED]"
    if isinstance(value, dict):
        return sanitize_manifest(value)
    if isinstance(value, list):
        return [sanitize_value(key, item) for item in value]
    return value


def sanitize_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of dictionary with sensitive values redacted."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        sanitized[k] = sanitize_value(k, v)
    return sanitized


def write_run_artifacts(
    artifacts_dir: Path,
    run: AgentRun,
    manifest: dict[str, object] | None = None,
) -> RunArtifacts:
    """Write run-scoped artifacts (manifest, patch, test log) into artifacts_dir.

    Ensures artifacts directory exists, sanitizes credentials from manifest,
    bounds log sizes, and returns structured artifact paths.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest
    manifest_data: dict[str, Any] = {
        "run_id": run.run_id,
        "task": run.task,
        "workspace": str(run.workspace),
        "status": run.status.value,
        "benchmark_outcome": (
            run.benchmark_outcome.value if run.benchmark_outcome else None
        ),
        "verification_quality": run.verification_quality.value,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "duration_seconds": run.duration_seconds,
        "steps": run.steps,
        "model_calls": run.model_calls,
        "tokens": {
            "prompt": run.total_prompt_tokens,
            "completion": run.total_completion_tokens,
            "total": run.total_tokens,
        },
        "changed_paths": run.changed_paths,
        "unresolved_issues": run.unresolved_issues,
    }
    if manifest:
        manifest_data["config"] = sanitize_manifest(manifest)

    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 2. Candidate patch (diff)
    patch_path: Path | None = None
    if run.candidate_diff and run.candidate_diff.strip():
        patch_path = artifacts_dir / "solution.patch"
        patch_path.write_text(run.candidate_diff, encoding="utf-8")

    # 3. Verification log
    verification_log_path: Path | None = None
    if run.test_invocations or run.last_test_result:
        log_lines: list[str] = []
        for i, invocation in enumerate(run.test_invocations, start=1):
            target = invocation.get("target", ".")
            exit_code = invocation.get("exit_code")
            stdout = str(invocation.get("stdout") or "")
            stderr = str(invocation.get("stderr") or "")
            log_lines.append(
                f"--- Test Invocation {i}: target={target} exit_code={exit_code} ---"
            )
            if stdout:
                log_lines.append(stdout)
            if stderr:
                log_lines.append(stderr)
        if log_lines:
            log_content = "\n".join(log_lines)
            if len(log_content.encode("utf-8")) > MAX_VERIFICATION_LOG_BYTES:
                log_content = (
                    log_content[:MAX_VERIFICATION_LOG_BYTES] + "\n... [TRUNCATED]"
                )
            verification_log_path = artifacts_dir / "verification.log"
            verification_log_path.write_text(log_content, encoding="utf-8")

    return RunArtifacts(
        artifacts_dir=artifacts_dir,
        manifest_path=manifest_path,
        patch_path=patch_path,
        verification_log_path=verification_log_path,
    )
