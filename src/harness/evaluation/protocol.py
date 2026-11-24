"""Frozen Phase 7 evaluation protocol, suite digest, and leakage checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.evaluation.manifest import ManifestError, TaskManifest


@dataclass(frozen=True)
class FrozenEvaluationProtocol:
    """Pinned model, server, budget, and workspace rules for Phase 7 comparisons."""

    protocol_version: str
    gguf_filename: str
    gguf_sha256: str
    quantization: str
    context_window: int
    output_reserve: int
    max_tokens: int
    tool_format: str
    max_model_calls: int
    max_steps: int
    max_consecutive_invalid_actions: int
    request_timeout: float
    run_timeout: float
    oracle_timeout: float
    setup_timeout: float
    cleanup_workspaces: bool = True

    @classmethod
    def from_json_file(cls, path: Path) -> FrozenEvaluationProtocol:
        """Load and validate a pinned protocol JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ManifestError("Protocol JSON must be an object.")
        required_str = (
            "protocol_version",
            "gguf_filename",
            "gguf_sha256",
            "quantization",
            "tool_format",
        )
        for key in required_str:
            val = raw.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ManifestError(
                    f"Protocol field '{key}' must be a non-empty string."
                )
        required_pos_int = (
            "context_window",
            "output_reserve",
            "max_tokens",
            "max_model_calls",
            "max_steps",
            "max_consecutive_invalid_actions",
        )
        for key in required_pos_int:
            val = raw.get(key)
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise ManifestError(
                    f"Protocol field '{key}' must be a positive integer."
                )
        required_pos_num = (
            "request_timeout",
            "run_timeout",
            "oracle_timeout",
            "setup_timeout",
        )
        for key in required_pos_num:
            val = raw.get(key)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
                raise ManifestError(
                    f"Protocol field '{key}' must be a positive number."
                )
        cleanup = raw.get("cleanup_workspaces", True)
        if not isinstance(cleanup, bool):
            raise ManifestError(
                "Protocol field 'cleanup_workspaces' must be a boolean."
            )
        return cls(
            protocol_version=raw["protocol_version"].strip(),
            gguf_filename=raw["gguf_filename"].strip(),
            gguf_sha256=raw["gguf_sha256"].strip(),
            quantization=raw["quantization"].strip(),
            context_window=raw["context_window"],
            output_reserve=raw["output_reserve"],
            max_tokens=raw["max_tokens"],
            tool_format=raw["tool_format"].strip(),
            max_model_calls=raw["max_model_calls"],
            max_steps=raw["max_steps"],
            max_consecutive_invalid_actions=raw["max_consecutive_invalid_actions"],
            request_timeout=float(raw["request_timeout"]),
            run_timeout=float(raw["run_timeout"]),
            oracle_timeout=float(raw["oracle_timeout"]),
            setup_timeout=float(raw["setup_timeout"]),
            cleanup_workspaces=cleanup,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize protocol to a JSON-compatible dictionary."""
        return asdict(self)


def compute_suite_digest(manifests: Sequence[TaskManifest]) -> str:
    """Compute a deterministic SHA-256 digest over pinned manifest fields."""
    payload = [
        {
            "task_id": m.task_id,
            "repo": m.repo,
            "base_commit": m.base_commit,
            "issue_text": m.issue_text,
            "oracle_tests": list(m.oracle_tests),
            "oracle_tier": m.oracle_tier.value,
            "agent_test_targets": list(m.agent_test_targets),
            "setup_commands": [list(cmd) for cmd in m.setup_commands],
        }
        for m in sorted(manifests, key=lambda item: item.task_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_suite_partition_isolation(
    *,
    smoke: Sequence[TaskManifest] = (),
    dev: Sequence[TaskManifest] = (),
    held_out: Sequence[TaskManifest] = (),
) -> dict[str, str]:
    """Verify zero task_id overlap across smoke, dev, and held_out partitions."""
    seen: dict[str, str] = {}
    for label, group in (("smoke", smoke), ("dev", dev), ("held_out", held_out)):
        for manifest in group:
            existing = seen.get(manifest.task_id)
            if existing is not None:
                raise ManifestError(
                    f"Task '{manifest.task_id}' appears in both '{existing}' and '{label}' partitions."
                )
            seen[manifest.task_id] = label
    return {
        "smoke_digest": compute_suite_digest(smoke),
        "dev_digest": compute_suite_digest(dev),
        "held_out_digest": compute_suite_digest(held_out),
    }


def verify_agent_prompt_isolation(manifest: TaskManifest) -> None:
    """Ensure agent_prompt() never leaks reference_patch or private metadata."""
    prompt = manifest.agent_prompt()
    if manifest.reference_patch and manifest.reference_patch.strip() in prompt:
        raise ManifestError(
            f"Task '{manifest.task_id}' leaked reference_patch into agent_prompt()."
        )
    for key in ("fail_to_pass", "pass_to_pass"):
        if key in manifest.metadata and key in prompt:
            raise ManifestError(
                f"Task '{manifest.task_id}' leaked metadata '{key}' into agent_prompt()."
            )
