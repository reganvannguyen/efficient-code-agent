"""Task manifest schema and oracle tier definitions for fixture evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OracleTier(str, Enum):
    """Granularity and authority of the benchmark evaluation oracle."""

    LOCAL_TARGETED = "local_targeted"
    LOCAL_FULL = "local_full"
    OFFICIAL = "official"


class ManifestError(ValueError):
    """Raised when a benchmark task manifest is malformed or incomplete."""


def _require_nonempty_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field '{key}' must be a non-empty string.")
    return value.strip()


def _parse_str_tuple(
    data: dict[str, Any],
    key: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        if required:
            raise ManifestError(f"Manifest field '{key}' is required.")
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ManifestError(
            f"Manifest field '{key}' must be a list of non-empty strings."
        )
    if required and not value:
        raise ManifestError(f"Manifest field '{key}' must not be empty.")
    return tuple(item.strip() for item in value)


def _parse_setup_commands(data: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    raw = data.get("setup_commands")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError("Manifest field 'setup_commands' must be a list of argvs.")
    commands: list[tuple[str, ...]] = []
    for entry in raw:
        if (
            not isinstance(entry, list)
            or not entry
            or not all(isinstance(part, str) and part for part in entry)
        ):
            raise ManifestError(
                "Each entry in 'setup_commands' must be a non-empty list of strings."
            )
        commands.append(tuple(entry))
    return tuple(commands)


@dataclass(frozen=True)
class TaskManifest:
    """Pinned specification for one isolated benchmark evaluation task."""

    task_id: str
    repo: str
    base_commit: str
    issue_text: str
    oracle_tests: tuple[str, ...]
    oracle_tier: OracleTier = OracleTier.LOCAL_TARGETED
    agent_test_targets: tuple[str, ...] = ()
    setup_commands: tuple[tuple[str, ...], ...] = ()
    reference_patch: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def agent_prompt(self) -> str:
        """Return only the public task description safe to expose to the agent."""
        return self.issue_text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskManifest:
        """Validate and construct a TaskManifest from a JSON-compatible dict."""
        if not isinstance(data, dict):
            raise ManifestError("Task manifest must be a JSON object.")

        task_id = _require_nonempty_str(data, "task_id")
        repo = _require_nonempty_str(data, "repo")
        base_commit = _require_nonempty_str(data, "base_commit")
        issue_text = _require_nonempty_str(data, "issue_text")
        oracle_tests = _parse_str_tuple(data, "oracle_tests", required=True)
        agent_test_targets = _parse_str_tuple(
            data, "agent_test_targets", required=False
        )
        setup_commands = _parse_setup_commands(data)

        raw_tier = data.get("oracle_tier", OracleTier.LOCAL_TARGETED.value)
        if not isinstance(raw_tier, str):
            raise ManifestError("Manifest field 'oracle_tier' must be a string.")
        try:
            oracle_tier = OracleTier(raw_tier)
        except ValueError as exc:
            valid = ", ".join(t.value for t in OracleTier)
            raise ManifestError(
                f"Invalid oracle_tier '{raw_tier}'. Expected one of: {valid}."
            ) from exc

        ref_patch = data.get("reference_patch")
        if ref_patch is not None and not isinstance(ref_patch, str):
            raise ManifestError(
                "Manifest field 'reference_patch' must be a string or null."
            )

        raw_meta = data.get("metadata", {})
        if not isinstance(raw_meta, dict):
            raise ManifestError("Manifest field 'metadata' must be a JSON object.")

        return cls(
            task_id=task_id,
            repo=repo,
            base_commit=base_commit,
            issue_text=issue_text,
            oracle_tests=oracle_tests,
            oracle_tier=oracle_tier,
            agent_test_targets=agent_test_targets,
            setup_commands=setup_commands,
            reference_patch=ref_patch,
            metadata=dict(raw_meta),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> TaskManifest:
        """Load and validate a TaskManifest from a JSON file on disk."""
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except OSError as exc:
            raise ManifestError(
                f"Could not read manifest file '{path}': {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid JSON in manifest '{path}': {exc}") from exc
        if not isinstance(data, dict):
            raise ManifestError(f"Manifest '{path}' must contain a JSON object.")
        return cls.from_dict(data)
