"""Immutable permissions and approval metadata for coding tools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ToolMode(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


def normalize_target(target: str) -> str:
    """Normalize harmless relative path spelling for schema/allowlist equality."""
    value = Path(target).as_posix()
    return value or "."


@dataclass(frozen=True)
class ToolPolicy:
    """Permissions for write and test tools; reads remain automatically allowed."""

    edit_file: ToolMode = ToolMode.ASK
    run_tests: ToolMode = ToolMode.ASK
    test_targets: tuple[str, ...] = (".",)
    test_timeout: float = 60.0
    auto_test_after_edit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.auto_test_after_edit, bool):
            raise ValueError("auto_test_after_edit must be a boolean")
        if not isinstance(self.edit_file, ToolMode) or not isinstance(
            self.run_tests, ToolMode
        ):
            raise ValueError("tool modes must be ToolMode values")
        if (
            not isinstance(self.test_targets, tuple)
            or not self.test_targets
            or any(not isinstance(target, str) for target in self.test_targets)
        ):
            raise ValueError("test_targets must be a non-empty tuple of strings")
        if (
            isinstance(self.test_timeout, bool)
            or not isinstance(self.test_timeout, (int, float))
            or not math.isfinite(self.test_timeout)
            or self.test_timeout <= 0
        ):
            raise ValueError("test_timeout must be finite and positive")


@dataclass(frozen=True)
class ApprovalRequest:
    """Read-only preview of the exact prepared operation requiring approval."""

    tool: str
    root: Path
    path: str | None = None
    expected_sha256: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    diff: str = ""
    diff_truncated: bool = False
    preset: str | None = None
    target: str | None = None
    executable: str | None = None
    argv: tuple[str, ...] = ()
