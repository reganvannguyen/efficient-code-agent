"""Recorder protocol and failure policy for agent trace storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.agent.events import RunEvent
    from harness.agent.state import AgentRun


class LoggingPolicy(str, Enum):
    """Behavior when a trace or artifact storage write fails."""

    DEGRADED = "degraded"
    FAIL = "fail"


class StorageError(RuntimeError):
    """Raised when trace or artifact persistence fails under FAIL policy."""


@dataclass(frozen=True)
class RunArtifacts:
    """Paths to generated run-scoped artifact files."""

    artifacts_dir: Path
    manifest_path: Path
    patch_path: Path | None = None
    verification_log_path: Path | None = None


@runtime_checkable
class RunRecorder(Protocol):
    """Protocol for recording runs, step events, and disk artifacts."""

    def on_run_started(
        self,
        run: AgentRun,
        manifest: dict[str, object] | None = None,
    ) -> None:
        """Record the start of a run with optional configuration manifest."""
        ...

    def on_event(self, event: RunEvent) -> None:
        """Record an immutable lifecycle or step event."""
        ...

    def on_run_finished(self, run: AgentRun) -> RunArtifacts | None:
        """Record terminal run outcome, persist final artifacts, and return paths."""
        ...
