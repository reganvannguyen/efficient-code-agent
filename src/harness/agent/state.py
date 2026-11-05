"""Mutable state owned by one agent runtime attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RunStatus(str, Enum):
    """Lifecycle states available to an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"
    CANCELLED = "cancelled"


@dataclass
class AgentRun:
    """Task and runtime-owned counters for one attempt."""

    task: str
    workspace: Path
    status: RunStatus = RunStatus.PENDING
    steps: int = 0
    model_calls: int = 0
    final_response: str | None = None
    stop_reason: str | None = None
    changed_paths: list[str] = field(default_factory=list)
    last_tool_result: dict[str, object] | None = None
    last_write_state: str | None = None
    last_test_result: dict[str, object] | None = None

    def transition_to(self, status: RunStatus) -> None:
        """Move to a lifecycle state allowed by the current MVP contract."""
        allowed = {
            RunStatus.PENDING: {RunStatus.RUNNING},
            RunStatus.RUNNING: {
                RunStatus.WAITING_APPROVAL,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.LIMIT_REACHED,
                RunStatus.CANCELLED,
            },
            RunStatus.WAITING_APPROVAL: {
                RunStatus.RUNNING,
                RunStatus.FAILED,
                RunStatus.LIMIT_REACHED,
                RunStatus.CANCELLED,
            },
        }
        if status not in allowed.get(self.status, set()):
            raise ValueError(
                f"Invalid run status transition: {self.status} -> {status}"
            )
        self.status = status
