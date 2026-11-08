"""Structured event schemas and benchmark outcome taxonomy for agent observability."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class RunEventType(str, Enum):
    """Lifecycle event types emitted during an agent run."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    MODEL_ERROR = "model_error"
    TOOL_REQUESTED = "tool_requested"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    VERIFICATION_RESULT = "verification_result"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


class BenchmarkOutcome(str, Enum):
    """Standard taxonomy for independent benchmark evaluation outcomes.

    These are separate from runtime terminal states (RunStatus) and model
    completion proposals.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    TIMEOUT = "timeout"
    AGENT_ERROR = "agent_error"
    SETUP_ERROR = "setup_error"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class RunEvent:
    """An immutable, timestamped event in an agent execution trace."""

    event_id: str
    run_id: str
    sequence: int
    timestamp: float
    event_type: RunEventType
    step: int = 0
    duration_ms: float | None = None
    data: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        event_type: RunEventType,
        step: int = 0,
        duration_ms: float | None = None,
        data: dict[str, object] | None = None,
        timestamp: float | None = None,
    ) -> RunEvent:
        """Construct a new event with auto-generated event_id and timestamp."""
        ts = time.time() if timestamp is None else timestamp
        return cls(
            event_id=f"{run_id[:8]}_{sequence:04d}",
            run_id=run_id,
            sequence=sequence,
            timestamp=ts,
            event_type=event_type,
            step=step,
            duration_ms=duration_ms,
            data={} if data is None else data,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize event to a JSON-compatible dictionary."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result
