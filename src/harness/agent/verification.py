"""Verification quality assessment and loop progress tracking for agent runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.agent.state import AgentRun
    from harness.tools.policy import ToolPolicy


class VerificationQuality(str, Enum):
    """Quality of runtime verification backing a completion proposal."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"


class LoopIntervention(str, Enum):
    """Graduated response to repetitive identical tool actions."""

    ALLOW = "allow"
    BLOCK_DUPLICATE = "block_duplicate"
    TEMP_BLOCK_TOOL = "temp_block_tool"
    STOP = "stop"


@dataclass
class NoProgressTracker:
    """Tracks repetitive actions and rejected completions with graduated intervention."""

    max_repetitions: int = 6
    max_completion_rejections: int = 4
    identical_tool_streak: int = 0
    last_action_signature: tuple[str, str] | None = None
    temporarily_blocked_tool: str | None = None
    rejected_completion_streak: int = 0

    def evaluate_action(
        self, name: str, arguments: dict[str, object]
    ) -> LoopIntervention:
        """Evaluate a tool call before execution and return the graduated action."""
        try:
            canonical_args = json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            canonical_args = str(arguments)
        signature = (name, canonical_args)

        self.rejected_completion_streak = 0

        if (
            self.temporarily_blocked_tool is not None
            and name == self.temporarily_blocked_tool
        ):
            self.identical_tool_streak += 1
            if self.identical_tool_streak >= self.max_repetitions:
                return LoopIntervention.STOP
            return LoopIntervention.TEMP_BLOCK_TOOL

        if signature == self.last_action_signature:
            self.identical_tool_streak += 1
            if self.identical_tool_streak >= self.max_repetitions:
                return LoopIntervention.STOP
            if self.identical_tool_streak >= 3:
                self.temporarily_blocked_tool = name
                return LoopIntervention.TEMP_BLOCK_TOOL
            return LoopIntervention.BLOCK_DUPLICATE

        self.last_action_signature = signature
        self.identical_tool_streak = 1
        self.temporarily_blocked_tool = None
        return LoopIntervention.ALLOW

    def record_action(self, name: str, arguments: dict[str, object]) -> bool:
        """Record a tool invocation. Return True if the hard repetition limit is hit."""
        return self.evaluate_action(name, arguments) is LoopIntervention.STOP

    def record_completion_rejection(self) -> bool:
        """Record an unverified completion rejection. Return True if limit hit."""
        self.rejected_completion_streak += 1
        self.last_action_signature = None
        self.identical_tool_streak = 0
        self.temporarily_blocked_tool = None
        return self.rejected_completion_streak >= self.max_completion_rejections

    def reset(self) -> None:
        """Reset all repetition counters and temporary blocks."""
        self.identical_tool_streak = 0
        self.last_action_signature = None
        self.temporarily_blocked_tool = None
        self.rejected_completion_streak = 0


def evaluate_verification(
    run: AgentRun,
    policy: ToolPolicy | None,
) -> tuple[VerificationQuality, str | None]:
    """Evaluate verification evidence for a completion proposal.

    Returns (quality, rejection_reason). If rejection_reason is None, the completion
    is accepted under the evaluated quality.
    """
    if policy is None:
        return VerificationQuality.UNVERIFIED, None

    if not run.changed_paths:
        # No files were modified; informational completion in coding mode
        return VerificationQuality.UNVERIFIED, None

    # Files were modified. Check if tests ran after the latest edit.
    if run.last_edit_step is None:
        return VerificationQuality.UNVERIFIED, None

    if run.last_test_step is None or run.last_test_step <= run.last_edit_step:
        changed_str = ", ".join(run.changed_paths)
        return (
            VerificationQuality.UNVERIFIED,
            f"Edits were made to {changed_str} at step {run.last_edit_step}, but no test "
            f"has been executed since. Run allowlisted tests using run_tests to verify "
            f"your changes before proposing completion.",
        )

    # A test was executed after the last edit. Inspect the latest test result.
    if run.last_test_result:
        test_data = run.last_test_result.get("data", {})
        exit_code = test_data.get("exit_code") if isinstance(test_data, dict) else None
        error_code = run.last_test_result.get("error_code")
        if exit_code == 0 and not error_code:
            return VerificationQuality.VERIFIED, None
        return (
            VerificationQuality.PARTIALLY_VERIFIED,
            f"Tests were executed after the last edit but failed (exit code {exit_code}). "
            f"Fix failing tests and re-run tests before proposing completion.",
        )

    return VerificationQuality.UNVERIFIED, None
