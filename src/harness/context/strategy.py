"""Context-strategy protocol and immutable input payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.agent.events import RunEvent


@dataclass(frozen=True)
class ContextInput:
    """Inputs supplied to a ContextStrategy for building one bounded model prompt."""

    instructions: str
    directive: str
    task: str
    history: list[str]
    byte_limit: int
    run_status: str = "running"
    steps: int = 0
    recent_events: tuple[RunEvent, ...] = ()
    repo_excerpts: tuple[str, ...] = ()
    task_notes: str | None = None
    memory_index: tuple[str, ...] = ()


@runtime_checkable
class ContextStrategy(Protocol):
    """Protocol for swappable prompt-assembly strategies."""

    name: str

    def parameters(self) -> dict[str, object]:
        """Return JSON-serializable effective strategy parameters for manifests."""
        ...

    def build_prompt(self, ctx: ContextInput) -> str | None:
        """Assemble a UTF-8-byte-bounded prompt, or None if the budget cannot be met."""
        ...
