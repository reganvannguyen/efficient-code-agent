"""Built-in context strategies: Phase 4 baseline and opt-in editable context."""

from __future__ import annotations

from dataclasses import dataclass

from harness.context.strategy import ContextInput, ContextStrategy


def _clip_utf8(value: str, byte_limit: int) -> str:
    raw = value.encode("utf-8", errors="replace")[:byte_limit]
    return raw.decode("utf-8", errors="ignore")


def _has_recorded_notes(notes: str | None) -> bool:
    if not notes or not notes.strip():
        return False
    in_body_section = False
    for line in notes.splitlines():
        stripped = line.strip()
        if stripped in {
            "## Confirmed Findings",
            "## Current Hypothesis",
            "## Next Action",
            "## Open Questions",
        }:
            in_body_section = True
            continue
        if stripped.startswith("#"):
            in_body_section = False
            continue
        if in_body_section and stripped:
            return True
    return "#" not in notes


@dataclass(frozen=True)
class BoundedHistoryStrategy:
    """Phase 2-4 baseline strategy: pinned instructions/task plus oldest-dropped history."""

    name: str = "bounded_history"

    def parameters(self) -> dict[str, object]:
        """Return effective strategy parameters."""
        return {"strategy": self.name, "drop_oldest_history": True}

    def build_prompt(self, ctx: ContextInput) -> str | None:
        """Build a UTF-8-byte-bounded prompt, dropping oldest complete exchanges."""
        pinned = f"{ctx.instructions}\nOriginal task (immutable):\n{ctx.task}"
        while True:
            prompt = pinned
            if ctx.history:
                prompt += (
                    "\n\nAction and observation history "
                    "(JSON exchanges, oldest first):\n"
                )
                prompt += "\n".join(ctx.history)
            prompt += "\n\nNext action:\n" + ctx.directive
            if len(prompt.encode("utf-8")) <= ctx.byte_limit:
                return prompt
            if len(ctx.history) <= 1:
                return None
            del ctx.history[0]


@dataclass(frozen=True)
class EditableContextStrategy:
    """Opt-in context strategy supporting task notes and repository excerpts."""

    name: str = "editable_context"
    max_notes_bytes: int | None = None
    max_excerpts_bytes: int | None = None

    def parameters(self) -> dict[str, object]:
        """Return effective strategy parameters."""
        return {
            "strategy": self.name,
            "max_notes_bytes": self.max_notes_bytes,
            "max_excerpts_bytes": self.max_excerpts_bytes,
        }

    def build_prompt(self, ctx: ContextInput) -> str | None:
        """Build a bounded prompt with immutable task/rules, optional notes/excerpts, and history."""
        pinned = f"{ctx.instructions}\nOriginal task (immutable):\n{ctx.task}"

        notes_block = ""
        if ctx.task_notes and ctx.task_notes.strip():
            notes_text = ctx.task_notes.strip()
            if self.max_notes_bytes is not None:
                notes_text = _clip_utf8(notes_text, self.max_notes_bytes)
            notes_block = f"\n\nWorking notes:\n{notes_text}"

        excerpts_block = ""
        if ctx.repo_excerpts:
            joined = "\n".join(item for item in ctx.repo_excerpts if item)
            if joined:
                if self.max_excerpts_bytes is not None:
                    joined = _clip_utf8(joined, self.max_excerpts_bytes)
                excerpts_block = f"\n\nRepository excerpts:\n{joined}"

        index_block = ""
        if ctx.memory_index:
            index_block = (
                "\n\nAvailable memory notes (read with read_memory): "
                + ", ".join(ctx.memory_index)
            )

        dropped_history = False
        while True:
            nudge_block = ""
            if dropped_history:
                nudge_block = "\n\n[Context hint: older history was pruned to fit budget; save key findings to working.md with update_context.]"
            elif ctx.steps >= 2 and not _has_recorded_notes(ctx.task_notes):
                nudge_block = "\n\n[Context hint: working.md has no saved findings yet; consider calling update_context if you have confirmed findings or a plan change.]"
            base_prompt = f"{pinned}{notes_block}{index_block}{excerpts_block}"
            if ctx.history:
                base_prompt += (
                    "\n\nAction and observation history "
                    "(JSON exchanges, oldest first):\n"
                )
                base_prompt += "\n".join(ctx.history)
            if nudge_block:
                nudged = f"{base_prompt}{nudge_block}\n\nNext action:\n{ctx.directive}"
                if len(nudged.encode("utf-8")) <= ctx.byte_limit:
                    return nudged
            prompt = f"{base_prompt}\n\nNext action:\n{ctx.directive}"
            if len(prompt.encode("utf-8")) <= ctx.byte_limit:
                return prompt
            if len(ctx.history) <= 1:
                return None
            del ctx.history[0]
            dropped_history = True


def resolve_context_strategy(
    strategy: ContextStrategy | str | None,
) -> ContextStrategy:
    """Resolve a strategy name or instance, defaulting to BoundedHistoryStrategy."""
    if strategy is None or strategy == "bounded_history":
        return BoundedHistoryStrategy()
    if strategy == "editable_context":
        return EditableContextStrategy()
    if isinstance(strategy, ContextStrategy):
        return strategy
    raise ValueError(
        f"Unknown context_strategy {strategy!r}. "
        "Expected 'bounded_history', 'editable_context', or a ContextStrategy."
    )
