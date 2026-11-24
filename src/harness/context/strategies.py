"""Built-in context strategies: Phase 4 baseline and opt-in editable context."""

from __future__ import annotations

from dataclasses import dataclass

from harness.context.budgeting import (
    PromptPruningReport,
    trim_notes_deterministically,
    within_budget,
)
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
        prompt, _ = self.build_prompt_with_report(ctx)
        return prompt

    def build_prompt_with_report(
        self, ctx: ContextInput
    ) -> tuple[str | None, PromptPruningReport]:
        """Build a bounded prompt and return a pruning report."""
        pinned = f"{ctx.instructions}\nOriginal task (immutable):\n{ctx.task}"
        protected = f"{pinned}\n\nNext action:\n{ctx.directive}"
        protected_fits, protected_bytes, protected_tokens = within_budget(
            protected,
            byte_limit=ctx.byte_limit,
            token_limit=ctx.token_limit,
            count_tokens=ctx.token_counter,
        )
        if not protected_fits:
            return (
                None,
                PromptPruningReport(
                    prompt_tokens=protected_tokens,
                    token_budget=ctx.token_limit,
                    prompt_bytes=protected_bytes,
                    byte_limit=ctx.byte_limit,
                    exceeded_reason=(
                        "context_budget_exceeded"
                        if ctx.token_limit is not None
                        else "prompt_budget"
                    ),
                ),
            )

        dropped_history = 0
        while True:
            prompt = pinned
            if ctx.history:
                prompt += (
                    "\n\nAction and observation history "
                    "(JSON exchanges, oldest first):\n"
                )
                prompt += "\n".join(ctx.history)
            prompt += "\n\nNext action:\n" + ctx.directive
            fits, byte_count, token_count = within_budget(
                prompt,
                byte_limit=ctx.byte_limit,
                token_limit=ctx.token_limit,
                count_tokens=ctx.token_counter,
            )
            if fits:
                return (
                    prompt,
                    PromptPruningReport(
                        pruned=dropped_history > 0,
                        dropped_history_count=dropped_history,
                        prompt_tokens=token_count,
                        token_budget=ctx.token_limit,
                        prompt_bytes=byte_count,
                        byte_limit=ctx.byte_limit,
                    ),
                )
            if len(ctx.history) <= 1:
                return (
                    None,
                    PromptPruningReport(
                        pruned=dropped_history > 0,
                        dropped_history_count=dropped_history,
                        prompt_tokens=token_count,
                        token_budget=ctx.token_limit,
                        prompt_bytes=byte_count,
                        byte_limit=ctx.byte_limit,
                        exceeded_reason=(
                            "context_budget_exceeded"
                            if ctx.token_limit is not None
                            else "prompt_budget"
                        ),
                    ),
                )
            del ctx.history[0]
            dropped_history += 1


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
        prompt, _ = self.build_prompt_with_report(ctx)
        return prompt

    def build_prompt_with_report(
        self, ctx: ContextInput
    ) -> tuple[str | None, PromptPruningReport]:
        """Build a bounded prompt by priority and return a deterministic pruning report."""
        pinned = f"{ctx.instructions}\nOriginal task (immutable):\n{ctx.task}"
        protected = f"{pinned}\n\nNext action:\n{ctx.directive}"
        protected_fits, protected_bytes, protected_tokens = within_budget(
            protected,
            byte_limit=ctx.byte_limit,
            token_limit=ctx.token_limit,
            count_tokens=ctx.token_counter,
        )
        if not protected_fits:
            return (
                None,
                PromptPruningReport(
                    prompt_tokens=protected_tokens,
                    token_budget=ctx.token_limit,
                    prompt_bytes=protected_bytes,
                    byte_limit=ctx.byte_limit,
                    exceeded_reason="context_budget_exceeded",
                ),
            )

        notes_text = (
            ctx.task_notes.strip() if ctx.task_notes and ctx.task_notes.strip() else ""
        )
        trimmed_notes = False
        if notes_text and self.max_notes_bytes is not None:
            clipped = _clip_utf8(notes_text, self.max_notes_bytes)
            trimmed_notes = len(clipped.encode("utf-8")) < len(
                notes_text.encode("utf-8")
            )
            notes_text = clipped

        excerpts_list = [item for item in ctx.repo_excerpts if item]
        excerpts_text = "\n".join(excerpts_list)
        capped_excerpts = 0
        if excerpts_text and self.max_excerpts_bytes is not None:
            clipped = _clip_utf8(excerpts_text, self.max_excerpts_bytes)
            if len(clipped.encode("utf-8")) < len(excerpts_text.encode("utf-8")):
                capped_excerpts = len(excerpts_list)
            excerpts_text = clipped

        index_items = list(ctx.memory_index)
        omitted_findings = 0
        dropped_history = 0

        def assemble(*, include_nudge: bool) -> str:
            notes_block = f"\n\nWorking notes:\n{notes_text}" if notes_text else ""
            index_block = (
                "\n\nAvailable memory notes (read with read_memory): "
                + ", ".join(index_items)
                if index_items
                else ""
            )
            excerpts_block = (
                f"\n\nRepository excerpts:\n{excerpts_text}" if excerpts_text else ""
            )
            base_prompt = f"{pinned}{notes_block}{index_block}{excerpts_block}"
            if ctx.history:
                base_prompt += (
                    "\n\nAction and observation history "
                    "(JSON exchanges, oldest first):\n"
                )
                base_prompt += "\n".join(ctx.history)
            if include_nudge:
                nudge_block = ""
                if dropped_history > 0:
                    nudge_block = "\n\n[Context hint: older history was pruned to fit budget; save key findings to working.md with update_context.]"
                elif ctx.steps >= 2 and not _has_recorded_notes(ctx.task_notes):
                    nudge_block = "\n\n[Context hint: working.md has no saved findings yet; consider calling update_context if you have confirmed findings or a plan change.]"
                if nudge_block:
                    return (
                        f"{base_prompt}{nudge_block}\n\nNext action:\n{ctx.directive}"
                    )
            return f"{base_prompt}\n\nNext action:\n{ctx.directive}"

        while True:
            nudged = assemble(include_nudge=True)
            fits_nudged, byte_count, token_count = within_budget(
                nudged,
                byte_limit=ctx.byte_limit,
                token_limit=ctx.token_limit,
                count_tokens=ctx.token_counter,
            )
            if fits_nudged:
                return (
                    nudged,
                    PromptPruningReport(
                        pruned=(
                            dropped_history > 0
                            or capped_excerpts > 0
                            or omitted_findings > 0
                            or trimmed_notes
                        ),
                        dropped_history_count=dropped_history,
                        capped_excerpts_count=capped_excerpts,
                        omitted_findings_count=omitted_findings,
                        trimmed_notes=trimmed_notes,
                        prompt_tokens=token_count,
                        token_budget=ctx.token_limit,
                        prompt_bytes=byte_count,
                        byte_limit=ctx.byte_limit,
                    ),
                )

            prompt = assemble(include_nudge=False)
            fits, byte_count, token_count = within_budget(
                prompt,
                byte_limit=ctx.byte_limit,
                token_limit=ctx.token_limit,
                count_tokens=ctx.token_counter,
            )
            if fits:
                return (
                    prompt,
                    PromptPruningReport(
                        pruned=(
                            dropped_history > 0
                            or capped_excerpts > 0
                            or omitted_findings > 0
                            or trimmed_notes
                        ),
                        dropped_history_count=dropped_history,
                        capped_excerpts_count=capped_excerpts,
                        omitted_findings_count=omitted_findings,
                        trimmed_notes=trimmed_notes,
                        prompt_tokens=token_count,
                        token_budget=ctx.token_limit,
                        prompt_bytes=byte_count,
                        byte_limit=ctx.byte_limit,
                    ),
                )

            # Deterministic overflow order:
            # 1) Drop oldest history (keeping newest if > 1)
            if len(ctx.history) > 1:
                del ctx.history[0]
                dropped_history += 1
                continue
            # 2) Drop repository excerpts
            if excerpts_text:
                capped_excerpts = max(capped_excerpts, len(excerpts_list))
                excerpts_text = ""
                continue
            # 3) Omit nonessential findings / memory index entries
            if index_items:
                omitted_findings += len(index_items)
                index_items = []
                continue
            # 4) Deterministically trim editable working notes
            if notes_text:
                remaining_bytes = max(
                    0,
                    ctx.byte_limit - protected_bytes - 32,
                )
                remaining_tokens: int | None = None
                if ctx.token_limit is not None:
                    base_without_notes = assemble(include_nudge=False).replace(
                        f"\n\nWorking notes:\n{notes_text}", ""
                    )
                    counter = ctx.token_counter
                    if counter is None:
                        from harness.context.budgeting import estimate_tokens

                        counter = estimate_tokens
                    remaining_tokens = max(
                        0,
                        ctx.token_limit - counter(base_without_notes) - 4,
                    )
                trimmed = trim_notes_deterministically(
                    notes_text,
                    max_bytes=remaining_bytes,
                    token_limit=remaining_tokens,
                    count_tokens=ctx.token_counter,
                )
                if trimmed and len(trimmed.encode("utf-8")) < len(
                    notes_text.encode("utf-8")
                ):
                    notes_text = trimmed
                    trimmed_notes = True
                    continue
                notes_text = ""
                trimmed_notes = True
                continue
            # 5) Nothing left to prune while preserving protected content
            return (
                None,
                PromptPruningReport(
                    pruned=(
                        dropped_history > 0
                        or capped_excerpts > 0
                        or omitted_findings > 0
                        or trimmed_notes
                    ),
                    dropped_history_count=dropped_history,
                    capped_excerpts_count=capped_excerpts,
                    omitted_findings_count=omitted_findings,
                    trimmed_notes=trimmed_notes,
                    prompt_tokens=token_count,
                    token_budget=ctx.token_limit,
                    prompt_bytes=byte_count,
                    byte_limit=ctx.byte_limit,
                    exceeded_reason=(
                        "context_budget_exceeded"
                        if ctx.token_limit is not None
                        else "prompt_budget"
                    ),
                ),
            )


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
