"""Sequential benchmark suite runner and summary reporting."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.agent.runtime import (
    DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS,
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RUN_TIMEOUT,
    Model,
)
from harness.context import ContextStrategy, TokenBudget, TokenCounter
from harness.evaluation.config import ExperimentConfig
from harness.evaluation.manifest import ManifestError, TaskManifest
from harness.evaluation.runner import EvaluationResult, evaluate_task
from harness.llama_client import chat
from harness.retrieval import RetrievalBackend
from harness.storage.database import SqliteRunRecorder
from harness.storage.recorder import RunRecorder
from harness.tools.policy import ToolPolicy


def load_suite_dir(suite_dir: Path) -> tuple[TaskManifest, ...]:
    """Load and validate all ``*.json`` task manifests in ``suite_dir`` in sorted order."""
    if not suite_dir.is_dir():
        raise ManifestError(f"Suite directory does not exist: '{suite_dir}'")
    paths = sorted(suite_dir.glob("*.json"))
    if not paths:
        raise ManifestError(f"No task manifest JSON files found in '{suite_dir}'")
    return tuple(TaskManifest.from_json_file(path) for path in paths)


@dataclass(frozen=True)
class SuiteSummary:
    """Aggregated outcome and tier breakdown across a sequential task suite."""

    results: tuple[EvaluationResult, ...]
    outcome_counts: dict[str, int]
    oracle_tier_counts: dict[str, int]

    @classmethod
    def from_results(cls, results: Sequence[EvaluationResult]) -> SuiteSummary:
        """Build a SuiteSummary from an ordered sequence of EvaluationResults."""
        outcomes = Counter(r.outcome.value for r in results)
        tiers = Counter(r.oracle_tier.value for r in results)
        return cls(
            results=tuple(results),
            outcome_counts=dict(sorted(outcomes.items())),
            oracle_tier_counts=dict(sorted(tiers.items())),
        )

    @property
    def resolved_count(self) -> int:
        return self.outcome_counts.get(BenchmarkOutcome.RESOLVED.value, 0)

    def to_dict(self) -> dict[str, object]:
        """Serialize suite summary to a JSON-compatible dictionary."""
        return {
            "total_tasks": len(self.results),
            "resolved_count": self.resolved_count,
            "outcome_counts": self.outcome_counts,
            "oracle_tier_counts": self.oracle_tier_counts,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "run_id": r.run_id,
                    "outcome": r.outcome.value,
                    "oracle_tier": r.oracle_tier.value,
                    "oracle_passed": r.oracle_passed,
                    "agent_status": r.agent_status.value if r.agent_status else None,
                    "error_detail": r.error_detail,
                    "duration_seconds": r.duration_seconds,
                    "model_calls": r.model_calls,
                    "steps": r.steps,
                    "tool_calls": r.tool_calls,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "context_edit_count": r.context_edit_count,
                    "context_pruned_count": r.context_pruned_count,
                    "retrieval_duration_ms": r.retrieval_duration_ms,
                    "retrieval_parse_count": r.retrieval_parse_count,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        """Render a concise Markdown summary table for baseline reporting."""
        lines = [
            f"Total tasks: {len(self.results)} | Resolved: {self.resolved_count}",
            "",
            "| Task ID | Outcome | Oracle Tier | Agent Status | Run ID | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in self.results:
            status = r.agent_status.value if r.agent_status else "-"
            run_id = r.run_id or "-"
            notes = (r.error_detail or "-").replace("\n", " ")[:120]
            lines.append(
                f"| `{r.task_id}` | `{r.outcome.value}` | `{r.oracle_tier.value}` "
                f"| `{status}` | `{run_id}` | {notes} |"
            )
        return "\n".join(lines) + "\n"


def evaluate_suite(
    manifests: Sequence[TaskManifest],
    *,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
    db_path: Path | None = None,
    recorder: RunRecorder | None = None,
    policy: ToolPolicy | None = None,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_consecutive_invalid_actions: int = DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 120.0,
    cleanup_workspaces: bool = True,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
    context_strategy: ContextStrategy | str = "bounded_history",
    token_budget: TokenBudget | None = None,
    token_counter: TokenCounter | Callable[[str], int] | None = None,
    max_tool_output_bytes: int | None = None,
    retrieval_backend: RetrievalBackend | str | None = None,
) -> SuiteSummary:
    """Run all manifests sequentially on clean checkouts and return a SuiteSummary."""
    active_recorder = recorder
    if active_recorder is None and db_path is not None:
        active_recorder = SqliteRunRecorder(db_path, artifacts_dir=artifacts_root)

    results: list[EvaluationResult] = []
    for manifest in manifests:
        result = evaluate_task(
            manifest,
            work_root=work_root,
            model=model,
            artifacts_root=artifacts_root,
            recorder=active_recorder,
            policy=policy,
            max_model_calls=max_model_calls,
            max_steps=max_steps,
            max_consecutive_invalid_actions=max_consecutive_invalid_actions,
            request_timeout=request_timeout,
            run_timeout=run_timeout,
            oracle_timeout=oracle_timeout,
            setup_timeout=setup_timeout,
            cleanup_workspaces=cleanup_workspaces,
            experiment_config=experiment_config,
            server_pid=server_pid,
            collect_memory=collect_memory,
            context_strategy=context_strategy,
            token_budget=token_budget,
            token_counter=token_counter,
            max_tool_output_bytes=max_tool_output_bytes,
            retrieval_backend=retrieval_backend,
        )
        results.append(result)

    summary = SuiteSummary.from_results(results)
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "suite_summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (artifacts_root / "suite_summary.md").write_text(
            summary.to_markdown(),
            encoding="utf-8",
        )
    return summary


@dataclass(frozen=True)
class StrategyComparisonReport:
    """Side-by-side Phase 5 comparison between bounded_history and editable_context."""

    baseline: SuiteSummary
    editable_context: SuiteSummary

    def to_markdown(self) -> str:
        """Render a transparent comparison table including negative results and overhead."""
        b_map = {r.task_id: r for r in self.baseline.results}
        e_map = {r.task_id: r for r in self.editable_context.results}
        task_ids = [r.task_id for r in self.baseline.results]

        lines = [
            f"Baseline resolved: {self.baseline.resolved_count}/{len(self.baseline.results)} | "
            f"Editable-context resolved: {self.editable_context.resolved_count}/{len(self.editable_context.results)}",
            "",
            "| Task ID | Baseline Outcome | Editable Outcome | Calls (B vs E) | Tool Calls (B vs E) | Prompt Tokens (B vs E) | Edits / Pruned (E) | Wall Time s (B vs E) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for tid in task_ids:
            b = b_map[tid]
            e = e_map[tid]
            b_tok = str(b.prompt_tokens) if b.prompt_tokens is not None else "-"
            e_tok = str(e.prompt_tokens) if e.prompt_tokens is not None else "-"
            lines.append(
                f"| `{tid}` | `{b.outcome.value}` ({b.oracle_tier.value}) "
                f"| `{e.outcome.value}` ({e.oracle_tier.value}) "
                f"| {b.model_calls} vs {e.model_calls} "
                f"| {b.tool_calls} vs {e.tool_calls} "
                f"| {b_tok} vs {e_tok} "
                f"| {e.context_edit_count} / {e.context_pruned_count} "
                f"| {b.duration_seconds:.2f} vs {e.duration_seconds:.2f} |"
            )
        return "\n".join(lines) + "\n"


def compare_strategies(
    manifests: Sequence[TaskManifest],
    *,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
    db_path: Path | None = None,
    policy: ToolPolicy | None = None,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_consecutive_invalid_actions: int = DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 120.0,
    cleanup_workspaces: bool = True,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
    token_budget: TokenBudget | None = None,
    token_counter: TokenCounter | Callable[[str], int] | None = None,
    max_tool_output_bytes: int | None = None,
) -> StrategyComparisonReport:
    """Run bounded_history and editable_context under identical pinned settings."""
    common_kwargs: dict[str, object] = {
        "model": model,
        "db_path": db_path,
        "policy": policy,
        "max_model_calls": max_model_calls,
        "max_steps": max_steps,
        "request_timeout": request_timeout,
        "run_timeout": run_timeout,
        "oracle_timeout": oracle_timeout,
        "setup_timeout": setup_timeout,
        "cleanup_workspaces": cleanup_workspaces,
        "experiment_config": experiment_config,
        "server_pid": server_pid,
        "collect_memory": collect_memory,
        "token_budget": token_budget,
        "token_counter": token_counter,
        "max_tool_output_bytes": max_tool_output_bytes,
    }
    baseline = evaluate_suite(
        manifests,
        work_root=work_root / "bounded_history",
        artifacts_root=(artifacts_root / "bounded_history" if artifacts_root else None),
        context_strategy="bounded_history",
        **common_kwargs,  # type: ignore[arg-type]
    )
    editable = evaluate_suite(
        manifests,
        work_root=work_root / "editable_context",
        artifacts_root=(
            artifacts_root / "editable_context" if artifacts_root else None
        ),
        context_strategy="editable_context",
        **common_kwargs,  # type: ignore[arg-type]
    )
    report = StrategyComparisonReport(baseline=baseline, editable_context=editable)
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "strategy_comparison.md").write_text(
            report.to_markdown(),
            encoding="utf-8",
        )
    return report


@dataclass(frozen=True)
class RetrievalComparisonReport:
    """Side-by-side Phase 6 comparison between lexical_only and code_aware."""

    lexical_only: SuiteSummary
    code_aware: SuiteSummary

    def to_markdown(self) -> str:
        """Render a transparent retrieval comparison table with timing and parse overhead."""
        l_map = {r.task_id: r for r in self.lexical_only.results}
        c_map = {r.task_id: r for r in self.code_aware.results}
        task_ids = [r.task_id for r in self.lexical_only.results]

        lines = [
            f"Lexical-only resolved: {self.lexical_only.resolved_count}/{len(self.lexical_only.results)} | "
            f"Code-aware resolved: {self.code_aware.resolved_count}/{len(self.code_aware.results)}",
            "",
            "| Task ID | Lexical Outcome | Code-Aware Outcome | Calls (L vs C) | Prompt Tokens (L vs C) | Retrieval ms (L vs C) | Parses (C) | Wall Time s (L vs C) |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
        for tid in task_ids:
            l_res = l_map[tid]
            c_res = c_map[tid]
            l_tok = str(l_res.prompt_tokens) if l_res.prompt_tokens is not None else "-"
            c_tok = str(c_res.prompt_tokens) if c_res.prompt_tokens is not None else "-"
            lines.append(
                f"| `{tid}` | `{l_res.outcome.value}` ({l_res.oracle_tier.value}) "
                f"| `{c_res.outcome.value}` ({c_res.oracle_tier.value}) "
                f"| {l_res.model_calls} vs {c_res.model_calls} "
                f"| {l_tok} vs {c_tok} "
                f"| {l_res.retrieval_duration_ms:.1f} vs {c_res.retrieval_duration_ms:.1f} "
                f"| {c_res.retrieval_parse_count} "
                f"| {l_res.duration_seconds:.2f} vs {c_res.duration_seconds:.2f} |"
            )
        return "\n".join(lines) + "\n"


def compare_retrieval_modes(
    manifests: Sequence[TaskManifest],
    *,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
    db_path: Path | None = None,
    policy: ToolPolicy | None = None,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_consecutive_invalid_actions: int = DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 120.0,
    cleanup_workspaces: bool = True,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
    token_budget: TokenBudget | None = None,
    token_counter: TokenCounter | Callable[[str], int] | None = None,
    max_tool_output_bytes: int | None = None,
) -> RetrievalComparisonReport:
    """Run lexical_only vs code_aware with bounded_history held constant."""
    common_kwargs: dict[str, object] = {
        "model": model,
        "db_path": db_path,
        "policy": policy,
        "max_model_calls": max_model_calls,
        "max_steps": max_steps,
        "request_timeout": request_timeout,
        "run_timeout": run_timeout,
        "oracle_timeout": oracle_timeout,
        "setup_timeout": setup_timeout,
        "cleanup_workspaces": cleanup_workspaces,
        "experiment_config": experiment_config,
        "server_pid": server_pid,
        "collect_memory": collect_memory,
        "context_strategy": "bounded_history",
        "token_budget": token_budget,
        "token_counter": token_counter,
        "max_tool_output_bytes": max_tool_output_bytes,
    }
    lexical = evaluate_suite(
        manifests,
        work_root=work_root / "lexical_only",
        artifacts_root=artifacts_root / "lexical_only" if artifacts_root else None,
        retrieval_backend="lexical_only",
        **common_kwargs,  # type: ignore[arg-type]
    )
    code_aware = evaluate_suite(
        manifests,
        work_root=work_root / "code_aware",
        artifacts_root=artifacts_root / "code_aware" if artifacts_root else None,
        retrieval_backend="code_aware",
        **common_kwargs,  # type: ignore[arg-type]
    )
    report = RetrievalComparisonReport(lexical_only=lexical, code_aware=code_aware)
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "retrieval_comparison.md").write_text(
            report.to_markdown(),
            encoding="utf-8",
        )
    return report
