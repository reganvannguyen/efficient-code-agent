"""Controlled four-way internal ablation runner (baseline, retrieval-only, CLM-only, combined)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.agent.runtime import (
    DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS,
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RUN_TIMEOUT,
    Model,
)
from harness.context import TokenBudget, TokenCounter
from harness.evaluation.config import ExperimentConfig
from harness.evaluation.manifest import TaskManifest
from harness.evaluation.protocol import FrozenEvaluationProtocol, compute_suite_digest
from harness.evaluation.suite import SuiteSummary, evaluate_suite
from harness.llama_client import chat
from harness.tools.policy import ToolPolicy


@dataclass(frozen=True)
class AblationCondition:
    """One condition in the 2x2 context-strategy x retrieval-mode ablation matrix."""

    name: str
    context_strategy: str
    retrieval_mode: str


FOUR_WAY_CONDITIONS: tuple[AblationCondition, ...] = (
    AblationCondition(
        name="baseline",
        context_strategy="bounded_history",
        retrieval_mode="lexical_only",
    ),
    AblationCondition(
        name="retrieval_only",
        context_strategy="bounded_history",
        retrieval_mode="code_aware",
    ),
    AblationCondition(
        name="clm_only",
        context_strategy="editable_context",
        retrieval_mode="lexical_only",
    ),
    AblationCondition(
        name="combined",
        context_strategy="editable_context",
        retrieval_mode="code_aware",
    ),
)


@dataclass(frozen=True)
class FourWayAblationReport:
    """Aggregated results and per-task comparisons across all four ablation conditions."""

    suite_digest: str
    conditions: dict[str, SuiteSummary]
    protocol: FrozenEvaluationProtocol | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the four-way ablation report to a JSON-compatible dictionary."""
        return {
            "suite_digest": self.suite_digest,
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "conditions": {
                name: {
                    "context_strategy": cond.context_strategy,
                    "retrieval_mode": cond.retrieval_mode,
                    "summary": self.conditions[name].to_dict(),
                }
                for cond in FOUR_WAY_CONDITIONS
                for name in (cond.name,)
                if name in self.conditions
            },
        }

    def to_markdown(self) -> str:
        """Render a concise condition-level and per-task Markdown comparison."""
        lines = [
            f"Suite digest: `{self.suite_digest[:12]}`",
            "",
            "| Condition | Context Strategy | Retrieval Mode | Resolved | Model Calls | Tool Calls | Context Edits | Retrieval ms | Parses | Prompt Tokens | Wall Time (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cond in FOUR_WAY_CONDITIONS:
            summary = self.conditions.get(cond.name)
            if summary is None:
                continue
            results = summary.results
            total_calls = sum(r.model_calls for r in results)
            total_tools = sum(r.tool_calls for r in results)
            total_edits = sum(r.context_edit_count for r in results)
            total_ret_ms = sum(r.retrieval_duration_ms for r in results)
            total_parses = sum(r.retrieval_parse_count for r in results)
            prompt_tokens = sum(r.prompt_tokens or 0 for r in results)
            wall = sum(r.duration_seconds for r in results)
            lines.append(
                f"| `{cond.name}` | `{cond.context_strategy}` | `{cond.retrieval_mode}` "
                f"| {summary.resolved_count}/{len(results)} | {total_calls} | {total_tools} "
                f"| {total_edits} | {total_ret_ms:.1f} | {total_parses} "
                f"| {prompt_tokens} | {wall:.2f} |"
            )

        if self.conditions:
            first_summary = next(iter(self.conditions.values()))
            lines.extend(
                [
                    "",
                    "### Per-task outcomes",
                    "",
                    "| Task ID | Baseline | Retrieval-Only | CLM-Only | Combined |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            by_cond = {
                name: {r.task_id: r for r in summary.results}
                for name, summary in self.conditions.items()
            }
            for task in first_summary.results:
                tid = task.task_id
                cells = []
                for cond in FOUR_WAY_CONDITIONS:
                    res = by_cond.get(cond.name, {}).get(tid)
                    cells.append(
                        f"`{res.outcome.value}` ({res.model_calls}c)"
                        if res is not None
                        else "-"
                    )
                lines.append(f"| `{tid}` | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def run_four_way_ablation(
    manifests: Sequence[TaskManifest],
    *,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
    db_path: Path | None = None,
    policy: ToolPolicy | None = None,
    protocol: FrozenEvaluationProtocol | None = None,
    max_model_calls: int | None = None,
    max_steps: int | None = None,
    max_consecutive_invalid_actions: int | None = None,
    request_timeout: float | None = None,
    run_timeout: float | None = None,
    oracle_timeout: float | None = None,
    setup_timeout: float | None = None,
    cleanup_workspaces: bool | None = None,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
    token_budget: TokenBudget | None = None,
    token_counter: TokenCounter | Callable[[str], int] | None = None,
    max_tool_output_bytes: int | None = None,
) -> FourWayAblationReport:
    """Run all four internal ablation conditions under identical pinned settings."""
    eff_calls = (
        max_model_calls
        if max_model_calls is not None
        else (protocol.max_model_calls if protocol else DEFAULT_MAX_MODEL_CALLS)
    )
    eff_steps = (
        max_steps
        if max_steps is not None
        else (protocol.max_steps if protocol else DEFAULT_MAX_STEPS)
    )
    eff_invalid = (
        max_consecutive_invalid_actions
        if max_consecutive_invalid_actions is not None
        else (
            protocol.max_consecutive_invalid_actions
            if protocol
            else DEFAULT_MAX_CONSECUTIVE_INVALID_ACTIONS
        )
    )
    eff_req_timeout = (
        request_timeout
        if request_timeout is not None
        else (protocol.request_timeout if protocol else DEFAULT_REQUEST_TIMEOUT)
    )
    eff_run_timeout = (
        run_timeout
        if run_timeout is not None
        else (protocol.run_timeout if protocol else DEFAULT_RUN_TIMEOUT)
    )
    eff_oracle_timeout = (
        oracle_timeout
        if oracle_timeout is not None
        else (protocol.oracle_timeout if protocol else 60.0)
    )
    eff_setup_timeout = (
        setup_timeout
        if setup_timeout is not None
        else (protocol.setup_timeout if protocol else 120.0)
    )
    eff_cleanup = (
        cleanup_workspaces
        if cleanup_workspaces is not None
        else (protocol.cleanup_workspaces if protocol else True)
    )
    eff_budget = token_budget or (
        TokenBudget(
            context_window=protocol.context_window,
            output_reserve=protocol.output_reserve,
        )
        if protocol is not None
        else None
    )

    conditions_out: dict[str, SuiteSummary] = {}
    for cond in FOUR_WAY_CONDITIONS:
        cond_artifacts = artifacts_root / cond.name if artifacts_root else None
        summary = evaluate_suite(
            manifests,
            work_root=work_root / cond.name,
            model=model,
            artifacts_root=cond_artifacts,
            db_path=db_path,
            policy=policy,
            max_model_calls=eff_calls,
            max_steps=eff_steps,
            max_consecutive_invalid_actions=eff_invalid,
            request_timeout=eff_req_timeout,
            run_timeout=eff_run_timeout,
            oracle_timeout=eff_oracle_timeout,
            setup_timeout=eff_setup_timeout,
            cleanup_workspaces=eff_cleanup,
            experiment_config=experiment_config,
            server_pid=server_pid,
            collect_memory=collect_memory,
            context_strategy=cond.context_strategy,
            token_budget=eff_budget,
            token_counter=token_counter,
            max_tool_output_bytes=max_tool_output_bytes,
            retrieval_backend=cond.retrieval_mode,
        )
        conditions_out[cond.name] = summary

    report = FourWayAblationReport(
        suite_digest=compute_suite_digest(manifests),
        conditions=conditions_out,
        protocol=protocol,
    )
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "four_way_ablation.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (artifacts_root / "four_way_ablation.md").write_text(
            report.to_markdown(),
            encoding="utf-8",
        )
    return report
