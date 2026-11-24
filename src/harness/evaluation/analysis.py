"""Repeated-trial execution and statistical/paired analysis for Phase 7 (Issue #42)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.agent.runtime import Model
from harness.context import TokenBudget, TokenCounter
from harness.evaluation.ablation import (
    FOUR_WAY_CONDITIONS,
    FourWayAblationReport,
    run_four_way_ablation,
)
from harness.evaluation.config import ExperimentConfig
from harness.evaluation.manifest import TaskManifest
from harness.evaluation.protocol import FrozenEvaluationProtocol
from harness.evaluation.runner import EvaluationResult
from harness.llama_client import chat
from harness.tools.policy import ToolPolicy


def _wilson_ci_95(successes: int, total: int) -> tuple[float, float]:
    """Return 95% Wilson score confidence interval in [0.0, 1.0]."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _mean_or_none(values: Sequence[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


@dataclass(frozen=True)
class ConditionAggregate:
    """Aggregated accuracy, latency, token, and memory metrics for one condition."""

    condition: str
    trials: int
    total_runs: int
    eligible_runs: int
    resolved_runs: int
    setup_error_runs: int
    resolution_rate_total: float
    resolution_rate_eligible: float
    wilson_ci_95_total: tuple[float, float]
    mean_model_calls: float
    mean_tool_calls: float
    mean_duration_seconds: float
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    mean_context_edits: float
    mean_context_pruned: float
    mean_retrieval_ms: float
    mean_retrieval_parses: float
    mean_rss_after_mib: float | None
    mean_swap_used_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairedTaskComparison:
    """Paired outcome comparison of one condition against baseline across tasks."""

    condition: str
    improved_tasks: tuple[str, ...]
    regressed_tasks: tuple[str, ...]
    unchanged_resolved: tuple[str, ...]
    unchanged_unresolved: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "improved_tasks": list(self.improved_tasks),
            "regressed_tasks": list(self.regressed_tasks),
            "unchanged_resolved": list(self.unchanged_resolved),
            "unchanged_unresolved": list(self.unchanged_unresolved),
        }


@dataclass(frozen=True)
class MultiTrialAnalysis:
    """Complete multi-trial ablation analysis with paired deltas and uncertainty."""

    suite_digest: str
    trials: int
    seeds: tuple[int, ...]
    unique_tasks: int
    per_task_pp_step: float
    conditions: dict[str, ConditionAggregate]
    paired_vs_baseline: dict[str, PairedTaskComparison]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_digest": self.suite_digest,
            "trials": self.trials,
            "seeds": list(self.seeds),
            "unique_tasks": self.unique_tasks,
            "per_task_pp_step": self.per_task_pp_step,
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
            "paired_vs_baseline": {
                k: v.to_dict() for k, v in self.paired_vs_baseline.items()
            },
        }

    def to_markdown(self) -> str:
        lines = [
            "# Phase 7 Multi-Trial Ablation Analysis",
            "",
            f"- **Suite digest:** `{self.suite_digest[:12]}` | **Trials/condition:** {self.trials} | **Tasks/trial:** {self.unique_tasks} ({self.per_task_pp_step:.1f} pp per task)",
            f"- **Seeds:** {', '.join(str(s) for s in self.seeds)}",
            "- **Hardware note:** Fewer prompt tokens do not automatically reduce allocated KV-cache RAM when `llama-server` reserves memory for a fixed context window; missing memory metrics are reported as `-`.",
            "",
            "| Condition | Resolved (Total) | Resolved (Eligible) | 95% Wilson CI | Mean Calls | Mean Tokens | Mean Edits | Retrieval ms | Mean Wall (s) | RSS After (MiB) | Swap (MiB) |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cond in FOUR_WAY_CONDITIONS:
            agg = self.conditions.get(cond.name)
            if agg is None:
                continue
            ci_lo, ci_hi = agg.wilson_ci_95_total
            tok = (
                f"{agg.mean_prompt_tokens:.0f}"
                if agg.mean_prompt_tokens is not None
                else "-"
            )
            rss = (
                f"{agg.mean_rss_after_mib:.1f}"
                if agg.mean_rss_after_mib is not None
                else "-"
            )
            swap = (
                f"{agg.mean_swap_used_mib:.1f}"
                if agg.mean_swap_used_mib is not None
                else "-"
            )
            lines.append(
                f"| `{agg.condition}` | {agg.resolved_runs}/{agg.total_runs} ({agg.resolution_rate_total:.1%}) "
                f"| {agg.resolved_runs}/{agg.eligible_runs} ({agg.resolution_rate_eligible:.1%}) "
                f"| [{ci_lo:.1%}, {ci_hi:.1%}] "
                f"| {agg.mean_model_calls:.1f} | {tok} | {agg.mean_context_edits:.1f} "
                f"| {agg.mean_retrieval_ms:.1f} | {agg.mean_duration_seconds:.2f} "
                f"| {rss} | {swap} |"
            )

        if self.paired_vs_baseline:
            lines.extend(["", "## Paired task comparisons vs `baseline`", ""])
            for name, paired in self.paired_vs_baseline.items():
                lines.append(
                    f"- **`{name}`:** improved={list(paired.improved_tasks)}, "
                    f"regressed={list(paired.regressed_tasks)}, "
                    f"unchanged_resolved={len(paired.unchanged_resolved)}, "
                    f"unchanged_unresolved={len(paired.unchanged_unresolved)}"
                )
        return "\n".join(lines) + "\n"


def aggregate_trial_reports(
    reports: Sequence[FourWayAblationReport],
    *,
    seeds: Sequence[int] | None = None,
) -> MultiTrialAnalysis:
    """Aggregate multiple FourWayAblationReport trials with compatibility checks."""
    if not reports:
        raise ValueError("reports must not be empty")
    expected_digest = reports[0].suite_digest
    for idx, rep in enumerate(reports[1:], start=1):
        if rep.suite_digest != expected_digest:
            raise ValueError(
                f"Trial {idx} suite_digest ({rep.suite_digest}) does not match trial 0 ({expected_digest})"
            )

    trials = len(reports)
    resolved_seeds = tuple(seeds) if seeds is not None else tuple(range(trials))
    if len(resolved_seeds) != trials:
        raise ValueError("seeds length must match number of trial reports")

    first_results = next(iter(reports[0].conditions.values())).results
    unique_tasks = len(first_results)
    pp_step = (100.0 / unique_tasks) if unique_tasks > 0 else 0.0

    per_cond_results: dict[str, list[EvaluationResult]] = {
        cond.name: [] for cond in FOUR_WAY_CONDITIONS
    }
    for rep in reports:
        for cond in FOUR_WAY_CONDITIONS:
            summary = rep.conditions.get(cond.name)
            if summary is not None:
                per_cond_results[cond.name].extend(summary.results)

    aggregates: dict[str, ConditionAggregate] = {}
    for cond in FOUR_WAY_CONDITIONS:
        items = per_cond_results[cond.name]
        if not items:
            continue
        total = len(items)
        setup_errors = sum(
            1 for r in items if r.outcome is BenchmarkOutcome.SETUP_ERROR
        )
        eligible = total - setup_errors
        resolved = sum(1 for r in items if r.outcome is BenchmarkOutcome.RESOLVED)
        rate_total = resolved / total if total > 0 else 0.0
        rate_eligible = resolved / eligible if eligible > 0 else 0.0
        aggregates[cond.name] = ConditionAggregate(
            condition=cond.name,
            trials=trials,
            total_runs=total,
            eligible_runs=eligible,
            resolved_runs=resolved,
            setup_error_runs=setup_errors,
            resolution_rate_total=rate_total,
            resolution_rate_eligible=rate_eligible,
            wilson_ci_95_total=_wilson_ci_95(resolved, total),
            mean_model_calls=sum(r.model_calls for r in items) / total,
            mean_tool_calls=sum(r.tool_calls for r in items) / total,
            mean_duration_seconds=sum(r.duration_seconds for r in items) / total,
            mean_prompt_tokens=_mean_or_none([r.prompt_tokens for r in items]),
            mean_completion_tokens=_mean_or_none([r.completion_tokens for r in items]),
            mean_context_edits=sum(r.context_edit_count for r in items) / total,
            mean_context_pruned=sum(r.context_pruned_count for r in items) / total,
            mean_retrieval_ms=sum(r.retrieval_duration_ms for r in items) / total,
            mean_retrieval_parses=sum(r.retrieval_parse_count for r in items) / total,
            mean_rss_after_mib=_mean_or_none(
                [
                    r.memory_after.process_rss_bytes / (1024 * 1024)
                    if r.memory_after and r.memory_after.process_rss_bytes is not None
                    else None
                    for r in items
                ]
            ),
            mean_swap_used_mib=_mean_or_none(
                [
                    r.memory_after.swap_used_bytes / (1024 * 1024)
                    if r.memory_after and r.memory_after.swap_used_bytes is not None
                    else None
                    for r in items
                ]
            ),
        )

    paired: dict[str, PairedTaskComparison] = {}
    base_items = per_cond_results.get("baseline", [])
    if base_items:
        base_rates: dict[str, float] = {}
        for r in base_items:
            base_rates.setdefault(r.task_id, 0.0)
        task_ids = sorted({r.task_id for r in base_items})
        for tid in task_ids:
            tid_runs = [r for r in base_items if r.task_id == tid]
            base_rates[tid] = sum(
                1 for r in tid_runs if r.outcome is BenchmarkOutcome.RESOLVED
            ) / len(tid_runs)

        for cond in FOUR_WAY_CONDITIONS:
            if cond.name == "baseline" or not per_cond_results[cond.name]:
                continue
            improved: list[str] = []
            regressed: list[str] = []
            unchanged_ok: list[str] = []
            unchanged_fail: list[str] = []
            cond_items = per_cond_results[cond.name]
            for tid in task_ids:
                tid_runs = [r for r in cond_items if r.task_id == tid]
                rate = sum(
                    1 for r in tid_runs if r.outcome is BenchmarkOutcome.RESOLVED
                ) / len(tid_runs)
                b_rate = base_rates[tid]
                if rate > b_rate:
                    improved.append(tid)
                elif rate < b_rate:
                    regressed.append(tid)
                elif rate > 0.0:
                    unchanged_ok.append(tid)
                else:
                    unchanged_fail.append(tid)
            paired[cond.name] = PairedTaskComparison(
                condition=cond.name,
                improved_tasks=tuple(improved),
                regressed_tasks=tuple(regressed),
                unchanged_resolved=tuple(unchanged_ok),
                unchanged_unresolved=tuple(unchanged_fail),
            )

    return MultiTrialAnalysis(
        suite_digest=expected_digest,
        trials=trials,
        seeds=resolved_seeds,
        unique_tasks=unique_tasks,
        per_task_pp_step=pp_step,
        conditions=aggregates,
        paired_vs_baseline=paired,
    )


def run_repeated_ablations(
    manifests: Sequence[TaskManifest],
    *,
    trials: int = 3,
    seeds: Sequence[int] | None = None,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
    db_path: Path | None = None,
    policy: ToolPolicy | None = None,
    protocol: FrozenEvaluationProtocol | None = None,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
    token_budget: TokenBudget | None = None,
    token_counter: TokenCounter | Callable[[str], int] | None = None,
    max_tool_output_bytes: int | None = None,
) -> tuple[list[FourWayAblationReport], MultiTrialAnalysis]:
    """Run ``trials`` independent four-way ablations and return aggregated analysis."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    resolved_seeds = tuple(seeds) if seeds is not None else tuple(range(trials))
    if len(resolved_seeds) != trials:
        raise ValueError("seeds length must equal trials")

    reports: list[FourWayAblationReport] = []
    for idx in range(trials):
        trial_artifacts = artifacts_root / f"trial_{idx}" if artifacts_root else None
        report = run_four_way_ablation(
            manifests,
            work_root=work_root / f"trial_{idx}",
            model=model,
            artifacts_root=trial_artifacts,
            db_path=db_path,
            policy=policy,
            protocol=protocol,
            experiment_config=experiment_config,
            server_pid=server_pid,
            collect_memory=collect_memory,
            token_budget=token_budget,
            token_counter=token_counter,
            max_tool_output_bytes=max_tool_output_bytes,
        )
        reports.append(report)

    analysis = aggregate_trial_reports(reports, seeds=resolved_seeds)
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "multi_trial_analysis.json").write_text(
            json.dumps(analysis.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (artifacts_root / "multi_trial_analysis.md").write_text(
            analysis.to_markdown(),
            encoding="utf-8",
        )
    return reports, analysis
