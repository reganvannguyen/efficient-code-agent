"""Patch re-validation, traceability verification, and final benchmark reporting (Issue #43)."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.evaluation.ablation import FOUR_WAY_CONDITIONS, FourWayAblationReport
from harness.evaluation.analysis import MultiTrialAnalysis
from harness.evaluation.external import ExternalComparisonReport
from harness.evaluation.manifest import TaskManifest
from harness.evaluation.protocol import FrozenEvaluationProtocol
from harness.evaluation.runner import EvaluationResult, _run_oracle_tests
from harness.evaluation.workspace import (
    EvaluationSetupError,
    PatchApplyError,
    apply_patch,
    prepare_clean_checkout,
    restore_paths,
)


def recheck_saved_patch(
    manifest: TaskManifest,
    candidate_patch: str,
    *,
    work_root: Path,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 300.0,
    cleanup_workspaces: bool = True,
) -> EvaluationResult:
    """Independently re-apply a saved candidate patch to a clean checkout and grade it."""
    if not candidate_patch.strip():
        return EvaluationResult(
            task_id=manifest.task_id,
            run_id=None,
            outcome=BenchmarkOutcome.UNRESOLVED,
            oracle_tier=manifest.oracle_tier,
            oracle_passed=False,
            candidate_patch="",
            error_detail="Saved candidate patch is empty.",
        )

    eval_dir = work_root / f"recheck_{manifest.task_id}_{uuid.uuid4().hex[:8]}"
    try:
        try:
            workspace = prepare_clean_checkout(
                manifest.repo,
                manifest.base_commit,
                eval_dir,
                setup_commands=manifest.setup_commands,
                timeout=setup_timeout,
            )
            apply_patch(workspace, candidate_patch)
            restore_paths(workspace, manifest.oracle_tests)
            oracle_passed, oracle_output, timed_out = _run_oracle_tests(
                workspace,
                manifest.oracle_tests,
                timeout=oracle_timeout,
            )
        except EvaluationSetupError as exc:
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=str(exc),
            )
        except PatchApplyError as exc:
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.UNRESOLVED,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=str(exc),
            )

        if oracle_passed:
            outcome = BenchmarkOutcome.RESOLVED
        elif timed_out:
            outcome = BenchmarkOutcome.TIMEOUT
        else:
            outcome = BenchmarkOutcome.UNRESOLVED

        return EvaluationResult(
            task_id=manifest.task_id,
            run_id=None,
            outcome=outcome,
            oracle_tier=manifest.oracle_tier,
            oracle_passed=oracle_passed,
            candidate_patch=candidate_patch,
            oracle_output=oracle_output,
        )
    finally:
        if cleanup_workspaces and eval_dir.exists():
            shutil.rmtree(eval_dir, ignore_errors=True)


def validate_report_traceability(
    analysis: MultiTrialAnalysis,
    reports: Sequence[FourWayAblationReport],
) -> dict[str, dict[str, int]]:
    """Verify every aggregate count in ``analysis`` traces back to ``reports`` and separate tiers."""
    if len(reports) != analysis.trials:
        raise ValueError(
            f"Expected {analysis.trials} trial reports, got {len(reports)}"
        )
    tier_counts: dict[str, dict[str, int]] = {}
    for cond in FOUR_WAY_CONDITIONS:
        agg = analysis.conditions.get(cond.name)
        if agg is None:
            continue
        raw_results: list[EvaluationResult] = []
        for rep in reports:
            summary = rep.conditions.get(cond.name)
            if summary is not None:
                raw_results.extend(summary.results)
        if len(raw_results) != agg.total_runs:
            raise ValueError(
                f"Condition '{cond.name}' total_runs mismatch: "
                f"analysis={agg.total_runs} vs raw={len(raw_results)}"
            )
        resolved = sum(1 for r in raw_results if r.outcome is BenchmarkOutcome.RESOLVED)
        setup_errors = sum(
            1 for r in raw_results if r.outcome is BenchmarkOutcome.SETUP_ERROR
        )
        if resolved != agg.resolved_runs or setup_errors != agg.setup_error_runs:
            raise ValueError(
                f"Condition '{cond.name}' outcome counts do not trace to saved runs"
            )
        tiers: dict[str, int] = {}
        for r in raw_results:
            tiers[r.oracle_tier.value] = tiers.get(r.oracle_tier.value, 0) + 1
        tier_counts[cond.name] = tiers
    return tier_counts


def build_final_benchmark_report(
    analysis: MultiTrialAnalysis,
    *,
    protocol: FrozenEvaluationProtocol,
    reports: Sequence[FourWayAblationReport] = (),
    external_report: ExternalComparisonReport | None = None,
    exclusions: Sequence[str] = (),
    conclusion: str = "",
) -> str:
    """Build a reproducible Phase 7 Markdown report with tier separation and caveats."""
    tier_summary = validate_report_traceability(analysis, reports) if reports else {}
    lines = [
        "# Phase 7 Final Benchmark Report",
        "",
        "> **Reporting guardrail:** Local targeted (`local_targeted`) and `local_full` results are **not** official SWE-bench Verified scores.",
        "",
        "## Frozen configuration",
        "",
        f"- **Protocol:** `{protocol.protocol_version}`",
        f"- **Model:** `{protocol.gguf_filename}` (`{protocol.quantization}`, `sha256: {protocol.gguf_sha256}`)",
        f"- **Context & limits:** `context_window={protocol.context_window}`, `output_reserve={protocol.output_reserve}`, `max_model_calls={protocol.max_model_calls}`, `max_steps={protocol.max_steps}`, `max_consecutive_invalid_actions={protocol.max_consecutive_invalid_actions}`",
        f"- **Suite digest:** `{analysis.suite_digest}` ({analysis.unique_tasks} tasks/trial, {analysis.trials} trials/condition, {analysis.per_task_pp_step:.1f} pp per task)",
        "",
        "## Four-way internal ablation results",
        "",
        analysis.to_markdown().strip(),
    ]
    if tier_summary:
        lines.extend(["", "### Oracle tier breakdown (traced to saved runs)", ""])
        for cond_name, counts in tier_summary.items():
            lines.append(f"- `{cond_name}`: {counts}")

    if external_report is not None:
        lines.extend(["", external_report.to_markdown().strip()])

    if exclusions:
        lines.extend(["", "## Exclusions and setup failures", ""])
        for item in exclusions:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Threats to validity and conclusion",
            "",
            conclusion
            or (
                "Results are qualified by tested sample size, local 7B model capacity, "
                "and `local_targeted` oracle scope."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
