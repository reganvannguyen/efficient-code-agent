"""Sequential benchmark suite runner and summary reporting."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.agent.runtime import (
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RUN_TIMEOUT,
    Model,
)
from harness.evaluation.config import ExperimentConfig
from harness.evaluation.manifest import ManifestError, TaskManifest
from harness.evaluation.runner import EvaluationResult, evaluate_task
from harness.llama_client import chat
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
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 120.0,
    cleanup_workspaces: bool = True,
    experiment_config: ExperimentConfig | None = None,
    server_pid: int | None = None,
    collect_memory: bool = True,
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
            request_timeout=request_timeout,
            run_timeout=run_timeout,
            oracle_timeout=oracle_timeout,
            setup_timeout=setup_timeout,
            cleanup_workspaces=cleanup_workspaces,
            experiment_config=experiment_config,
            server_pid=server_pid,
            collect_memory=collect_memory,
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
