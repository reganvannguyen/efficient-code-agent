"""External harness comparison adapter (e.g., mini-SWE-agent) using identical task oracles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.evaluation.manifest import TaskManifest
from harness.evaluation.protocol import FrozenEvaluationProtocol
from harness.evaluation.runner import EvaluationResult, _run_oracle_tests
from harness.evaluation.suite import SuiteSummary
from harness.evaluation.workspace import (
    EvaluationSetupError,
    PatchApplyError,
    apply_patch,
    capture_workspace_patch,
    prepare_clean_checkout,
    restore_paths,
)


@dataclass(frozen=True)
class ExternalHarnessSpec:
    """Explicitly annotated configuration and differences for an external harness."""

    harness_name: str
    pinned_version: str
    invocation_summary: str
    model_name: str
    base_url: str
    context_window: int
    tool_differences: str
    permission_differences: str
    prompt_differences: str
    budget_differences: str
    unmatched_parameters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize external spec to a JSON-compatible dictionary."""
        data = asdict(self)
        data["unmatched_parameters"] = list(self.unmatched_parameters)
        return data


def default_mini_swe_agent_spec(
    *,
    protocol: FrozenEvaluationProtocol | None = None,
    base_url: str = "http://127.0.0.1:9931/v1",
    pinned_version: str = "2.4.6",
) -> ExternalHarnessSpec:
    """Build a documented ExternalHarnessSpec for mini-SWE-agent on local llama.cpp."""
    model_name = (
        protocol.gguf_filename if protocol else "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    )
    ctx = protocol.context_window if protocol else 8192
    return ExternalHarnessSpec(
        harness_name="mini-swe-agent",
        pinned_version=pinned_version,
        invocation_summary=f"uvx --python 3.11 --from mini-swe-agent=={pinned_version} mini",
        model_name=model_name,
        base_url=base_url,
        context_window=ctx,
        tool_differences=(
            "mini-swe-agent uses an unrestricted bash tool in the checkout; "
            "this harness uses bounded JSON-schema tools (list_files, search_code, read_file, edit_file, run_tests)."
        ),
        permission_differences=(
            "mini-swe-agent executes arbitrary shell commands in the isolated checkout; "
            "this harness enforces ToolPolicy and fixed pytest targets."
        ),
        prompt_differences=(
            "mini-swe-agent uses its own bash-agent system prompt and trajectory format; "
            "only the task issue_text and base_commit are shared."
        ),
        budget_differences=(
            "mini-swe-agent uses step/cost limits; this harness enforces max_model_calls, "
            "max_steps, token_budget, and max_consecutive_invalid_actions."
        ),
        unmatched_parameters=(
            "python_runtime>=3.10_via_uvx",
            "system_prompt_and_tool_schema",
            "bash_execution_privileges",
            "internal_context_pruning_rules",
        ),
    )


def default_aider_spec(
    *,
    protocol: FrozenEvaluationProtocol | None = None,
    base_url: str = "http://127.0.0.1:9931/v1",
    pinned_version: str = "0.86.2",
) -> ExternalHarnessSpec:
    """Build a documented ExternalHarnessSpec for Aider on local llama.cpp."""
    model_name = (
        protocol.gguf_filename if protocol else "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    )
    ctx = protocol.context_window if protocol else 8192
    return ExternalHarnessSpec(
        harness_name="aider",
        pinned_version=pinned_version,
        invocation_summary=f"uvx --python 3.12 --from aider-chat=={pinned_version} aider --map-tokens 0",
        model_name=model_name,
        base_url=base_url,
        context_window=ctx,
        tool_differences=(
            "Aider uses whole/search-replace edit blocks, direct file arguments, and --auto-test "
            "(--map-tokens 0 on macOS to avoid scipy==1.15.3 _spropack dlopen errors); "
            "this harness uses bounded JSON-schema tools and working.md context editing."
        ),
        permission_differences=(
            "Aider runs non-interactively with --yes-always --no-auto-commits and an explicit --test-cmd; "
            "this harness enforces ToolPolicy and fixed pytest targets."
        ),
        prompt_differences=(
            "Aider uses its own edit-format system prompt and chat history; "
            "only the task issue_text, solution file, test command, and base_commit are shared."
        ),
        budget_differences=(
            "Aider uses wall-clock timeout and --max-chat-history-tokens; "
            "this harness enforces max_model_calls, max_steps, and token_budget."
        ),
        unmatched_parameters=(
            "python_runtime>=3.10_via_uvx",
            "system_prompt_and_edit_format",
            "aider_repo_map_and_auto_test_loop",
            "internal_context_pruning_rules",
        ),
    )


ExternalRunnerFn = Callable[[TaskManifest, Path, float], tuple[int, str]]


def _default_subprocess_runner(
    command_template: tuple[str, ...],
    *,
    base_url: str,
    model_name: str,
) -> ExternalRunnerFn:
    def _run(
        manifest: TaskManifest, workspace: Path, timeout: float
    ) -> tuple[int, str]:
        solution_file = str(manifest.metadata.get("solution_file", ""))
        test_file = manifest.oracle_tests[0] if manifest.oracle_tests else ""
        argv = [
            part.format(
                task=manifest.agent_prompt(),
                workspace=str(workspace),
                model=model_name,
                base_url=base_url,
                solution_file=solution_file,
                test_file=test_file,
            )
            for part in command_template
        ]
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = base_url
        env["OPENAI_BASE_URL"] = base_url
        env.setdefault("OPENAI_API_KEY", "local-llama")
        env.setdefault("MSWEA_CONFIGURED", "true")
        env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        proc = subprocess.run(
            argv,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = "\n".join(
            part for part in (proc.stdout, proc.stderr) if part
        ).strip()
        return proc.returncode, combined

    return _run


def evaluate_external_task(
    manifest: TaskManifest,
    *,
    work_root: Path,
    spec: ExternalHarnessSpec,
    runner: ExternalRunnerFn | None = None,
    command_template: tuple[str, ...] = (),
    run_timeout: float = 300.0,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 300.0,
    cleanup_workspaces: bool = True,
) -> EvaluationResult:
    """Run an external harness in a clean checkout and grade with the shared oracle."""
    trial_dir = work_root / f"ext_{manifest.task_id}_{uuid.uuid4().hex[:8]}"
    agent_dir = trial_dir / "agent"
    eval_dir = trial_dir / "evaluator"

    try:
        try:
            agent_workspace = prepare_clean_checkout(
                manifest.repo,
                manifest.base_commit,
                agent_dir,
                setup_commands=manifest.setup_commands,
                timeout=setup_timeout,
            )
        except EvaluationSetupError as exc:
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=str(exc),
            )

        active_runner = runner
        if active_runner is None:
            if not command_template:
                return EvaluationResult(
                    task_id=manifest.task_id,
                    run_id=None,
                    outcome=BenchmarkOutcome.SETUP_ERROR,
                    oracle_tier=manifest.oracle_tier,
                    oracle_passed=False,
                    candidate_patch="",
                    error_detail="No external runner or command_template configured.",
                )
            active_runner = _default_subprocess_runner(
                command_template,
                base_url=spec.base_url,
                model_name=spec.model_name,
            )

        start = time.monotonic()
        try:
            exit_code, runner_output = active_runner(
                manifest, agent_workspace, run_timeout
            )
        except FileNotFoundError as exc:
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=f"External harness binary not found: {exc}",
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.TIMEOUT,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=f"External harness timed out after {run_timeout}s",
                duration_seconds=duration,
            )
        duration = time.monotonic() - start

        try:
            candidate_patch = capture_workspace_patch(agent_workspace)
        except EvaluationSetupError as exc:
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=str(exc),
                duration_seconds=duration,
            )

        if exit_code != 0 and not candidate_patch.strip():
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.AGENT_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=f"External harness exited {exit_code}: {runner_output[:240]}",
                duration_seconds=duration,
            )

        if not candidate_patch.strip():
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=None,
                outcome=BenchmarkOutcome.UNRESOLVED,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail="External harness produced no changes.",
                duration_seconds=duration,
            )

        try:
            eval_workspace = prepare_clean_checkout(
                manifest.repo,
                manifest.base_commit,
                eval_dir,
                setup_commands=manifest.setup_commands,
                timeout=setup_timeout,
            )
            apply_patch(eval_workspace, candidate_patch)
            restore_paths(eval_workspace, manifest.oracle_tests)
            oracle_passed, oracle_output, oracle_timed_out = _run_oracle_tests(
                eval_workspace,
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
                duration_seconds=duration,
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
                duration_seconds=duration,
            )

        if oracle_passed:
            outcome = BenchmarkOutcome.RESOLVED
        elif oracle_timed_out:
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
            duration_seconds=duration,
        )
    finally:
        if cleanup_workspaces and trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)


@dataclass(frozen=True)
class ExternalComparisonReport:
    """Comparison between an internal suite run and an external harness."""

    internal_label: str
    internal_summary: SuiteSummary
    external_spec: ExternalHarnessSpec
    external_summary: SuiteSummary
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize external comparison report to a JSON-compatible dictionary."""
        return {
            "comparison_kind": "external_harness_comparison",
            "internal_label": self.internal_label,
            "internal_summary": self.internal_summary.to_dict(),
            "external_spec": self.external_spec.to_dict(),
            "external_summary": self.external_summary.to_dict(),
            "limitations": list(self.limitations),
        }

    def to_markdown(self) -> str:
        """Render a clearly labeled external comparison report with unmatched annotations."""
        i_map = {r.task_id: r for r in self.internal_summary.results}
        e_map = {r.task_id: r for r in self.external_summary.results}
        lines = [
            f"# External Harness Comparison: `{self.internal_label}` vs `{self.external_spec.harness_name}` (`{self.external_spec.pinned_version}`)",
            "",
            "> **Label:** External harness comparison (not a controlled retrieval/context ablation).",
            "",
            f"- **Internal ({self.internal_label}):** {self.internal_summary.resolved_count}/{len(self.internal_summary.results)} resolved",
            f"- **External ({self.external_spec.harness_name}):** {self.external_summary.resolved_count}/{len(self.external_summary.results)} resolved",
            f"- **Unmatched parameters:** {', '.join(self.external_spec.unmatched_parameters) or 'none'}",
            "",
            "| Task ID | Internal Outcome | External Outcome | Wall Time s (Int vs Ext) |",
            "| --- | --- | --- | --- |",
        ]
        for tid in [r.task_id for r in self.internal_summary.results]:
            i_res = i_map[tid]
            e_res = e_map[tid]
            lines.append(
                f"| `{tid}` | `{i_res.outcome.value}` ({i_res.oracle_tier.value}) "
                f"| `{e_res.outcome.value}` ({e_res.oracle_tier.value}) "
                f"| {i_res.duration_seconds:.2f} vs {e_res.duration_seconds:.2f} |"
            )
        if self.limitations:
            lines.extend(["", "## Documented limitations", ""])
            for item in self.limitations:
                lines.append(f"- {item}")
        return "\n".join(lines) + "\n"


def compare_with_external_harness(
    manifests: list[TaskManifest] | tuple[TaskManifest, ...],
    *,
    internal_label: str,
    internal_summary: SuiteSummary,
    external_spec: ExternalHarnessSpec,
    work_root: Path,
    artifacts_root: Path | None = None,
    runner: ExternalRunnerFn | None = None,
    command_template: tuple[str, ...] = (),
    run_timeout: float = 300.0,
    oracle_timeout: float = 60.0,
    setup_timeout: float = 300.0,
    cleanup_workspaces: bool = True,
    limitations: tuple[str, ...] = (),
) -> ExternalComparisonReport:
    """Run an external harness on the same manifests/oracle and compare with an internal summary."""
    ext_results = [
        evaluate_external_task(
            manifest,
            work_root=work_root,
            spec=external_spec,
            runner=runner,
            command_template=command_template,
            run_timeout=run_timeout,
            oracle_timeout=oracle_timeout,
            setup_timeout=setup_timeout,
            cleanup_workspaces=cleanup_workspaces,
        )
        for manifest in manifests
    ]
    ext_summary = SuiteSummary.from_results(ext_results)
    report = ExternalComparisonReport(
        internal_label=internal_label,
        internal_summary=internal_summary,
        external_spec=external_spec,
        external_summary=ext_summary,
        limitations=limitations,
    )
    if artifacts_root is not None:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        (artifacts_root / "external_comparison.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (artifacts_root / "external_comparison.md").write_text(
            report.to_markdown(),
            encoding="utf-8",
        )
    return report
