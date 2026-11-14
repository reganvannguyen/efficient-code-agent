"""Isolated fixture evaluation runner and independent oracle grading."""

from __future__ import annotations

import functools
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness.agent.events import BenchmarkOutcome
from harness.agent.runtime import (
    CODING_SYSTEM_PROMPT,
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RUN_TIMEOUT,
    Model,
    action_schema,
    run_agent,
)
from harness.agent.state import AgentRun, RunStatus
from harness.evaluation.config import ExperimentConfig
from harness.evaluation.manifest import OracleTier, TaskManifest
from harness.evaluation.resources import MemorySnapshot, sample_macos_memory
from harness.evaluation.workspace import (
    EvaluationSetupError,
    PatchApplyError,
    apply_patch,
    capture_workspace_patch,
    prepare_clean_checkout,
    restore_paths,
)
from harness.llama_client import chat
from harness.storage.artifacts import RunArtifacts, write_run_artifacts
from harness.storage.recorder import RunRecorder
from harness.tools.policy import ToolMode, ToolPolicy
from harness.tools.testing import run_tests, validate_test_policy


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of running one task in Copy A and grading its patch in Copy B."""

    task_id: str
    run_id: str | None
    outcome: BenchmarkOutcome
    oracle_tier: OracleTier
    oracle_passed: bool
    candidate_patch: str
    oracle_output: str = ""
    error_detail: str | None = None
    agent_status: RunStatus | None = None
    artifacts: RunArtifacts | None = None
    experiment_config: ExperimentConfig | None = None
    memory_before: MemorySnapshot | None = None
    memory_after: MemorySnapshot | None = None


def _default_agent_policy(manifest: TaskManifest) -> ToolPolicy:
    if manifest.agent_test_targets:
        return ToolPolicy(
            edit_file=ToolMode.AUTO,
            run_tests=ToolMode.AUTO,
            test_targets=manifest.agent_test_targets,
        )
    return ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.DENY,
        test_targets=(".",),
    )


def _run_oracle_tests(
    workspace: Path,
    targets: tuple[str, ...],
    *,
    timeout: float,
) -> tuple[bool, str, bool]:
    """Validate and run declared oracle pytest targets in an isolated checkout."""
    oracle_policy = ToolPolicy(
        edit_file=ToolMode.DENY,
        run_tests=ToolMode.AUTO,
        test_targets=targets,
        test_timeout=timeout,
    )
    try:
        validated = validate_test_policy(workspace, oracle_policy)
    except ValueError as exc:
        raise EvaluationSetupError(f"Invalid oracle test target: {exc}") from exc

    outputs: list[str] = []
    for target in validated:
        result = run_tests(
            workspace,
            target,
            timeout,
            deadline=time.monotonic() + timeout,
        )
        stdout = str(result.data.get("stdout", ""))
        stderr = str(result.data.get("stderr", ""))
        combined = "\n".join(part for part in (stdout, stderr) if part).strip()
        if combined:
            outputs.append(f"[{target}]\n{combined}")
        if result.data.get("timed_out"):
            return False, "\n\n".join(outputs), True
        if not result.ok:
            return False, "\n\n".join(outputs), False

    return True, "\n\n".join(outputs), False


def _bind_chat_defaults(model: Model, policy: ToolPolicy) -> Model:
    """Bind action_schema and CODING_SYSTEM_PROMPT when using harness.llama_client.chat."""
    if model is chat:
        return functools.partial(
            chat,
            response_schema=action_schema(policy),
            system_prompt=CODING_SYSTEM_PROMPT,
        )
    if isinstance(model, functools.partial) and model.func is chat:
        kwargs = dict(model.keywords)
        kwargs.setdefault("response_schema", action_schema(policy))
        kwargs.setdefault("system_prompt", CODING_SYSTEM_PROMPT)
        return functools.partial(chat, *model.args, **kwargs)
    return model


def _write_artifacts_if_configured(
    artifacts_root: Path | None,
    run: AgentRun,
    manifest: TaskManifest,
    *,
    experiment_config: ExperimentConfig | None = None,
    memory_before: MemorySnapshot | None = None,
    memory_after: MemorySnapshot | None = None,
) -> RunArtifacts | None:
    if artifacts_root is None:
        return None
    manifest_payload: dict[str, object] = {
        "task_id": manifest.task_id,
        "repo": manifest.repo,
        "base_commit": manifest.base_commit,
        "oracle_tier": manifest.oracle_tier.value,
    }
    if experiment_config is not None:
        manifest_payload["experiment_config"] = experiment_config.to_dict()
    if memory_before is not None or memory_after is not None:
        manifest_payload["resources"] = {
            "memory_before": memory_before.to_dict() if memory_before else None,
            "memory_after": memory_after.to_dict() if memory_after else None,
        }
    return write_run_artifacts(
        artifacts_root / run.run_id,
        run,
        manifest=manifest_payload,
    )


def evaluate_task(
    manifest: TaskManifest,
    *,
    work_root: Path,
    model: Model = chat,
    artifacts_root: Path | None = None,
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
) -> EvaluationResult:
    """Run one task in a fresh agent checkout and grade its patch in a separate checkout."""
    trial_dir = work_root / f"{manifest.task_id}_{uuid.uuid4().hex[:8]}"
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

        effective_policy = (
            policy if policy is not None else _default_agent_policy(manifest)
        )
        bound_model = _bind_chat_defaults(model, effective_policy)
        memory_before = (
            sample_macos_memory(server_pid=server_pid) if collect_memory else None
        )
        run_manifest: dict[str, object] = {
            "task_id": manifest.task_id,
            "base_commit": manifest.base_commit,
            "oracle_tier": manifest.oracle_tier.value,
        }
        if experiment_config is not None:
            run_manifest["experiment_config"] = experiment_config.to_dict()

        run = run_agent(
            manifest.agent_prompt(),
            agent_workspace,
            model=bound_model,
            max_model_calls=max_model_calls,
            max_steps=max_steps,
            request_timeout=request_timeout,
            run_timeout=run_timeout,
            policy=effective_policy,
            recorder=recorder,
            manifest=run_manifest,
        )
        memory_after = (
            sample_macos_memory(server_pid=server_pid) if collect_memory else None
        )

        def _artifacts() -> RunArtifacts | None:
            return _write_artifacts_if_configured(
                artifacts_root,
                run,
                manifest,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        try:
            candidate_patch = capture_workspace_patch(agent_workspace)
        except EvaluationSetupError as exc:
            run.benchmark_outcome = BenchmarkOutcome.SETUP_ERROR
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=str(exc),
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        if run.status in (RunStatus.FAILED, RunStatus.CANCELLED):
            run.benchmark_outcome = BenchmarkOutcome.AGENT_ERROR
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=BenchmarkOutcome.AGENT_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=run.stop_reason,
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        if not candidate_patch.strip():
            outcome = (
                BenchmarkOutcome.TIMEOUT
                if run.status == RunStatus.LIMIT_REACHED
                else BenchmarkOutcome.UNRESOLVED
            )
            run.benchmark_outcome = outcome
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=outcome,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch="",
                error_detail=run.stop_reason or "Agent produced no changes.",
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        try:
            eval_workspace = prepare_clean_checkout(
                manifest.repo,
                manifest.base_commit,
                eval_dir,
                setup_commands=manifest.setup_commands,
                timeout=setup_timeout,
            )
        except EvaluationSetupError as exc:
            run.benchmark_outcome = BenchmarkOutcome.SETUP_ERROR
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=str(exc),
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        try:
            apply_patch(eval_workspace, candidate_patch)
        except PatchApplyError as exc:
            run.benchmark_outcome = BenchmarkOutcome.UNRESOLVED
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=BenchmarkOutcome.UNRESOLVED,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=str(exc),
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        try:
            restore_paths(eval_workspace, manifest.oracle_tests)
            oracle_passed, oracle_output, oracle_timed_out = _run_oracle_tests(
                eval_workspace,
                manifest.oracle_tests,
                timeout=oracle_timeout,
            )
        except EvaluationSetupError as exc:
            run.benchmark_outcome = BenchmarkOutcome.SETUP_ERROR
            artifacts = _artifacts()
            return EvaluationResult(
                task_id=manifest.task_id,
                run_id=run.run_id,
                outcome=BenchmarkOutcome.SETUP_ERROR,
                oracle_tier=manifest.oracle_tier,
                oracle_passed=False,
                candidate_patch=candidate_patch,
                error_detail=str(exc),
                agent_status=run.status,
                artifacts=artifacts,
                experiment_config=experiment_config,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        if oracle_passed:
            outcome = BenchmarkOutcome.RESOLVED
        elif oracle_timed_out:
            outcome = BenchmarkOutcome.TIMEOUT
        else:
            outcome = BenchmarkOutcome.UNRESOLVED

        run.benchmark_outcome = outcome
        artifacts = _artifacts()
        return EvaluationResult(
            task_id=manifest.task_id,
            run_id=run.run_id,
            outcome=outcome,
            oracle_tier=manifest.oracle_tier,
            oracle_passed=oracle_passed,
            candidate_patch=candidate_patch,
            oracle_output=oracle_output,
            agent_status=run.status,
            artifacts=artifacts,
            experiment_config=experiment_config,
            memory_before=memory_before,
            memory_after=memory_after,
        )
    finally:
        if cleanup_workspaces and trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
