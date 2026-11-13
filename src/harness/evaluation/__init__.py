"""Fixture-based evaluation runner, configuration pinning, and resource sampling."""

from harness.evaluation.config import (
    ExperimentConfig,
    ModelSpec,
    capture_git_commit,
    capture_host_info,
    compute_file_sha256,
)
from harness.evaluation.manifest import ManifestError, OracleTier, TaskManifest
from harness.evaluation.resources import (
    MemorySnapshot,
    parse_memory_pressure,
    parse_ps_rss_kib,
    parse_sysctl_swapusage,
    sample_macos_memory,
)
from harness.evaluation.runner import EvaluationResult, evaluate_task
from harness.evaluation.workspace import (
    EvaluationSetupError,
    PatchApplyError,
    apply_patch,
    capture_workspace_patch,
    prepare_clean_checkout,
    restore_paths,
)

__all__ = [
    "EvaluationResult",
    "EvaluationSetupError",
    "ExperimentConfig",
    "ManifestError",
    "MemorySnapshot",
    "ModelSpec",
    "OracleTier",
    "PatchApplyError",
    "TaskManifest",
    "apply_patch",
    "capture_git_commit",
    "capture_host_info",
    "capture_workspace_patch",
    "compute_file_sha256",
    "evaluate_task",
    "parse_memory_pressure",
    "parse_ps_rss_kib",
    "parse_sysctl_swapusage",
    "prepare_clean_checkout",
    "restore_paths",
    "sample_macos_memory",
]
