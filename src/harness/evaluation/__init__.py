"""Fixture-based evaluation runner, configuration pinning, and resource sampling."""

from harness.evaluation.config import (
    ExperimentConfig,
    ModelSpec,
    capture_git_commit,
    capture_host_info,
    compute_file_sha256,
)
from harness.evaluation.manifest import ManifestError, OracleTier, TaskManifest
from harness.evaluation.protocol import (
    FrozenEvaluationProtocol,
    compute_suite_digest,
    verify_agent_prompt_isolation,
    verify_suite_partition_isolation,
)
from harness.evaluation.resources import (
    MemorySnapshot,
    parse_memory_pressure,
    parse_ps_rss_kib,
    parse_sysctl_swapusage,
    sample_macos_memory,
)
from harness.evaluation.runner import EvaluationResult, evaluate_task
from harness.evaluation.suite import (
    RetrievalComparisonReport,
    StrategyComparisonReport,
    SuiteSummary,
    compare_retrieval_modes,
    compare_strategies,
    evaluate_suite,
    load_suite_dir,
)
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
    "FrozenEvaluationProtocol",
    "ManifestError",
    "MemorySnapshot",
    "ModelSpec",
    "OracleTier",
    "PatchApplyError",
    "RetrievalComparisonReport",
    "StrategyComparisonReport",
    "SuiteSummary",
    "TaskManifest",
    "apply_patch",
    "capture_git_commit",
    "capture_host_info",
    "capture_workspace_patch",
    "compare_retrieval_modes",
    "compare_strategies",
    "compute_file_sha256",
    "compute_suite_digest",
    "evaluate_suite",
    "evaluate_task",
    "load_suite_dir",
    "parse_memory_pressure",
    "parse_ps_rss_kib",
    "parse_sysctl_swapusage",
    "prepare_clean_checkout",
    "restore_paths",
    "sample_macos_memory",
    "verify_agent_prompt_isolation",
    "verify_suite_partition_isolation",
]
