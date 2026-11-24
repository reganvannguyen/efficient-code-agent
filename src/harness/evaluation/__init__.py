"""Fixture-based evaluation runner, configuration pinning, and resource sampling."""

from harness.evaluation.ablation import (
    FOUR_WAY_CONDITIONS,
    AblationCondition,
    FourWayAblationReport,
    run_four_way_ablation,
)
from harness.evaluation.analysis import (
    ConditionAggregate,
    MultiTrialAnalysis,
    PairedTaskComparison,
    aggregate_trial_reports,
    run_repeated_ablations,
)
from harness.evaluation.config import (
    ExperimentConfig,
    ModelSpec,
    capture_git_commit,
    capture_host_info,
    compute_file_sha256,
)
from harness.evaluation.external import (
    ExternalComparisonReport,
    ExternalHarnessSpec,
    compare_with_external_harness,
    default_aider_spec,
    default_mini_swe_agent_spec,
    evaluate_external_task,
)
from harness.evaluation.manifest import ManifestError, OracleTier, TaskManifest
from harness.evaluation.protocol import (
    FrozenEvaluationProtocol,
    compute_suite_digest,
    verify_agent_prompt_isolation,
    verify_suite_partition_isolation,
)
from harness.evaluation.reporting import (
    build_final_benchmark_report,
    recheck_saved_patch,
    validate_report_traceability,
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
    "AblationCondition",
    "ConditionAggregate",
    "EvaluationResult",
    "EvaluationSetupError",
    "ExperimentConfig",
    "ExternalComparisonReport",
    "ExternalHarnessSpec",
    "FOUR_WAY_CONDITIONS",
    "FourWayAblationReport",
    "FrozenEvaluationProtocol",
    "ManifestError",
    "MemorySnapshot",
    "ModelSpec",
    "MultiTrialAnalysis",
    "OracleTier",
    "PairedTaskComparison",
    "PatchApplyError",
    "RetrievalComparisonReport",
    "StrategyComparisonReport",
    "SuiteSummary",
    "TaskManifest",
    "aggregate_trial_reports",
    "apply_patch",
    "build_final_benchmark_report",
    "capture_git_commit",
    "capture_host_info",
    "capture_workspace_patch",
    "compare_retrieval_modes",
    "compare_strategies",
    "compare_with_external_harness",
    "compute_file_sha256",
    "compute_suite_digest",
    "default_aider_spec",
    "default_mini_swe_agent_spec",
    "evaluate_external_task",
    "evaluate_suite",
    "evaluate_task",
    "load_suite_dir",
    "parse_memory_pressure",
    "parse_ps_rss_kib",
    "parse_sysctl_swapusage",
    "prepare_clean_checkout",
    "recheck_saved_patch",
    "restore_paths",
    "run_four_way_ablation",
    "run_repeated_ablations",
    "sample_macos_memory",
    "validate_report_traceability",
    "verify_agent_prompt_isolation",
    "verify_suite_partition_isolation",
]
