"""Fixture-based evaluation runner and independent oracle."""

from harness.evaluation.manifest import ManifestError, OracleTier, TaskManifest
from harness.evaluation.runner import EvaluationResult, evaluate_task
from harness.evaluation.workspace import (
    EvaluationSetupError,
    PatchApplyError,
    apply_patch,
    capture_workspace_patch,
    prepare_clean_checkout,
)

__all__ = [
    "EvaluationResult",
    "EvaluationSetupError",
    "ManifestError",
    "OracleTier",
    "PatchApplyError",
    "TaskManifest",
    "apply_patch",
    "capture_workspace_patch",
    "evaluate_task",
    "prepare_clean_checkout",
]
