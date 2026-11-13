"""Clean-checkout workspace management for isolated benchmark evaluation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class EvaluationSetupError(RuntimeError):
    """Raised when preparing a clean checkout or environment setup fails."""


class PatchApplyError(ValueError):
    """Raised when a candidate patch cannot be applied to a clean evaluator checkout."""


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationSetupError(f"git {' '.join(args)} failed: {exc}") from exc


def prepare_clean_checkout(
    repo: str,
    base_commit: str,
    dest_dir: Path,
    *,
    setup_commands: tuple[tuple[str, ...], ...] = (),
    timeout: float = 120.0,
) -> Path:
    """Create a fresh, isolated checkout of ``repo`` at ``base_commit`` in ``dest_dir``.

    Any existing directory at ``dest_dir`` is removed first so prior edits or
    working notes cannot leak into a new trial.
    """
    if repo.startswith("-"):
        raise EvaluationSetupError(f"Invalid repository specification: {repo!r}")
    if base_commit.startswith("-"):
        raise EvaluationSetupError(
            f"Invalid base commit specification: {base_commit!r}"
        )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    clone = _run_git(
        ["clone", "--no-checkout", "--", repo, str(dest_dir)],
        timeout=timeout,
    )
    if clone.returncode != 0:
        raise EvaluationSetupError(
            f"Failed to clone '{repo}': {clone.stderr.strip() or clone.stdout.strip()}"
        )

    checkout = _run_git(
        ["-C", str(dest_dir), "checkout", "--detach", base_commit],
        timeout=timeout,
    )
    if checkout.returncode != 0:
        raise EvaluationSetupError(
            f"Failed to checkout '{base_commit}' in '{dest_dir}': "
            f"{checkout.stderr.strip() or checkout.stdout.strip()}"
        )

    clean = _run_git(["-C", str(dest_dir), "clean", "-fdx"], timeout=timeout)
    if clean.returncode != 0:
        raise EvaluationSetupError(
            f"Failed to clean '{dest_dir}': {clean.stderr.strip() or clean.stdout.strip()}"
        )

    for argv in setup_commands:
        try:
            proc = subprocess.run(
                list(argv),
                cwd=dest_dir,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvaluationSetupError(
                f"Setup command {list(argv)!r} failed: {exc}"
            ) from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise EvaluationSetupError(
                f"Setup command {list(argv)!r} exited with {proc.returncode}: {detail}"
            )

    return dest_dir.resolve()


def capture_workspace_patch(workspace: Path, *, timeout: float = 30.0) -> str:
    """Return a unified diff of all tracked and untracked changes against HEAD."""
    intent = _run_git(["-C", str(workspace), "add", "-N", "."], timeout=timeout)
    if intent.returncode != 0:
        raise EvaluationSetupError(
            f"Failed to stage untracked files in '{workspace}': {intent.stderr.strip()}"
        )
    diff = _run_git(["-C", str(workspace), "diff", "HEAD"], timeout=timeout)
    if diff.returncode != 0:
        raise EvaluationSetupError(
            f"Failed to capture diff in '{workspace}': {diff.stderr.strip()}"
        )
    return diff.stdout


def apply_patch(
    workspace: Path,
    patch_text: str,
    *,
    timeout: float = 30.0,
) -> None:
    """Apply a unified diff patch to a clean evaluator workspace."""
    if not patch_text.strip():
        return
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", "-"],
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PatchApplyError(f"Failed to run git apply: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise PatchApplyError(f"Candidate patch did not apply cleanly: {detail}")


def restore_paths(
    workspace: Path,
    paths: tuple[str, ...],
    *,
    timeout: float = 30.0,
) -> None:
    """Restore oracle test paths to HEAD so candidate patches cannot alter the oracle."""
    targets = [p for p in paths if p and p != "."]
    if not targets:
        return
    proc = _run_git(
        ["-C", str(workspace), "checkout", "HEAD", "--", *targets],
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise EvaluationSetupError(
            f"Failed to restore oracle paths {targets!r}: {detail}"
        )
