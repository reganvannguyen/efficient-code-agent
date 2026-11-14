"""Task- and session-scoped Markdown working memory with integrity safeguards."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

DEFAULT_MEMORY_BASE_DIR = Path(".harness/runs")
DEFAULT_MAX_NOTE_BYTES = 16_384
_SCOPE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINDING_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.md$")


class MemoryStorageError(ValueError):
    """Raised when a working-memory operation violates path, size, or immutability rules."""


def validate_scope_id(scope_id: str) -> str:
    """Validate a run_id or session_id for safe directory naming."""
    if not isinstance(scope_id, str) or not _SCOPE_ID_RE.match(scope_id):
        raise MemoryStorageError(
            "scope_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
        )
    if ".." in scope_id:
        raise MemoryStorageError("scope_id must not contain '..'")
    return scope_id


def _atomic_write_text(
    target: Path, content: str, *, max_bytes: int | None = None
) -> None:
    encoded = content.encode("utf-8")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise MemoryStorageError(
            f"Note '{target.name}' exceeds byte limit ({len(encoded)} > {max_bytes})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".harness-mem-{secrets.token_hex(8)}"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise


def _default_working_markdown(task: str) -> str:
    return (
        "# Working Context\n\n"
        f"## Goal\n{task.strip()}\n\n"
        "## Confirmed Findings\n\n"
        "## Current Hypothesis\n\n"
        "## Next Action\n\n"
        "## Open Questions\n"
    )


def _default_task_markdown(task: str) -> str:
    return (
        "# Task Progress\n\n"
        f"## Goal Summary\n{task.strip()}\n\n"
        "## Constraints\n\n"
        "## Completed Work\n\n"
        "## Failed Attempts\n\n"
        "## Pending Checks\n\n"
        "## Final Outcome\n"
    )


class WorkingMemoryStore:
    """Isolated Markdown note store under ``<base_dir>/<scope_id>/``."""

    def __init__(
        self,
        base_dir: Path,
        scope_id: str,
        *,
        max_note_bytes: int = DEFAULT_MAX_NOTE_BYTES,
    ) -> None:
        if max_note_bytes <= 0:
            raise MemoryStorageError("max_note_bytes must be positive")
        self.scope_id = validate_scope_id(scope_id)
        self.base_dir = Path(base_dir).resolve()
        self.run_dir = (self.base_dir / self.scope_id).resolve()
        if self.run_dir.parent != self.base_dir:
            raise MemoryStorageError("Resolved run_dir escapes base_dir")
        self.max_note_bytes = max_note_bytes

    @property
    def metadata_path(self) -> Path:
        """Path to immutable run metadata JSON."""
        return self.run_dir / "metadata.json"

    @property
    def working_path(self) -> Path:
        """Path to active working notes."""
        return self.run_dir / "working.md"

    @property
    def task_path(self) -> Path:
        """Path to durable task progress record."""
        return self.run_dir / "task.md"

    @property
    def findings_dir(self) -> Path:
        """Directory for optional selective findings."""
        return self.run_dir / "findings"

    def initialize(
        self,
        task: str,
        *,
        run_id: str,
        session_id: str | None = None,
        policy_summary: dict[str, object] | None = None,
    ) -> None:
        """Create run-scoped memory files if absent without overwriting existing notes."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            metadata: dict[str, object] = {
                "scope_id": self.scope_id,
                "initial_run_id": run_id,
                "session_id": session_id,
                "original_task": task,
                "policy": policy_summary or {},
                "created_at": time.time(),
            }
            _atomic_write_text(
                self.metadata_path,
                json.dumps(metadata, indent=2, sort_keys=True),
                max_bytes=None,
            )
        if not self.working_path.exists():
            _atomic_write_text(
                self.working_path,
                _default_working_markdown(task),
                max_bytes=self.max_note_bytes,
            )
        if not self.task_path.exists():
            _atomic_write_text(
                self.task_path,
                _default_task_markdown(task),
                max_bytes=self.max_note_bytes,
            )

    def read_metadata(self) -> dict[str, object]:
        """Load the immutable run metadata JSON."""
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def resolve_note_path(self, target: str) -> Path:
        """Resolve an editable note target, rejecting escapes and immutable files."""
        if not isinstance(target, str) or not target or "\\" in target:
            raise MemoryStorageError(f"Invalid note target: {target!r}")
        if target.startswith("/") or ".." in Path(target).parts:
            raise MemoryStorageError(f"Note target escapes memory root: {target!r}")
        if target in {"working.md", "task.md"}:
            candidate = self.run_dir / target
        elif target.startswith("findings/") and _FINDING_SLUG_RE.match(
            target.removeprefix("findings/")
        ):
            candidate = self.run_dir / target
        else:
            raise MemoryStorageError(
                f"Unsupported or immutable note target: {target!r}"
            )
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise MemoryStorageError(f"Symlinked note path rejected: {target!r}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.run_dir)
        except ValueError:
            raise MemoryStorageError(
                f"Note target escapes memory root: {target!r}"
            ) from None
        return resolved

    def read_note(self, target: str) -> str:
        """Read a working-memory note, returning '' if not yet created."""
        path = self.resolve_note_path(target)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_note(self, target: str, content: str) -> Path:
        """Atomically replace an editable note within size and path bounds."""
        path = self.resolve_note_path(target)
        _atomic_write_text(path, content, max_bytes=self.max_note_bytes)
        return path

    def append_note(self, target: str, addition: str) -> Path:
        """Atomically append text to an editable note within size and path bounds."""
        existing = self.read_note(target)
        if existing and not existing.endswith("\n") and addition:
            updated = f"{existing}\n{addition}"
        else:
            updated = f"{existing}{addition}"
        return self.write_note(target, updated)

    def write_finding(
        self,
        slug: str,
        *,
        title: str,
        body: str,
        source: str,
        observed_at: str | None = None,
    ) -> Path:
        """Create or replace a structured finding under ``findings/<slug>.md``."""
        filename = slug if slug.endswith(".md") else f"{slug}.md"
        stamp = observed_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        content = (
            f"# {title.strip()}\n\n"
            f"- Source: {source.strip()}\n"
            f"- Observed: {stamp}\n\n"
            f"{body.strip()}\n"
        )
        return self.write_note(f"findings/{filename}", content)

    def list_findings(self) -> list[str]:
        """Return sorted relative ``findings/*.md`` paths."""
        if not self.findings_dir.exists():
            return []
        return sorted(
            f"findings/{path.name}"
            for path in self.findings_dir.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix == ".md"
        )
