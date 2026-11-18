"""Candidate filtering and explicit skip reasons for repository retrieval."""

from __future__ import annotations

from collections.abc import Collection
from fnmatch import fnmatch
from pathlib import Path

MAX_CANDIDATE_FILE_BYTES = 1024 * 1024

EXCLUDED_DIRS = frozenset(
    {"node_modules", ".venv", "__pycache__", "dist", "build", ".git"}
)

BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".webp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".gguf",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".woff",
        ".woff2",
        ".ttf",
    }
)

GENERATED_SUFFIXES = (".min.js", ".min.css", ".js.map", ".css.map")
GENERATED_FILENAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Cargo.lock",
    }
)


def classify_candidate(
    relative_path: str,
    *,
    size_bytes: int | None = None,
    sample: bytes | None = None,
    approved_overrides: Collection[str] = (),
) -> str | None:
    """Return a skip reason for a workspace-relative candidate, or None if accepted.

    Security confinement (workspace-root and symlink checks) must already be enforced
    before calling this helper. ``approved_overrides`` may allow specific hidden,
    excluded, or generated paths when task-relevant.
    """
    path_obj = Path(relative_path)
    parts = path_obj.parts
    is_overridden = relative_path in approved_overrides or any(
        fnmatch(relative_path, pattern) or path_obj.match(pattern)
        for pattern in approved_overrides
    )

    if not is_overridden:
        if any(part in EXCLUDED_DIRS for part in parts):
            return "excluded_directory"
        if any(part.startswith(".") for part in parts):
            return "hidden_or_ignored"

    if size_bytes is not None and size_bytes > MAX_CANDIDATE_FILE_BYTES:
        return "file_too_large"

    lower_name = path_obj.name.lower()
    if sample is not None and b"\x00" in sample:
        return "binary_or_unsupported"
    if path_obj.suffix.lower() in BINARY_EXTENSIONS:
        return "binary_or_unsupported"

    if not is_overridden and (
        lower_name in GENERATED_FILENAMES
        or any(lower_name.endswith(suffix) for suffix in GENERATED_SUFFIXES)
        or ".generated." in lower_name
    ):
        return "generated_or_minified"

    return None
