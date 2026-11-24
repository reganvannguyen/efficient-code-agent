"""Selectable retrieval backend interface and normalized match metadata."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from harness.retrieval.filtering import classify_candidate
from harness.retrieval.ranking import (
    DEFAULT_MAX_PER_FILE,
    DEFAULT_MAX_SNIPPET_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    rank_and_extract_snippets,
)
from harness.retrieval.repo_map import RepoSymbolIndex

if TYPE_CHECKING:
    from harness.tools.dispatcher import ToolResult


@dataclass(frozen=True)
class NormalizedRetrievalMatch:
    """Normalized metadata for a retrieval hit across lexical and code-aware modes."""

    path: str
    start_line: int
    end_line: int
    preview: str
    truncated: bool = False
    matched_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize match metadata to a JSON-compatible dictionary."""
        payload: dict[str, object] = {
            "path": self.path,
            "line": self.start_line,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "preview": self.preview,
            "truncated": self.truncated,
        }
        if self.matched_reason is not None:
            payload["matched_reason"] = self.matched_reason
        return payload


@runtime_checkable
class RetrievalBackend(Protocol):
    """Protocol for swappable repository retrieval behind existing model-facing tools."""

    mode: str

    def list_files(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        """List visible workspace files."""
        ...

    def search_code(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        """Search repository code and return bounded matches."""
        ...

    def read_file(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        """Read a bounded line range from a workspace file."""
        ...


@dataclass(frozen=True)
class LexicalRetrievalBackend:
    """Baseline Phase 3-5 retrieval backend using ripgrep lexical discovery."""

    mode: str = "lexical_only"

    def list_files(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.repository import list_files

        return list_files(root, arguments)

    def search_code(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.search import search_code

        return search_code(root, arguments)

    def read_file(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.repository import read_file

        return read_file(root, arguments)


@dataclass(frozen=True)
class CodeAwareRetrievalBackend:
    """Phase 6 enhanced retrieval boundary preserving model-facing tools and policy.

    Filters binary/generated/huge candidates with explicit skip reasons while allowing
    approved overrides and normalizing match metadata. Symbol parsing and ranking
    (issues #35-#38) plug into this boundary without changing tool names.
    """

    mode: str = "code_aware"
    approved_overrides: tuple[str, ...] = ()
    max_per_file: int = DEFAULT_MAX_PER_FILE
    max_snippet_bytes: int = DEFAULT_MAX_SNIPPET_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_total_tokens: int | None = None
    token_counter: Callable[[str], int] | None = None
    _indexes: dict[Path, RepoSymbolIndex] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_index(self, root: Path) -> RepoSymbolIndex:
        """Return or create the per-run RepoSymbolIndex for ``root``."""
        resolved = root.resolve()
        index = self._indexes.get(resolved)
        if index is None:
            index = RepoSymbolIndex(
                resolved, approved_overrides=self.approved_overrides
            )
            self._indexes[resolved] = index
        return index

    def invalidate(
        self, relative_path: str | None = None, *, root: Path | None = None
    ) -> None:
        """Invalidate cached symbol entries after edits, renames, or deletions."""
        if root is not None:
            index = self._indexes.get(root.resolve())
            if index is not None:
                index.invalidate(relative_path)
            return
        for index in self._indexes.values():
            index.invalidate(relative_path)

    def list_files(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.dispatcher import ToolResult
        from harness.tools.repository import list_files

        result = list_files(root, arguments)
        raw_paths = result.data.get("paths")
        if not isinstance(raw_paths, list):
            return result
        kept, skipped = _filter_paths(
            root, raw_paths, approved_overrides=self.approved_overrides
        )
        data = dict(result.data)
        data["paths"] = kept
        if skipped:
            data["skipped"] = skipped
        return ToolResult(
            result.tool,
            result.ok,
            data,
            truncated=result.truncated,
            message=result.message,
        )

    def search_code(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.dispatcher import ToolResult
        from harness.tools.search import search_code

        max_results = int(arguments.get("max_results", 20))
        broad_args = dict(arguments)
        broad_args["max_results"] = max(max_results * 5, 50)
        result = search_code(root, broad_args)
        raw_matches = result.data.get("matches")
        if not isinstance(raw_matches, list):
            return result
        query = str(arguments.get("query", ""))
        glob_arg = arguments.get("glob")
        glob_str = str(glob_arg) if glob_arg is not None else None
        ranked, skipped, budget_truncated = rank_and_extract_snippets(
            root,
            query=query,
            lexical_matches=[m for m in raw_matches if isinstance(m, dict)],
            index=self.get_index(root),
            glob=glob_str,
            max_results=max_results,
            max_per_file=self.max_per_file,
            max_snippet_bytes=self.max_snippet_bytes,
            max_total_bytes=self.max_total_bytes,
            max_total_tokens=self.max_total_tokens,
            token_counter=self.token_counter,
            approved_overrides=self.approved_overrides,
        )
        data = dict(result.data)
        data["matches"] = [item.to_dict() for item in ranked]
        if skipped:
            data["skipped"] = skipped
        return ToolResult(
            result.tool,
            result.ok,
            data,
            truncated=result.truncated or budget_truncated,
            message=result.message,
        )

    def read_file(self, root: Path, arguments: dict[str, object]) -> ToolResult:
        from harness.tools.repository import read_file

        # Explicit authorized reads remain possible even for hidden/ignored files.
        return read_file(root, arguments)


def _candidate_skip_reason(
    root: Path,
    relative_path: str,
    *,
    approved_overrides: Collection[str],
) -> str | None:
    candidate = root / relative_path
    size_bytes: int | None = None
    try:
        size_bytes = candidate.stat().st_size
    except OSError:
        pass
    return classify_candidate(
        relative_path,
        size_bytes=size_bytes,
        approved_overrides=approved_overrides,
    )


def _filter_paths(
    root: Path,
    paths: list[object],
    *,
    approved_overrides: Collection[str],
) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    skipped: list[dict[str, str]] = []
    for raw in paths:
        rel_path = str(raw)
        reason = _candidate_skip_reason(
            root, rel_path, approved_overrides=approved_overrides
        )
        if reason is not None:
            skipped.append({"path": rel_path, "reason": reason})
        else:
            kept.append(rel_path)
    return kept, skipped


def resolve_retrieval_backend(
    backend: RetrievalBackend | str | None,
) -> RetrievalBackend:
    """Resolve a retrieval mode name or backend instance, defaulting to lexical_only."""
    if backend is None or backend == "lexical_only":
        return LexicalRetrievalBackend()
    if backend == "code_aware":
        return CodeAwareRetrievalBackend()
    if isinstance(backend, RetrievalBackend):
        return backend
    raise ValueError(
        f"Unknown retrieval_backend {backend!r}. "
        "Expected 'lexical_only', 'code_aware', or a RetrievalBackend."
    )
