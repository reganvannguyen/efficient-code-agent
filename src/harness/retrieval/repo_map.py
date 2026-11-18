"""Bounded in-memory repository symbol map with fingerprint caching and invalidation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from harness.retrieval.filtering import MAX_CANDIDATE_FILE_BYTES, classify_candidate
from harness.retrieval.python_symbols import (
    PYTHON_EXTENSIONS,
    PythonParseResult,
    extract_python_symbols_from_bytes,
)

DEFAULT_MAX_INDEXED_FILES = 200
DEFAULT_MAP_MAX_BYTES = 4096
DEFAULT_MAX_SYMBOLS_PER_FILE = 25
_DEFINITION_KINDS = frozenset(
    {"class", "function", "async_function", "method", "async_method"}
)


@dataclass(frozen=True)
class CachedFileIndex:
    """Cached parse result for one workspace file keyed by path, stat, and sha256."""

    canonical_path: str
    mtime_ns: int
    size_bytes: int
    sha256: str
    parse_result: PythonParseResult


@dataclass(frozen=True)
class RepoMapResult:
    """Bounded compact repository symbol map and discovery metadata."""

    text: str
    file_count: int
    symbol_count: int
    truncated: bool
    skipped: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize map result to a JSON-compatible dictionary."""
        return {
            "text": self.text,
            "file_count": self.file_count,
            "symbol_count": self.symbol_count,
            "truncated": self.truncated,
            "skipped": list(self.skipped),
        }


class RepoSymbolIndex:
    """In-memory per-run Python symbol index with mtime/hash validation and invalidation."""

    def __init__(
        self,
        root: Path,
        *,
        approved_overrides: tuple[str, ...] = (),
        max_indexed_files: int = DEFAULT_MAX_INDEXED_FILES,
        max_file_bytes: int = MAX_CANDIDATE_FILE_BYTES,
    ) -> None:
        self.root = root.resolve()
        self.approved_overrides = approved_overrides
        self.max_indexed_files = max_indexed_files
        self.max_file_bytes = max_file_bytes
        self._cache: dict[str, CachedFileIndex] = {}
        self.parse_count: int = 0

    def invalidate(self, relative_path: str | None = None) -> None:
        """Invalidate one workspace-relative path or clear the entire cache."""
        if relative_path is None:
            self._cache.clear()
            return
        normalized = Path(relative_path).as_posix().lstrip("./")
        self._cache.pop(normalized, None)

    def cached_paths(self) -> tuple[str, ...]:
        """Return sorted cached workspace-relative paths."""
        return tuple(sorted(self._cache))

    def _confine(self, relative_path: str) -> tuple[str, Path] | None:
        try:
            candidate = (self.root / relative_path).resolve(strict=True)
            rel = candidate.relative_to(self.root).as_posix()
        except (OSError, RuntimeError, ValueError):
            return None
        if not candidate.is_file():
            return None
        return rel, candidate

    def get_file_symbols(
        self, relative_path: str
    ) -> tuple[PythonParseResult | None, str | None]:
        """Return ``(parse_result, skip_reason)`` using cache when file is unchanged."""
        confined = self._confine(relative_path)
        if confined is None:
            self.invalidate(relative_path)
            return None, "not_found_or_outside_workspace"
        rel, target = confined

        try:
            stat = target.stat()
        except OSError:
            self._cache.pop(rel, None)
            return None, "io_error"

        skip_reason = classify_candidate(
            rel,
            size_bytes=stat.st_size,
            approved_overrides=self.approved_overrides,
        )
        if skip_reason is not None:
            self._cache.pop(rel, None)
            return None, skip_reason

        if Path(rel).suffix.lower() not in PYTHON_EXTENSIONS:
            return None, "unsupported_language"

        cached = self._cache.get(rel)
        if (
            cached is not None
            and cached.mtime_ns == stat.st_mtime_ns
            and cached.size_bytes == stat.st_size
        ):
            return cached.parse_result, None

        try:
            with target.open("rb") as handle:
                raw = handle.read(self.max_file_bytes + 1)
        except OSError:
            self._cache.pop(rel, None)
            return None, "io_error"

        digest = hashlib.sha256(raw).hexdigest()
        if cached is not None and cached.sha256 == digest:
            self._cache[rel] = CachedFileIndex(
                canonical_path=rel,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
                sha256=digest,
                parse_result=cached.parse_result,
            )
            return cached.parse_result, None

        parsed = extract_python_symbols_from_bytes(
            rel, raw, max_file_bytes=self.max_file_bytes
        )
        self.parse_count += 1
        self._cache[rel] = CachedFileIndex(
            canonical_path=rel,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            sha256=digest,
            parse_result=parsed,
        )
        return parsed, parsed.fallback_reason

    def _prune_deleted(self) -> None:
        for rel in list(self._cache):
            if self._confine(rel) is None:
                self._cache.pop(rel, None)

    def build_compact_map(
        self,
        paths: Iterable[str] | None = None,
        *,
        max_bytes: int = DEFAULT_MAP_MAX_BYTES,
        max_symbols_per_file: int = DEFAULT_MAX_SYMBOLS_PER_FILE,
    ) -> RepoMapResult:
        """Build a bounded compact map of Python definitions without full-repo bloat."""
        self._prune_deleted()
        if paths is None:
            from harness.tools.repository import list_files

            listed = list_files(self.root, {"pattern": "*.py"})
            raw_paths = listed.data.get("paths")
            candidate_paths = (
                [str(p) for p in raw_paths] if isinstance(raw_paths, list) else []
            )
        else:
            candidate_paths = sorted(set(paths))

        skipped: list[dict[str, str]] = []
        blocks: list[str] = []
        indexed_files = 0
        total_symbols = 0
        truncated = False

        for rel in candidate_paths:
            if indexed_files >= self.max_indexed_files:
                truncated = True
                break
            parsed, reason = self.get_file_symbols(rel)
            if parsed is None:
                if reason is not None:
                    skipped.append({"path": rel, "reason": reason})
                continue
            if reason is not None and not parsed.symbols:
                skipped.append({"path": rel, "reason": reason})
                continue

            indexed_files += 1
            defs = [s for s in parsed.symbols if s.kind in _DEFINITION_KINDS]
            if not defs:
                continue
            file_truncated = len(defs) > max_symbols_per_file
            if file_truncated:
                truncated = True
            shown = defs[:max_symbols_per_file]
            lines = [f"{parsed.path}:"]
            for sym in shown:
                first_sig = (
                    sym.header.splitlines()[0].strip() if sym.header else sym.name
                )
                lines.append(
                    f"  L{sym.start_line}-{sym.end_line} {sym.kind} {sym.qualified_name}: {first_sig}"
                )
                total_symbols += 1
            if file_truncated:
                lines.append("  [...more symbols omitted...]")
            blocks.append("\n".join(lines))

        assembled_blocks: list[str] = []
        used_bytes = 0
        for block in blocks:
            sep_bytes = 1 if assembled_blocks else 0
            block_bytes = len(block.encode("utf-8"))
            if used_bytes + sep_bytes + block_bytes > max_bytes:
                truncated = True
                break
            assembled_blocks.append(block)
            used_bytes += sep_bytes + block_bytes

        text = "\n".join(assembled_blocks)
        if not text and blocks and max_bytes > 0:
            raw = blocks[0].encode("utf-8")[:max_bytes]
            text = raw.decode("utf-8", errors="ignore")
            truncated = True

        return RepoMapResult(
            text=text,
            file_count=indexed_files,
            symbol_count=total_symbols,
            truncated=truncated,
            skipped=tuple(skipped),
        )
