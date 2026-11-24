"""Deterministic candidate ranking and bounded snippet extraction for Phase 6."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.retrieval.filtering import classify_candidate
from harness.retrieval.python_symbols import PYTHON_EXTENSIONS, PythonSymbol
from harness.retrieval.repo_map import RepoSymbolIndex

DEFAULT_MAX_SNIPPET_LINES = 40
DEFAULT_MAX_SNIPPET_BYTES = 2048
DEFAULT_MAX_TOTAL_BYTES = 12_288
DEFAULT_MAX_PER_FILE = 3
DEFAULT_CONTEXT_LINES = 4

_DEF_KINDS = frozenset(
    {"class", "function", "async_function", "method", "async_method"}
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RankedSnippet:
    """A ranked, bounded, line-numbered code snippet with provenance."""

    path: str
    start_line: int
    end_line: int
    preview: str
    snippet: str
    score: int
    matched_reason: str
    truncated: bool = False
    symbol: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize snippet metadata to a JSON-compatible dictionary."""
        payload: dict[str, object] = {
            "path": self.path,
            "line": self.start_line,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "preview": self.preview,
            "snippet": self.snippet,
            "score": self.score,
            "matched_reason": self.matched_reason,
            "truncated": self.truncated,
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        return payload


@dataclass
class _Candidate:
    path: str
    start_line: int
    end_line: int
    score: int
    matched_reason: str
    symbol: str | None = None
    is_complete_symbol: bool = False


def _query_tokens(query: str) -> tuple[str, ...]:
    tokens = _IDENT_RE.findall(query)
    if not tokens and query.strip():
        return (query.strip(),)
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            ordered.append(tok)
    return tuple(ordered)


def _is_test_path(rel_path: str) -> bool:
    name = Path(rel_path).name.lower()
    parts = Path(rel_path).parts
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or any(p in {"test", "tests"} for p in parts)
    )


def _score_symbol(
    sym: PythonSymbol,
    tokens: tuple[str, ...],
    raw_query: str,
) -> tuple[int, str] | None:
    q_stripped = raw_query.strip()
    name_low = sym.name.lower()
    qname_low = sym.qualified_name.lower()
    if sym.name == q_stripped or any(sym.name == tok for tok in tokens):
        return 100, "exact_symbol_definition"
    if name_low == q_stripped.lower() or any(name_low == tok.lower() for tok in tokens):
        return 90, "exact_symbol_definition"
    if any(tok.lower() in qname_low for tok in tokens):
        return 60, "symbol_definition"
    return None


def _clip_utf8(text: str, byte_limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= byte_limit:
        return text, False
    return encoded[:byte_limit].decode("utf-8", errors="ignore"), True


def _extract_snippet(
    root: Path,
    candidate: _Candidate,
    *,
    max_snippet_lines: int,
    max_snippet_bytes: int,
    context_lines: int,
) -> tuple[str, str, int, int, bool]:
    target = root / candidate.path
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", "", candidate.start_line, candidate.end_line, True

    total = len(lines)
    if total == 0:
        return "", "", 1, 1, False

    start = max(1, min(candidate.start_line, total))
    end = max(start, min(candidate.end_line, total))
    truncated = False

    if end - start + 1 > max_snippet_lines:
        end = start + max_snippet_lines - 1
        truncated = True
    elif not candidate.is_complete_symbol and start == end:
        start = max(1, start - context_lines)
        end = min(total, end + context_lines)
        if start > 1 or end < total:
            truncated = True

    selected = lines[start - 1 : end]
    numbered = "\n".join(f"{start + i}: {line}" for i, line in enumerate(selected))
    bounded, byte_cut = _clip_utf8(numbered, max_snippet_bytes)
    if byte_cut:
        truncated = True
        kept_lines = bounded.splitlines()
        if kept_lines:
            end = start + len(kept_lines) - 1
        bounded = bounded.rstrip("\n") + "\n[...snippet truncated...]"

    preview = (
        lines[candidate.start_line - 1].strip() if candidate.start_line <= total else ""
    )
    return preview, bounded, start, end, truncated


def _merge_file_candidates(items: list[_Candidate]) -> list[_Candidate]:
    if not items:
        return []
    ordered = sorted(items, key=lambda c: (c.start_line, c.end_line, -c.score))
    merged: list[_Candidate] = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start_line <= prev.end_line + 2:
            if cur.score > prev.score:
                best_score = cur.score
                best_reason = cur.matched_reason
                best_symbol = cur.symbol or prev.symbol
            else:
                best_score = prev.score
                best_reason = prev.matched_reason
                best_symbol = prev.symbol or cur.symbol
            merged[-1] = _Candidate(
                path=prev.path,
                start_line=min(prev.start_line, cur.start_line),
                end_line=max(prev.end_line, cur.end_line),
                score=best_score,
                matched_reason=best_reason,
                symbol=best_symbol,
                is_complete_symbol=prev.is_complete_symbol or cur.is_complete_symbol,
            )
        else:
            merged.append(cur)
    return merged


def rank_and_extract_snippets(
    root: Path,
    *,
    query: str,
    lexical_matches: Sequence[dict[str, object]],
    index: RepoSymbolIndex,
    glob: str | None = None,
    max_results: int = 20,
    max_per_file: int = DEFAULT_MAX_PER_FILE,
    max_snippet_lines: int = DEFAULT_MAX_SNIPPET_LINES,
    max_snippet_bytes: int = DEFAULT_MAX_SNIPPET_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_total_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    approved_overrides: tuple[str, ...] = (),
) -> tuple[list[RankedSnippet], list[dict[str, str]], bool]:
    """Combine lexical hits and cached Python symbols into bounded ranked snippets."""
    tokens = _query_tokens(query)
    wants_tests = any("test" in tok.lower() for tok in tokens)
    candidates_by_file: dict[str, list[_Candidate]] = {}
    skipped: list[dict[str, str]] = []
    candidate_files: set[str] = set()

    for match in lexical_matches:
        rel_path = str(match.get("path", ""))
        if not rel_path:
            continue
        if glob is not None and not Path(rel_path).match(glob):
            continue
        reason = classify_candidate(rel_path, approved_overrides=approved_overrides)
        if reason is not None:
            skipped.append({"path": rel_path, "reason": reason})
            continue
        candidate_files.add(rel_path)

    # Also inspect already-cached Python files in the index (bounded)
    for cached_rel in index.cached_paths():
        if glob is None or Path(cached_rel).match(glob):
            candidate_files.add(cached_rel)

    # Index candidate Python files lazily
    symbol_defs_by_file: dict[str, list[PythonSymbol]] = {}
    for rel_path in sorted(candidate_files):
        if Path(rel_path).suffix.lower() not in PYTHON_EXTENSIONS:
            continue
        parsed, _ = index.get_file_symbols(rel_path)
        if parsed is None:
            continue
        defs = [s for s in parsed.symbols if s.kind in _DEF_KINDS]
        symbol_defs_by_file[rel_path] = defs
        for sym in defs:
            scored = _score_symbol(sym, tokens, query)
            if scored is not None:
                score, reason = scored
                if not wants_tests and _is_test_path(rel_path):
                    score -= 5
                if any(tok.lower() in Path(rel_path).stem.lower() for tok in tokens):
                    score += 10
                candidates_by_file.setdefault(rel_path, []).append(
                    _Candidate(
                        path=rel_path,
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        score=score,
                        matched_reason=reason,
                        symbol=sym.qualified_name or sym.name,
                        is_complete_symbol=True,
                    )
                )

    for match in lexical_matches:
        rel_path = str(match.get("path", ""))
        if rel_path not in candidate_files:
            continue
        line = int(match.get("line", 1))
        preview = str(match.get("preview", ""))
        stripped_preview = preview.lstrip()
        is_comment = stripped_preview.startswith("#")
        score = 10 if is_comment else 30
        reason = "comment_mention" if is_comment else "lexical_match"

        if any(tok.lower() in Path(rel_path).stem.lower() for tok in tokens):
            score += 15
        if not wants_tests and _is_test_path(rel_path):
            score -= 5

        enclosing_sym: str | None = None
        for sym in symbol_defs_by_file.get(rel_path, ()):
            if sym.start_line <= line <= sym.end_line:
                enclosing_sym = sym.qualified_name or sym.name
                break

        candidates_by_file.setdefault(rel_path, []).append(
            _Candidate(
                path=rel_path,
                start_line=line,
                end_line=line,
                score=score,
                matched_reason=reason,
                symbol=enclosing_sym,
                is_complete_symbol=False,
            )
        )

    merged_all: list[_Candidate] = []
    per_file_truncated = False
    for rel_path, items in candidates_by_file.items():
        merged = _merge_file_candidates(items)
        merged.sort(key=lambda c: (-c.score, c.start_line, c.end_line))
        if len(merged) > max_per_file:
            per_file_truncated = True
        merged_all.extend(merged[:max_per_file])

    merged_all.sort(key=lambda c: (-c.score, c.path, c.start_line, c.end_line))

    results: list[RankedSnippet] = []
    used_bytes = 0
    used_tokens = 0
    budget_truncated = per_file_truncated or len(merged_all) > max_results

    for cand in merged_all[:max_results]:
        preview, snippet, start, end, snip_truncated = _extract_snippet(
            root,
            cand,
            max_snippet_lines=max_snippet_lines,
            max_snippet_bytes=max_snippet_bytes,
            context_lines=context_lines,
        )
        snip_bytes = len(snippet.encode("utf-8"))
        if results and used_bytes + snip_bytes > max_total_bytes:
            budget_truncated = True
            break
        if max_total_tokens is not None and token_counter is not None:
            snip_tokens = token_counter(snippet)
            if results and used_tokens + snip_tokens > max_total_tokens:
                budget_truncated = True
                break
            used_tokens += snip_tokens
        used_bytes += snip_bytes
        results.append(
            RankedSnippet(
                path=cand.path,
                start_line=start,
                end_line=end,
                preview=preview,
                snippet=snippet,
                score=cand.score,
                matched_reason=cand.matched_reason,
                truncated=snip_truncated,
                symbol=cand.symbol,
            )
        )

    return results, skipped, budget_truncated
