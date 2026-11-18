"""Lightweight Python symbol and source-range extraction using Tree-sitter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

from harness.retrieval.filtering import MAX_CANDIDATE_FILE_BYTES

_PY_LANGUAGE = Language(tsp.language())
PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})


@dataclass(frozen=True)
class PythonSymbol:
    """Extracted Python definition or import with 1-based source range and fingerprint."""

    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    header: str
    file_sha256: str
    language: str = "python"
    parent: str | None = None
    qualified_name: str = ""
    uncertain: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize symbol metadata to a JSON-compatible dictionary."""
        return {
            "path": self.path,
            "language": self.language,
            "kind": self.kind,
            "name": self.name,
            "parent": self.parent,
            "qualified_name": self.qualified_name or self.name,
            "header": self.header,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_sha256": self.file_sha256,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True)
class PythonParseResult:
    """Structured outcome of parsing one workspace file for Python symbols."""

    path: str
    language: str | None
    file_sha256: str | None
    symbols: tuple[PythonSymbol, ...]
    has_errors: bool = False
    uncertain: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize parse result to a JSON-compatible dictionary."""
        return {
            "path": self.path,
            "language": self.language,
            "file_sha256": self.file_sha256,
            "symbols": [s.to_dict() for s in self.symbols],
            "has_errors": self.has_errors,
            "uncertain": self.uncertain,
            "fallback_reason": self.fallback_reason,
        }


def parse_python_file(
    root: Path,
    relative_path: str,
    *,
    max_file_bytes: int = MAX_CANDIDATE_FILE_BYTES,
) -> PythonParseResult:
    """Parse one workspace-relative Python file within size and encoding limits."""
    path_obj = Path(relative_path)
    if path_obj.suffix.lower() not in PYTHON_EXTENSIONS:
        return PythonParseResult(
            path=relative_path,
            language=None,
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="unsupported_language",
        )

    target = root / relative_path
    try:
        if not target.is_file():
            return PythonParseResult(
                path=relative_path,
                language="python",
                file_sha256=None,
                symbols=(),
                uncertain=True,
                fallback_reason="not_found",
            )
        with target.open("rb") as handle:
            raw = handle.read(max_file_bytes + 1)
    except OSError:
        return PythonParseResult(
            path=relative_path,
            language="python",
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="io_error",
        )

    return extract_python_symbols_from_bytes(
        relative_path, raw, max_file_bytes=max_file_bytes
    )


def extract_python_symbols_from_bytes(
    relative_path: str,
    raw: bytes,
    *,
    max_file_bytes: int = MAX_CANDIDATE_FILE_BYTES,
) -> PythonParseResult:
    """Extract Python symbols and 1-based source ranges from raw file bytes."""
    if Path(relative_path).suffix.lower() not in PYTHON_EXTENSIONS:
        return PythonParseResult(
            path=relative_path,
            language=None,
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="unsupported_language",
        )
    if len(raw) > max_file_bytes:
        return PythonParseResult(
            path=relative_path,
            language="python",
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="file_too_large",
        )
    if b"\x00" in raw:
        return PythonParseResult(
            path=relative_path,
            language="python",
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="binary_or_encoding_error",
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return PythonParseResult(
            path=relative_path,
            language="python",
            file_sha256=None,
            symbols=(),
            uncertain=True,
            fallback_reason="binary_or_encoding_error",
        )

    digest = hashlib.sha256(raw).hexdigest()
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(raw)
    root_node = tree.root_node
    has_errors = root_node.has_error

    symbols: list[PythonSymbol] = []
    _walk_block(
        root_node,
        raw=raw,
        path=relative_path,
        digest=digest,
        parent_stack=(),
        parent_kind=None,
        out=symbols,
    )
    return PythonParseResult(
        path=relative_path,
        language="python",
        file_sha256=digest,
        symbols=tuple(symbols),
        has_errors=has_errors,
        uncertain=has_errors,
        fallback_reason="syntax_errors" if has_errors else None,
    )


def _node_text(raw: bytes, start_byte: int, end_byte: int) -> str:
    return raw[start_byte:end_byte].decode("utf-8", errors="replace")


def _extract_header(raw: bytes, outer_node: Node, def_node: Node) -> str:
    body = def_node.child_by_field_name("body")
    header_end = body.start_byte if body is not None else def_node.end_byte
    return _node_text(raw, outer_node.start_byte, header_end).strip()


def _is_async_function(def_node: Node) -> bool:
    return any(child.type == "async" for child in def_node.children)


def _walk_block(
    container: Node,
    *,
    raw: bytes,
    path: str,
    digest: str,
    parent_stack: tuple[str, ...],
    parent_kind: str | None,
    out: list[PythonSymbol],
) -> None:
    for child in container.children:
        if child.type == "decorated_definition":
            inner = child.child_by_field_name("definition")
            if inner is not None and inner.type in {
                "function_definition",
                "class_definition",
            }:
                _handle_definition(
                    outer_node=child,
                    def_node=inner,
                    raw=raw,
                    path=path,
                    digest=digest,
                    parent_stack=parent_stack,
                    parent_kind=parent_kind,
                    out=out,
                )
            continue

        if child.type in {"function_definition", "class_definition"}:
            _handle_definition(
                outer_node=child,
                def_node=child,
                raw=raw,
                path=path,
                digest=digest,
                parent_stack=parent_stack,
                parent_kind=parent_kind,
                out=out,
            )
            continue

        if child.type in {"import_statement", "import_from_statement"}:
            text = _node_text(raw, child.start_byte, child.end_byte).strip()
            name = _import_name(child, raw) or text
            parent = ".".join(parent_stack) if parent_stack else None
            out.append(
                PythonSymbol(
                    path=path,
                    kind="import",
                    name=name,
                    parent=parent,
                    qualified_name=f"{parent}.{name}" if parent else name,
                    header=text,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    file_sha256=digest,
                    uncertain=child.has_error,
                )
            )
            continue

        if child.type == "block":
            _walk_block(
                child,
                raw=raw,
                path=path,
                digest=digest,
                parent_stack=parent_stack,
                parent_kind=parent_kind,
                out=out,
            )


def _import_name(node: Node, raw: bytes) -> str | None:
    if node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node is not None:
            return _node_text(raw, module_node.start_byte, module_node.end_byte).strip()
    for child in node.named_children:
        if child.type in {"dotted_name", "aliased_import"}:
            return _node_text(raw, child.start_byte, child.end_byte).strip()
    return None


def _handle_definition(
    *,
    outer_node: Node,
    def_node: Node,
    raw: bytes,
    path: str,
    digest: str,
    parent_stack: tuple[str, ...],
    parent_kind: str | None,
    out: list[PythonSymbol],
) -> None:
    name_node = def_node.child_by_field_name("name")
    name = (
        _node_text(raw, name_node.start_byte, name_node.end_byte).strip()
        if name_node is not None
        else "<anonymous>"
    )
    if def_node.type == "class_definition":
        kind = "class"
    else:
        is_async = _is_async_function(def_node)
        if parent_kind == "class":
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"

    parent = ".".join(parent_stack) if parent_stack else None
    qualified_name = f"{parent}.{name}" if parent else name
    out.append(
        PythonSymbol(
            path=path,
            kind=kind,
            name=name,
            parent=parent,
            qualified_name=qualified_name,
            header=_extract_header(raw, outer_node, def_node),
            start_line=outer_node.start_point[0] + 1,
            end_line=outer_node.end_point[0] + 1,
            file_sha256=digest,
            uncertain=outer_node.has_error,
        )
    )

    body = def_node.child_by_field_name("body")
    if body is not None:
        _walk_block(
            body,
            raw=raw,
            path=path,
            digest=digest,
            parent_stack=(*parent_stack, name),
            parent_kind=kind,
            out=out,
        )
