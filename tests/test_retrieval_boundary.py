"""Tests for Phase 6 retrieval boundary, candidate filtering, and normalized metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agent.events import RunEventType
from harness.agent.protocol import ToolAction
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.llama_client import ChatResponse
from harness.retrieval import (
    CodeAwareRetrievalBackend,
    LexicalRetrievalBackend,
    classify_candidate,
    resolve_retrieval_backend,
)
from harness.tools.dispatcher import execute_tool


def test_resolve_retrieval_backend_defaults_to_lexical_and_rejects_unknown() -> None:
    assert isinstance(resolve_retrieval_backend(None), LexicalRetrievalBackend)
    assert isinstance(
        resolve_retrieval_backend("lexical_only"), LexicalRetrievalBackend
    )
    assert isinstance(
        resolve_retrieval_backend("code_aware"), CodeAwareRetrievalBackend
    )
    with pytest.raises(ValueError, match="Unknown retrieval_backend"):
        resolve_retrieval_backend("vector_magic")


def test_classify_candidate_skip_reasons_and_approved_overrides() -> None:
    assert classify_candidate("src/app.py", size_bytes=200) is None
    assert classify_candidate("node_modules/pkg/index.js") == "excluded_directory"
    assert classify_candidate(".env.local") == "hidden_or_ignored"
    assert classify_candidate("big.txt", size_bytes=2 * 1024 * 1024) == "file_too_large"
    assert classify_candidate("image.png") == "binary_or_unsupported"
    assert (
        classify_candidate("data.txt", sample=b"abc\x00def") == "binary_or_unsupported"
    )
    assert classify_candidate("bundle.min.js") == "generated_or_minified"
    assert classify_candidate("package-lock.json") == "generated_or_minified"
    assert classify_candidate("schema.generated.py") == "generated_or_minified"

    # Approved override allows task-relevant hidden or generated file
    assert (
        classify_candidate(
            ".github/workflows/ci.yml", approved_overrides={".github/**"}
        )
        is None
    )
    assert (
        classify_candidate("bundle.min.js", approved_overrides={"bundle.min.js"})
        is None
    )


def test_code_aware_backend_filters_candidates_and_normalizes_matches(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "def target_fn():\n    return 42\n", encoding="utf-8"
    )
    (tmp_path / "bundle.min.js").write_text(
        "function target_fn(){return 1}", encoding="utf-8"
    )
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    (tmp_path / ".secret.txt").write_text("hidden_token=123\n", encoding="utf-8")

    # Lexical-only baseline remains unchanged
    lex_list = execute_tool(
        ToolAction("list_files", {}), tmp_path, retrieval_backend="lexical_only"
    )
    assert lex_list.ok
    assert "app.py" in lex_list.data["paths"]
    assert "bundle.min.js" in lex_list.data["paths"]

    # Code-aware filters generated/binary files with explicit skip reasons
    aware_list = execute_tool(
        ToolAction("list_files", {}), tmp_path, retrieval_backend="code_aware"
    )
    assert aware_list.ok
    assert aware_list.data["paths"] == ["app.py"]
    assert {
        "path": "bundle.min.js",
        "reason": "generated_or_minified",
    } in aware_list.data["skipped"]
    assert {"path": "logo.png", "reason": "binary_or_unsupported"} in aware_list.data[
        "skipped"
    ]

    # search_code normalizes match metadata and omits generated hits
    aware_search = execute_tool(
        ToolAction("search_code", {"query": "target_fn"}),
        tmp_path,
        retrieval_backend="code_aware",
    )
    assert aware_search.ok
    matches = aware_search.data["matches"]
    assert len(matches) == 1
    assert matches[0]["path"] == "app.py"
    assert matches[0]["start_line"] == 1
    assert matches[0]["end_line"] == 2
    assert matches[0]["preview"] == "def target_fn():"
    assert matches[0]["matched_reason"] == "exact_symbol_definition"
    assert matches[0]["symbol"] == "target_fn"
    assert "1: def target_fn():" in str(matches[0]["snippet"])

    # Explicit authorized read of hidden file still works in code_aware mode
    hidden_read = execute_tool(
        ToolAction("read_file", {"path": ".secret.txt"}),
        tmp_path,
        retrieval_backend="code_aware",
    )
    assert hidden_read.ok
    assert "hidden_token=123" in str(hidden_read.data["content"])


def test_run_agent_with_code_aware_retrieval_keeps_tools_and_records_mode(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "def greet():\n    return 'hi'\n", encoding="utf-8"
    )
    (tmp_path / "dist.min.js").write_text("var greet=1;", encoding="utf-8")

    actions = iter(
        [
            {"kind": "tool", "name": "list_files", "arguments": {}},
            {"kind": "tool", "name": "search_code", "arguments": {"query": "greet"}},
            {"kind": "complete", "summary": "greet() is defined in main.py."},
        ]
    )

    run = run_agent(
        "Find greet",
        tmp_path,
        model=lambda _p, *, timeout: ChatResponse(json.dumps(next(actions))),
        retrieval_backend="code_aware",
    )

    assert run.status is RunStatus.COMPLETED
    started = next(e for e in run.events if e.event_type == RunEventType.RUN_STARTED)
    assert started.data["retrieval_mode"] == "code_aware"
