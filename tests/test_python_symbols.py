"""Tests for Tree-sitter Python symbol and source-range extraction (Issue #35)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from harness.retrieval import (
    extract_python_symbols_from_bytes,
    parse_python_file,
)


def test_extracts_imports_classes_decorated_methods_nested_and_multiline_signatures(
    tmp_path: Path,
) -> None:
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "@dataclass\n"
        "class Service(Base):\n"
        "    @classmethod\n"
        "    async def fetch(\n"
        "        cls,\n"
        "        item_id: str,\n"
        "    ) -> dict[str, object]:\n"
        "        def helper(x: int) -> int:\n"
        "            return x + 1\n"
        "        return {'id': helper(1)}\n"
        "\n"
        "def top_level(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )
    target = tmp_path / "service.py"
    target.write_text(source, encoding="utf-8")

    result = parse_python_file(tmp_path, "service.py")
    assert result.language == "python"
    assert result.has_errors is False
    assert result.uncertain is False
    assert result.fallback_reason is None
    assert result.file_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()

    by_qname = {s.qualified_name: s for s in result.symbols}
    assert by_qname["os"].kind == "import"
    assert by_qname["os"].start_line == 1

    assert by_qname["pathlib"].kind == "import"
    assert by_qname["pathlib"].start_line == 2

    svc = by_qname["Service"]
    assert svc.kind == "class"
    assert svc.start_line == 4  # includes @dataclass decorator
    assert svc.end_line == 13
    assert "@dataclass\nclass Service(Base):" in svc.header

    fetch = by_qname["Service.fetch"]
    assert fetch.kind == "async_method"
    assert fetch.parent == "Service"
    assert fetch.start_line == 6  # includes @classmethod
    assert fetch.end_line == 13
    assert (
        "async def fetch(\n        cls,\n        item_id: str,\n    ) -> dict[str, object]:"
        in fetch.header
    )

    helper = by_qname["Service.fetch.helper"]
    assert helper.kind == "function"
    assert helper.parent == "Service.fetch"
    assert helper.start_line == 11
    assert helper.end_line == 12

    top = by_qname["top_level"]
    assert top.kind == "function"
    assert top.parent is None
    assert top.start_line == 15
    assert top.end_line == 16


def test_syntax_errors_and_incomplete_python_mark_uncertain() -> None:
    broken = b"def valid_fn():\n    return 1\n\ndef broken_fn(:\n    pass\n"
    result = extract_python_symbols_from_bytes("broken.py", broken)

    assert result.language == "python"
    assert result.has_errors is True
    assert result.uncertain is True
    assert result.fallback_reason == "syntax_errors"
    assert any(s.name == "valid_fn" and not s.uncertain for s in result.symbols)


def test_unsupported_language_oversized_binary_and_missing_fallbacks(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.js").write_text("function f() {}\n", encoding="utf-8")
    js_res = parse_python_file(tmp_path, "app.js")
    assert js_res.language is None
    assert js_res.uncertain is True
    assert js_res.fallback_reason == "unsupported_language"
    assert js_res.symbols == ()

    (tmp_path / "huge.py").write_text("x = 1\n" * 500, encoding="utf-8")
    big_res = parse_python_file(tmp_path, "huge.py", max_file_bytes=64)
    assert big_res.uncertain is True
    assert big_res.fallback_reason == "file_too_large"
    assert big_res.symbols == ()

    bad_utf8 = extract_python_symbols_from_bytes("bad.py", b"\xff\xfe\x00")
    assert bad_utf8.uncertain is True
    assert bad_utf8.fallback_reason == "binary_or_encoding_error"

    missing = parse_python_file(tmp_path, "does_not_exist.py")
    assert missing.uncertain is True
    assert missing.fallback_reason == "not_found"
