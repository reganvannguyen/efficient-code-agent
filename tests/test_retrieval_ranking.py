"""Tests for deterministic candidate ranking and bounded snippets (Issue #37)."""

from __future__ import annotations

from pathlib import Path

from harness.agent.protocol import ToolAction
from harness.retrieval import CodeAwareRetrievalBackend
from harness.tools.dispatcher import execute_tool


def test_definition_outranks_comments_and_returns_complete_function(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.py").write_text(
        "# compute_tax is used in billing\n"
        "# TODO: check compute_tax edge cases\n"
        "x = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "billing.py").write_text(
        "def helper():\n    return 0\n\n"
        "def compute_tax(amount: float, rate: float) -> float:\n"
        "    return round(amount * rate, 2)\n",
        encoding="utf-8",
    )
    (tmp_path / "test_billing.py").write_text(
        "from billing import compute_tax\n"
        "def test_compute_tax():\n"
        "    assert compute_tax(100, 0.1) == 10.0\n",
        encoding="utf-8",
    )

    backend = CodeAwareRetrievalBackend()
    first = execute_tool(
        ToolAction("search_code", {"query": "compute_tax"}),
        tmp_path,
        retrieval_backend=backend,
    )
    second = execute_tool(
        ToolAction("search_code", {"query": "compute_tax"}),
        tmp_path,
        retrieval_backend=backend,
    )
    assert first.ok
    assert first.data == second.data  # deterministic & reproducible

    matches = first.data["matches"]
    assert len(matches) >= 2
    # Definition in billing.py ranks #1 ahead of test_billing.py and comments in notes.py
    top = matches[0]
    assert top["path"] == "billing.py"
    assert top["symbol"] == "compute_tax"
    assert top["matched_reason"] == "exact_symbol_definition"
    assert top["start_line"] == 4
    assert top["end_line"] == 5
    assert "4: def compute_tax" in str(top["snippet"])
    assert "5:     return round(amount * rate, 2)" in str(top["snippet"])
    assert matches[-1]["path"] == "notes.py"
    assert matches[-1]["matched_reason"] == "comment_mention"


def test_per_file_cap_and_budget_truncation_preserve_diversity(
    tmp_path: Path,
) -> None:
    many_defs = "\n".join(
        f"def target_{i}(x: int) -> int:\n    val = x + {i}\n    return val * 2\n"
        for i in range(6)
    )
    (tmp_path / "many.py").write_text(many_defs, encoding="utf-8")
    (tmp_path / "other.py").write_text(
        "def target_other(x: int) -> int:\n"
        "    # a longer line to force snippet byte truncation in this test\n"
        "    return x * 2\n",
        encoding="utf-8",
    )

    backend = CodeAwareRetrievalBackend(
        max_per_file=2,
        max_snippet_bytes=90,
        max_total_bytes=220,
        max_total_tokens=100,
        token_counter=lambda s: len(s.split()),
    )
    res = execute_tool(
        ToolAction("search_code", {"query": "target"}),
        tmp_path,
        retrieval_backend=backend,
    )
    assert res.ok
    matches = res.data["matches"]
    many_hits = [m for m in matches if m["path"] == "many.py"]
    assert len(many_hits) <= 2
    assert res.truncated is True
    assert any(m["truncated"] for m in matches)
