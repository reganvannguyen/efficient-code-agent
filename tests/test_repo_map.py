"""Tests for compact repository symbol map, caching, and invalidation (Issue #36)."""

from __future__ import annotations

from pathlib import Path

from harness.retrieval import CodeAwareRetrievalBackend, RepoSymbolIndex


def test_repo_map_caches_unchanged_files_and_updates_on_modify_rename_delete(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.py").write_text(
        "def greet() -> str:\n    return 'hi'\n", encoding="utf-8"
    )
    (tmp_path / "beta.py").write_text(
        "class Worker:\n    def run(self) -> None:\n        pass\n",
        encoding="utf-8",
    )

    index = RepoSymbolIndex(tmp_path)
    first = index.build_compact_map()
    assert first.file_count == 2
    assert first.symbol_count == 3
    assert "alpha.py:" in first.text
    assert "L1-2 function greet" in first.text
    assert "L1-3 class Worker" in first.text
    assert index.parse_count == 2

    # Unchanged second query reuses cache without re-parsing
    second = index.build_compact_map()
    assert second.text == first.text
    assert index.parse_count == 2

    # Modify alpha.py -> re-parses only alpha.py
    (tmp_path / "alpha.py").write_text(
        "def greet_v2() -> str:\n    return 'hello'\n", encoding="utf-8"
    )
    third = index.build_compact_map()
    assert "greet_v2" in third.text
    assert "function greet:" not in third.text
    assert index.parse_count == 3

    # Rename beta.py -> gamma.py and delete alpha.py -> no stale entries
    (tmp_path / "beta.py").rename(tmp_path / "gamma.py")
    (tmp_path / "alpha.py").unlink()
    fourth = index.build_compact_map()
    assert "alpha.py" not in fourth.text
    assert "beta.py" not in fourth.text
    assert "gamma.py:" in fourth.text
    assert "Worker" in fourth.text
    assert index.cached_paths() == ("gamma.py",)


def test_repo_map_bounds_large_output_skips_generated_and_enforces_confinement(
    tmp_path: Path,
) -> None:
    for i in range(6):
        (tmp_path / f"mod_{i}.py").write_text(
            f"def func_{i}_long_name(arg_one: int, arg_two: int) -> int:\n"
            f"    return arg_one + arg_two\n",
            encoding="utf-8",
        )
    (tmp_path / "schema.generated.py").write_text(
        "def generated_fn() -> None:\n    pass\n", encoding="utf-8"
    )
    outside = tmp_path.parent / "outside_secret.py"
    outside.write_text("def secret() -> None:\n    pass\n", encoding="utf-8")
    (tmp_path / "escape_link.py").symlink_to(outside)

    index = RepoSymbolIndex(tmp_path)
    bounded = index.build_compact_map(max_bytes=220)
    assert bounded.truncated is True
    assert len(bounded.text.encode("utf-8")) <= 220
    assert "generated_fn" not in bounded.text
    assert "secret" not in bounded.text
    assert {
        "path": "schema.generated.py",
        "reason": "generated_or_minified",
    } in bounded.skipped

    # Direct lookup of outside symlink is rejected
    parsed, reason = index.get_file_symbols("escape_link.py")
    assert parsed is None
    assert reason == "not_found_or_outside_workspace"


def test_code_aware_backend_invalidates_cached_file_after_edit(tmp_path: Path) -> None:
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    backend = CodeAwareRetrievalBackend()
    index = backend.get_index(tmp_path)
    initial = index.build_compact_map()
    assert "L1-2 function add" in initial.text
    assert index.parse_count == 1

    target.write_text(
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    backend.invalidate("calc.py", root=tmp_path)
    assert index.cached_paths() == ()

    refreshed = index.build_compact_map()
    assert "function sub" in refreshed.text
    assert index.parse_count == 2
