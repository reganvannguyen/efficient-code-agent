import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from harness.agent.protocol import ToolAction
from harness.tools.dispatcher import MAX_RESULT_BYTES, execute_tool


def edit(root, path, before, old, new, **overrides):
    args = {
        "path": path,
        "expected_sha256": hashlib.sha256(before).hexdigest(),
        "old_text": old,
        "new_text": new,
    }
    args.update(overrides)
    return execute_tool(ToolAction("edit_file", args), root, allow_edits=True)


def test_read_hash_authorized_edit_and_diff_are_end_to_end(tmp_path):
    target = tmp_path / "a.py"
    original = b"def answer():\n    return 1\n"
    target.write_bytes(original)
    os.chmod(target, 0o640)
    read = execute_tool(ToolAction("read_file", {"path": "a.py"}), tmp_path)

    result = edit(
        tmp_path,
        "a.py",
        original,
        "return 1",
        "return 2",
        expected_sha256=read.data["sha256"].upper(),
    )

    assert read.data["sha256"] == hashlib.sha256(original).hexdigest()
    assert result.ok and result.data["write_state"] == "applied"
    assert result.data["changed_paths"] == ["a.py"]
    assert "-    return 1\n" in result.data["diff"]
    assert "+    return 2\n" in result.data["diff"]
    assert target.read_bytes() == b"def answer():\n    return 2\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_edit_is_denied_by_default_and_bad_requests_do_not_resolve_workspace(
    tmp_path, monkeypatch
):
    action = ToolAction("edit_file", {"path": "missing", "extra": True})
    result = execute_tool(action, tmp_path)
    assert result.error_code == "invalid_arguments"

    valid = ToolAction(
        "edit_file",
        {
            "path": "missing",
            "expected_sha256": "0" * 64,
            "old_text": "a",
            "new_text": "b",
        },
    )
    denied = execute_tool(valid, tmp_path)
    assert denied.error_code == "permission_denied"

    monkeypatch.setattr(
        "harness.tools.dispatcher._workspace",
        lambda _root: pytest.fail("invalid authorization flag reached filesystem"),
    )
    assert (
        execute_tool(valid, tmp_path, allow_edits=1).error_code == "invalid_arguments"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "x", "expected_sha256": "x", "old_text": "a", "new_text": "b"},
        {"path": "x", "expected_sha256": "0" * 64, "old_text": "\x00", "new_text": "b"},
        {
            "path": "x",
            "expected_sha256": "0" * 64,
            "old_text": "a",
            "new_text": "\ud800",
        },
        {
            "path": "x",
            "expected_sha256": "0" * 64,
            "old_text": "a",
            "new_text": "b",
            "extra": 1,
        },
        {"path": "../x", "expected_sha256": "0" * 64, "old_text": "a", "new_text": "b"},
    ],
)
def test_edit_schema_fails_before_workspace_resolution(
    tmp_path, monkeypatch, arguments
):
    monkeypatch.setattr(
        "harness.tools.dispatcher._workspace",
        lambda _root: pytest.fail("schema validation reached filesystem"),
    )
    result = execute_tool(
        ToolAction("edit_file", arguments), tmp_path, allow_edits=True
    )
    assert not result.ok


@pytest.mark.parametrize(
    ("content", "old", "new"),
    [
        (b"stale", "stale", "fresh"),
        (b"one one", "one", "two"),
        (b"aaa", "aa", "x"),  # overlapping second occurrence
        (b"not empty", "", "added"),
    ],
)
def test_conflicts_leave_target_untouched(tmp_path, content, old, new):
    target = tmp_path / "x"
    target.write_bytes(content)
    before = target.stat().st_mtime_ns
    if content == b"stale":
        target.write_bytes(b"changed after read")
    result = edit(tmp_path, "x", content, old, new)
    assert not result.ok and result.error_code == "conflict"
    assert (
        target.read_bytes() != content
        if content == b"stale"
        else target.read_bytes() == content
    )
    assert target.stat().st_mtime_ns >= before


def test_empty_file_insert_delete_noop_and_repeated_request(tmp_path):
    target = tmp_path / "x"
    target.write_bytes(b"")
    insert = edit(tmp_path, "x", b"", "", "hello")
    assert insert.ok and target.read_bytes() == b"hello"
    assert edit(tmp_path, "x", b"hello", "hello", "").ok
    assert target.read_bytes() == b""
    before = target.stat().st_mtime_ns
    noop = edit(tmp_path, "x", b"", "", "")
    assert noop.ok and noop.data["write_state"] == "unchanged"
    assert target.stat().st_mtime_ns == before
    same = edit(tmp_path, "x", b"", "", "")
    assert same.ok and same.data["write_state"] == "unchanged"
    repeated = edit(tmp_path, "x", b"", "", "hello")
    assert repeated.ok
    stale = edit(tmp_path, "x", b"", "", "hello")
    assert not stale.ok and stale.error_code == "conflict"


def test_stale_request_conflicts_even_when_new_text_contains_old_text(tmp_path):
    target = tmp_path / "x"
    target.write_text("a")
    first = edit(tmp_path, "x", b"a", "a", "aa")
    second = edit(tmp_path, "x", b"a", "a", "aa")
    assert first.ok and target.read_text() == "aa"
    assert not second.ok and second.error_code == "conflict"


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".codex/settings",
        ".ssh/key",
        ".venv/file",
        "__pycache__/x",
        "node_modules/x",
        "dist/x",
        "build/x",
        ".pytest_cache/x",
        ".ruff_cache/x",
        ".GIT/config",
        ".ENV.LOCAL",
        ".env",
        ".env.local",
    ],
)
def test_edit_protected_paths_are_rejected(tmp_path, path):
    result = edit(tmp_path, path, b"", "", "x")
    assert result.error_code == "path_denied"


def test_edit_rejects_symlink_parent_target_hardlink_and_nonwritable(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)
    assert (
        edit(tmp_path, "link", b"secret", "secret", "safe").error_code == "path_denied"
    )
    (tmp_path / "dir-link").symlink_to(tmp_path, target_is_directory=True)
    assert edit(tmp_path, "dir-link/outside", b"", "", "x").error_code == "path_denied"

    target = tmp_path / "hard"
    target.write_text("a")
    os.link(target, tmp_path / "alias")
    assert edit(tmp_path, "hard", b"a", "a", "b").error_code == "path_denied"
    (tmp_path / "readonly").write_text("a")
    os.chmod(tmp_path / "readonly", 0o444)
    assert edit(tmp_path, "readonly", b"a", "a", "b").error_code == "unwritable"


def test_edit_preserves_newlines_and_reports_no_final_newline(tmp_path):
    target = tmp_path / "x.txt"
    target.write_bytes("one\r\ntwo".encode())
    result = edit(tmp_path, "x.txt", b"one\r\ntwo", "two", "three")
    assert result.ok
    assert target.read_bytes() == b"one\r\nthree"
    assert "\\ No newline at end of file" in result.data["diff"]


@pytest.mark.parametrize("payload", [b"a\x00b", b"\xff", b"x" * (1024 * 1024 + 1)])
def test_edit_rejects_binary_invalid_utf8_and_oversized_targets(tmp_path, payload):
    target = tmp_path / "x"
    target.write_bytes(payload)
    result = edit(tmp_path, "x", payload, "x", "y")
    assert not result.ok
    assert result.error_code in {"binary_file", "encoding_error", "file_too_large"}
    assert target.read_bytes() == payload


def test_edit_accepts_exact_file_and_fragment_size_limit(tmp_path):
    from harness.tools.repository import MAX_FILE_BYTES

    target = tmp_path / "x"
    original = b"a" * MAX_FILE_BYTES
    target.write_bytes(original)
    result = edit(tmp_path, "x", original, original.decode(), "b")
    assert result.ok and result.data["write_state"] == "applied"
    assert target.read_bytes() == b"b"


@pytest.mark.parametrize("field", ["old_text", "new_text"])
def test_edit_rejects_fragment_larger_than_file_limit_before_workspace(
    tmp_path, monkeypatch, field
):
    arguments = {
        "path": "x",
        "expected_sha256": "0" * 64,
        "old_text": "a",
        "new_text": "b",
    }
    arguments[field] = "x" * (1024 * 1024 + 1)
    monkeypatch.setattr(
        "harness.tools.dispatcher._workspace",
        lambda _root: pytest.fail("oversized edit fragment reached filesystem"),
    )
    result = execute_tool(
        ToolAction("edit_file", arguments), tmp_path, allow_edits=True
    )
    assert result.error_code == "invalid_arguments"


def test_edit_fsync_failure_cleans_created_temporary_file(tmp_path, monkeypatch):
    from harness.tools import editing

    target = tmp_path / "x"
    target.write_text("old")
    before = {entry.name for entry in tmp_path.iterdir()}
    monkeypatch.setattr(
        editing.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError())
    )
    result = edit(tmp_path, "x", b"old", "old", "new")
    assert not result.ok and result.error_code == "io_error"
    assert result.data == {} and target.read_text() == "old"
    assert {entry.name for entry in tmp_path.iterdir()} == before


def test_directory_and_fifo_targets_are_rejected_without_opening(tmp_path):
    (tmp_path / "directory").mkdir()
    assert edit(tmp_path, "directory", b"", "", "x").error_code == "wrong_type"
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable")
    os.mkfifo(tmp_path / "fifo")
    assert edit(tmp_path, "fifo", b"", "", "x").error_code == "wrong_type"


def test_replace_reconciliation_marks_unexpected_target_unknown(tmp_path, monkeypatch):
    from harness.tools import editing

    target = tmp_path / "x"
    target.write_bytes(b"old")

    def replace_with_other(_source, destination):
        destination.write_bytes(b"external")
        raise OSError("unexpected state")

    monkeypatch.setattr(editing.os, "replace", replace_with_other)
    result = edit(tmp_path, "x", b"old", "old", "new")
    assert not result.ok and result.data["write_state"] == "unknown"
    assert result.data["changed_paths"] == ["x"]
    assert "external" in result.data["diff"]


@pytest.mark.parametrize("after_error", [OSError, KeyboardInterrupt])
def test_edit_failure_before_and_after_replace_is_reconciled(
    tmp_path, monkeypatch, after_error
):
    from harness.tools import editing

    target = tmp_path / "x"
    original = b"old"
    target.write_bytes(original)
    real_replace = editing.os.replace

    before_entries = {entry.name for entry in tmp_path.iterdir()}

    def interrupt_before(_source, _destination):
        raise KeyboardInterrupt

    monkeypatch.setattr(editing.os, "replace", interrupt_before)
    before = edit(tmp_path, "x", original, "old", "new")
    assert not before.ok and before.error_code == "cancelled"
    assert before.data["write_state"] == "unchanged"
    assert before.data["changed_paths"] == [] and target.read_bytes() == original
    assert {entry.name for entry in tmp_path.iterdir()} == before_entries

    def fail_after_with_ioerror(source, destination):
        real_replace(source, destination)
        raise after_error("after replace")

    monkeypatch.setattr(editing.os, "replace", fail_after_with_ioerror)
    after = edit(tmp_path, "x", original, "old", "new")
    assert not after.ok
    assert after.error_code == (
        "cancelled" if after_error is KeyboardInterrupt else "io_error"
    )
    assert after.data["write_state"] == "applied"
    assert after.data["changed_paths"] == ["x"]
    assert target.read_bytes() == b"new"
    assert {entry.name for entry in tmp_path.iterdir()} == before_entries


def test_unreadable_postreplace_target_is_reported_unknown(tmp_path, monkeypatch):
    from harness.tools import editing

    target = tmp_path / "x"
    target.write_bytes(b"old")
    real_replace = editing.os.replace

    def replace_then_remove(source, destination):
        real_replace(source, destination)
        Path(destination).unlink()
        raise OSError("target unavailable for reconciliation")

    monkeypatch.setattr(editing.os, "replace", replace_then_remove)
    result = edit(tmp_path, "x", b"old", "old", "new")
    assert not result.ok and result.data["write_state"] == "unknown"
    assert result.data["changed_paths"] == []
    assert result.data["path"] == "x"


def test_staging_content_conflict_cleans_temporary_file(tmp_path, monkeypatch):
    from harness.tools import editing

    target = tmp_path / "x"
    target.write_bytes(b"old")
    before_entries = {entry.name for entry in tmp_path.iterdir()}
    create_temp = editing._create_temp

    def race_after_stage(path, contents, mode):
        temp = create_temp(path, contents, mode)
        target.write_bytes(b"external")
        return temp

    monkeypatch.setattr(editing, "_create_temp", race_after_stage)
    result = edit(tmp_path, "x", b"old", "old", "new")
    assert not result.ok and result.error_code == "conflict"
    assert target.read_bytes() == b"external"
    assert {entry.name for entry in tmp_path.iterdir()} == before_entries


def test_parent_symlink_swap_during_staging_does_not_edit_outside_file(
    tmp_path, monkeypatch
):
    from harness.tools import editing

    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "x"
    target.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "x"
    outside_target.write_bytes(b"outside")
    create_temp = editing._create_temp

    def swap_parent_after_stage(path, contents, mode):
        temp = create_temp(path, contents, mode)
        temp.unlink()
        target.unlink()
        parent.rmdir()
        parent.symlink_to(outside, target_is_directory=True)
        return temp

    monkeypatch.setattr(editing, "_create_temp", swap_parent_after_stage)
    result = edit(tmp_path, "parent/x", b"inside", "inside", "changed")
    assert not result.ok
    assert outside_target.read_bytes() == b"outside"


def test_edit_succeeds_with_long_valid_basename_and_cleans_staging_file(tmp_path):
    name = "f" * 240
    target = tmp_path / name
    target.write_bytes(b"old")
    before_entries = {entry.name for entry in tmp_path.iterdir()}
    result = edit(tmp_path, name, b"old", "old", "new")
    assert result.ok and target.read_bytes() == b"new"
    assert {entry.name for entry in tmp_path.iterdir()} == before_entries


def test_edit_diff_and_serialized_result_stay_bounded(tmp_path):
    target = tmp_path / "x"
    original = "é" * 20000
    target.write_text(original, encoding="utf-8")
    result = edit(tmp_path, "x", original.encode(), original, "x")
    payload = json.dumps(result.to_dict(), ensure_ascii=True).encode()
    assert result.ok and result.truncated
    assert len(result.data["diff"].encode()) <= 16 * 1024
    assert len(payload) <= MAX_RESULT_BYTES
    assert result.data["write_state"] == "applied"
    assert result.data["changed_paths"] == ["x"]


def test_edit_line_range_replaces_block_and_preserves_trailing_newline(tmp_path):
    target = tmp_path / "sample.py"
    original = "line1\nline2\nline3\nline4\n"
    target.write_text(original, encoding="utf-8")
    result = execute_tool(
        ToolAction(
            "edit_file",
            {
                "path": "sample.py",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "start_line": 2,
                "end_line": 3,
                "new_text": "updated2\nupdated3",
            },
        ),
        tmp_path,
        allow_edits=True,
    )
    assert result.ok
    assert result.data["write_state"] == "applied"
    assert target.read_text(encoding="utf-8") == "line1\nupdated2\nupdated3\nline4\n"


def test_edit_line_range_deletes_lines_and_handles_empty_file(tmp_path):
    target = tmp_path / "sample.py"
    original = "a\nb\nc\n"
    target.write_text(original, encoding="utf-8")
    deleted = execute_tool(
        ToolAction(
            "edit_file",
            {
                "path": "sample.py",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "start_line": 2,
                "end_line": 2,
                "new_text": "",
            },
        ),
        tmp_path,
        allow_edits=True,
    )
    assert deleted.ok
    assert target.read_text(encoding="utf-8") == "a\nc\n"

    empty = tmp_path / "empty.py"
    empty.write_text("", encoding="utf-8")
    filled = execute_tool(
        ToolAction(
            "edit_file",
            {
                "path": "empty.py",
                "expected_sha256": hashlib.sha256(b"").hexdigest(),
                "start_line": 1,
                "end_line": 1,
                "new_text": "x = 1\n",
            },
        ),
        tmp_path,
        allow_edits=True,
    )
    assert filled.ok
    assert empty.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_line_range_rejects_out_of_bounds_and_mixed_arguments(tmp_path):
    target = tmp_path / "sample.py"
    original = "a\nb\n"
    target.write_text(original, encoding="utf-8")
    oob = execute_tool(
        ToolAction(
            "edit_file",
            {
                "path": "sample.py",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "start_line": 1,
                "end_line": 5,
                "new_text": "x",
            },
        ),
        tmp_path,
        allow_edits=True,
    )
    assert not oob.ok and oob.error_code == "conflict"

    mixed = execute_tool(
        ToolAction(
            "edit_file",
            {
                "path": "sample.py",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "old_text": "a",
                "start_line": 1,
                "end_line": 1,
                "new_text": "x",
            },
        ),
        tmp_path,
        allow_edits=True,
    )
    assert not mixed.ok and mixed.error_code == "invalid_arguments"
