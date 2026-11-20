import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.agent.protocol import ToolAction, parse_action
from harness.tools.dispatcher import MAX_RESULT_BYTES, ToolResult, execute_tool


def call(workspace: Path, name: str, arguments: dict[str, object]):
    return execute_tool(ToolAction(name, arguments), workspace)


def test_read_file_default_range_and_metadata(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    result = call(tmp_path, "read_file", {"path": "sample.txt"})

    assert result.ok
    assert result.data == {
        "path": "sample.txt",
        "size": 11,
        "sha256": hashlib.sha256(b"alpha\nbeta\n").hexdigest(),
        "total_lines": 2,
        "start_line": 1,
        "end_line": 2,
        "content": "1: alpha\n2: beta",
    }
    assert not result.truncated


def test_read_file_line_range_and_empty_beyond_end(tmp_path):
    (tmp_path / "sample.txt").write_text("a\nb\nc", encoding="utf-8")

    selected = call(
        tmp_path, "read_file", {"path": "sample.txt", "start_line": 2, "end_line": 2}
    )
    beyond = call(tmp_path, "read_file", {"path": "sample.txt", "start_line": 8})

    assert selected.data["content"] == "2: b"
    assert selected.truncated
    assert selected.data["sha256"] == hashlib.sha256(b"a\nb\nc").hexdigest()
    assert beyond.ok and beyond.truncated
    assert beyond.data["start_line"] is None
    assert beyond.data["end_line"] is None
    assert beyond.data["content"] == ""


def test_read_multiple_files_nested_directory_and_default_start_range(tmp_path):
    nested = tmp_path / "pkg" / "inner"
    nested.mkdir(parents=True)
    (nested / "first.py").write_text("a\nb\nc", encoding="utf-8")
    (tmp_path / "second.py").write_text("x", encoding="utf-8")

    listing = call(tmp_path, "list_files", {"directory": "pkg"})
    read_first = call(
        tmp_path, "read_file", {"path": "pkg/inner/first.py", "start_line": 2}
    )
    read_second = call(tmp_path, "read_file", {"path": "second.py"})

    assert listing.data["paths"] == ["pkg/inner/first.py"]
    assert read_first.data["content"] == "2: b\n3: c"
    assert read_second.data["content"] == "1: x"


def test_read_range_is_limited_to_200_lines(tmp_path):
    (tmp_path / "many.txt").write_text("\n".join(str(i) for i in range(250)))

    result = call(tmp_path, "read_file", {"path": "many.txt", "start_line": 1})

    assert result.data["total_lines"] == 250
    assert result.data["end_line"] == 200
    assert result.truncated


def test_read_empty_file_is_successful_and_not_truncated(tmp_path):
    (tmp_path / "empty.txt").write_bytes(b"")

    result = call(tmp_path, "read_file", {"path": "empty.txt"})

    assert result.ok and not result.truncated
    assert result.data["total_lines"] == 0
    assert result.data["start_line"] is None
    assert result.data["end_line"] is None


def test_list_files_honors_ignore_files_in_non_git_workspace_and_exclusions(
    tmp_path, monkeypatch
):
    (tmp_path / ".ignore").write_text("ignored.txt\n", encoding="utf-8")
    for name in ("visible.txt", "ignored.txt", ".hidden.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    for directory in ("node_modules", ".venv", "__pycache__", "dist", "build"):
        target = tmp_path / directory
        target.mkdir()
        (target / "generated.py").write_text("x", encoding="utf-8")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", "/missing/config")

    result = call(tmp_path, "list_files", {})

    assert result.ok
    assert result.data["paths"] == ["visible.txt"]


def test_list_pattern_cannot_reveal_ignored_or_generated_files(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("secret", encoding="utf-8")
    (tmp_path / "visible.py").write_text("visible", encoding="utf-8")
    generated = tmp_path / "node_modules"
    generated.mkdir()
    (generated / "secret.py").write_text("secret", encoding="utf-8")

    result = call(tmp_path, "list_files", {"pattern": "*.py"})

    assert result.data["paths"] == ["visible.py"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("missing", {}),
        ("read_file", {}),
        ("read_file", {"path": "a", "unknown": 1}),
        ("read_file", {"path": "a", "start_line": True}),
        ("read_file", {"path": "a", "start_line": 1, "end_line": 201}),
        ("list_files", {"pattern": ""}),
        ("list_files", {"pattern": None}),
        ("list_files", {"pattern": "!*.py"}),
        ("list_files", {"directory": 1}),
        ("list_files", {"extra": 2}),
    ],
)
def test_bad_schemas_are_controlled(name, arguments, tmp_path):
    result = call(tmp_path, name, arguments)
    assert not result.ok
    assert result.error_code in {"invalid_arguments", "unknown_tool"}


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "bad\x00path"])
def test_paths_reject_traversal_and_absolute_names(path, tmp_path):
    result = call(tmp_path, "read_file", {"path": path})
    assert not result.ok
    assert result.error_code in {"invalid_arguments", "path_denied"}


def test_symlink_escape_is_rejected_and_in_root_symlink_is_allowed(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "inside-link.txt").symlink_to(tmp_path / "inside.txt")

    escaped = call(tmp_path, "read_file", {"path": "escape.txt"})
    allowed = call(tmp_path, "read_file", {"path": "inside-link.txt"})

    assert not escaped.ok and escaped.error_code == "path_denied"
    assert allowed.ok and allowed.data["path"] == "inside.txt"


def test_symlink_loop_is_denied_and_listing_does_not_follow_directory_links(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    real = tmp_path / "real"
    real.mkdir()
    (real / "child.txt").write_text("child", encoding="utf-8")
    (tmp_path / "dir-link").symlink_to(real, target_is_directory=True)

    loop_result = call(tmp_path, "read_file", {"path": "loop"})
    listing = call(tmp_path, "list_files", {})

    assert not loop_result.ok and loop_result.error_code == "path_denied"
    assert listing.data["paths"] == ["real/child.txt"]


def test_explicit_read_allows_hidden_ignored_and_generated_files(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    generated = tmp_path / "node_modules"
    generated.mkdir()
    (generated / "generated.txt").write_text("generated", encoding="utf-8")

    hidden = call(tmp_path, "read_file", {"path": ".hidden.txt"})
    ignored = call(tmp_path, "read_file", {"path": "ignored.txt"})
    generated_read = call(tmp_path, "read_file", {"path": "node_modules/generated.txt"})

    assert all(result.ok for result in (hidden, ignored, generated_read))
    assert hidden.data["content"] == "1: hidden"
    assert ignored.data["content"] == "1: ignored"
    assert generated_read.data["content"] == "1: generated"


def test_workspace_must_exist_and_be_directory(tmp_path):
    assert call(tmp_path / "missing", "list_files", {}).error_code == "workspace_error"
    file = tmp_path / "file"
    file.write_text("x")
    assert call(file, "list_files", {}).error_code == "workspace_error"
    assert call(tmp_path, "read_file", {"path": "missing"}).error_code == "not_found"


def test_read_file_rejects_directory_binary_invalid_utf8_and_large_files(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "binary").write_bytes(b"a\x00b")
    (tmp_path / "invalid").write_bytes(b"\xff")
    (tmp_path / "large").write_bytes(b"x" * (1024 * 1024 + 1))

    assert call(tmp_path, "read_file", {"path": "folder"}).error_code == "wrong_type"
    assert call(tmp_path, "read_file", {"path": "binary"}).error_code == "binary_file"
    assert (
        call(tmp_path, "read_file", {"path": "invalid"}).error_code == "encoding_error"
    )
    assert call(tmp_path, "read_file", {"path": "large"}).error_code == "file_too_large"


def test_read_file_permission_error_is_controlled(tmp_path, monkeypatch):
    target = tmp_path / "private.txt"
    target.write_text("secret", encoding="utf-8")
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    result = call(tmp_path, "read_file", {"path": "private.txt"})
    assert result.error_code == "unreadable"


def test_fifo_is_rejected_without_opening_it(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable")
    os.mkfifo(tmp_path / "fifo")
    assert call(tmp_path, "read_file", {"path": "fifo"}).error_code == "wrong_type"


def test_read_content_and_serialized_result_are_bounded(tmp_path):
    (tmp_path / "long").write_text("é" * 20000, encoding="utf-8")

    result = call(tmp_path, "read_file", {"path": "long"})
    serialized = json.dumps(result.to_dict()).encode("utf-8")

    assert result.truncated
    assert len(result.data["content"].encode("utf-8")) <= 16 * 1024
    assert len(serialized) <= MAX_RESULT_BYTES
    assert result.data["end_line"] == 1


def test_schema_is_rejected_before_workspace_resolution(tmp_path, monkeypatch):
    from harness.tools import dispatcher

    def fail(_workspace):
        raise AssertionError("workspace must not resolve for invalid arguments")

    monkeypatch.setattr(dispatcher, "_workspace", fail)
    result = call(tmp_path, "read_file", {"path": "a", "start_line": False})
    assert result.error_code == "invalid_arguments"


def test_missing_ripgrep_is_a_controlled_error(tmp_path, monkeypatch):
    from harness.tools import repository

    monkeypatch.setattr(repository.shutil, "which", lambda _name: None)
    assert call(tmp_path, "list_files", {}).error_code == "missing_rg"


def test_ripgrep_timeout_is_a_controlled_error(tmp_path, monkeypatch):
    from harness.tools import repository

    monkeypatch.setattr(repository.shutil, "which", lambda _name: "rg")
    monkeypatch.setattr(
        repository,
        "_run_bounded",
        lambda *_args: (_ for _ in ()).throw(subprocess.TimeoutExpired("rg", 5)),
    )
    assert call(tmp_path, "list_files", {}).error_code == "timeout"


def test_ripgrep_execution_error_is_controlled(tmp_path, monkeypatch):
    from harness.tools import repository

    monkeypatch.setattr(repository.shutil, "which", lambda _name: "rg")
    monkeypatch.setattr(repository, "_run_bounded", lambda *_args: (b"", False, 2))
    assert call(tmp_path, "list_files", {}).error_code == "io_error"


def test_ripgrep_command_is_fixed_and_environment_ignores_user_config(
    tmp_path, monkeypatch
):
    from harness.tools import repository

    captured = {}

    def fake_run(command, root, env, pattern):
        captured.update(command=command, root=root, env=env, pattern=pattern)
        return b"", False, 1

    monkeypatch.setattr(repository.shutil, "which", lambda _name: "rg")
    monkeypatch.setattr(repository, "_run_bounded", fake_run)
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", "/tmp/config")

    result = call(tmp_path, "list_files", {"pattern": "*.py"})

    assert result.ok and result.data["paths"] == []
    assert "--no-config" in captured["command"]
    assert "--no-require-git" in captured["command"]
    assert "--no-ignore-global" in captured["command"]
    assert captured["pattern"] == "*.py"
    assert "RIPGREP_CONFIG_PATH" not in captured["env"]
    assert "--glob" not in captured["command"]


def test_protocol_action_dispatch_and_read_only_workspace(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    before = source.read_bytes()
    action = parse_action(
        '{"kind":"tool","name":"read_file","arguments":{"path":"a.py"}}'
    )

    assert isinstance(action, ToolAction)
    result = execute_tool(action, tmp_path)
    assert result.ok and "answer = 42" in result.data["content"]
    assert source.read_bytes() == before


def test_result_payload_cap_also_covers_metadata():
    result = ToolResult("x" * 40000, True, {"content": "small"})
    payload = json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")
    assert result.truncated
    assert len(payload) <= MAX_RESULT_BYTES


def test_list_path_count_cap_and_byte_cap_terminate_subprocess(tmp_path):
    from harness.tools.repository import _run_bounded

    for index in range(205):
        (tmp_path / f"f{index:03}.txt").write_text("x", encoding="utf-8")
    rg = shutil_which_rg()
    output, capped, _returncode = _run_bounded(
        [rg, "--files", "--null", str(tmp_path)], tmp_path, os.environ.copy(), None
    )
    assert capped
    assert output.endswith(b"\0")
    assert len(output.split(b"\0")) - 1 == 201

    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x' * 100000 + b'\\0')",
    ]
    output, capped, _returncode = _run_bounded(
        command, tmp_path, os.environ.copy(), None
    )
    assert capped and len(output) <= 64 * 1024


def test_streaming_list_timeout_kills_and_reaps_child(tmp_path, monkeypatch):
    from harness.tools import repository

    monkeypatch.setattr(repository, "LIST_TIMEOUT_SECONDS", 0.02)
    command = [sys.executable, "-c", "import time; time.sleep(1)"]
    with pytest.raises(subprocess.TimeoutExpired):
        repository._run_bounded(command, tmp_path, os.environ.copy(), None)


def test_streaming_list_reaps_child_after_reader_error(tmp_path, monkeypatch):
    from harness.tools import repository

    processes = []
    popen = subprocess.Popen

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(repository.subprocess, "Popen", track)
    monkeypatch.setattr(
        repository,
        "_accept_path",
        lambda *_args: (_ for _ in ()).throw(OSError("reader failed")),
    )
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x\\0'); sys.stdout.flush(); import time; time.sleep(1)",
    ]

    with pytest.raises(OSError, match="reader failed"):
        repository._run_bounded(command, tmp_path, os.environ.copy(), None)
    assert processes[0].poll() is not None


def test_empty_glob_has_recovery_hint_and_literal_pipe_still_matches(tmp_path):
    (tmp_path / "one|two.py").write_text("example", encoding="utf-8")

    missing = call(tmp_path, "list_files", {"pattern": "one.py|two.py"})
    assert missing.ok and missing.data["paths"] == []
    assert "literal, not OR" in missing.message
    assert "omit pattern" in missing.message

    literal = call(tmp_path, "list_files", {"pattern": "one|two.py"})
    assert literal.ok and literal.data["paths"] == ["one|two.py"]
    assert literal.message is None


def shutil_which_rg():
    from shutil import which

    return which("rg") or pytest.skip("ripgrep is not installed")


def test_list_files_missing_directory_includes_root_hint(tmp_path):
    res = call(tmp_path, "list_files", {"directory": "path/to/pkg"})
    assert not res.ok
    assert res.error_code == "not_found"
    assert "omit directory" in res.message
    assert "'.'" in res.message
