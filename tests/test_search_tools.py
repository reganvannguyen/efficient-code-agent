import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.agent.protocol import ToolAction
from harness.tools import search as search_module
from harness.tools.dispatcher import MAX_RESULT_BYTES, ToolResult, execute_tool


def search(root: Path, **arguments):
    return execute_tool(ToolAction("search_code", arguments), root)


def test_search_returns_sorted_line_matches_and_regex_alternation(tmp_path):
    (tmp_path / "z.py").write_text("alpha\nfoo_bar\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("foo baz\nfoo\n", encoding="utf-8")

    result = search(tmp_path, query="foo(_bar)?|baz")

    assert result.ok
    assert result.data == {
        "query": "foo(_bar)?|baz",
        "matches": [
            {"path": "a.py", "line": 1, "preview": "foo baz"},
            {"path": "a.py", "line": 2, "preview": "foo"},
            {"path": "z.py", "line": 2, "preview": "foo_bar"},
        ],
    }


def test_search_glob_is_filename_filter_and_empty_search_is_success(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = search(tmp_path, query="needle", glob="*.py")
    empty = search(tmp_path, query="absent")

    assert [item["path"] for item in result.data["matches"]] == ["src/module.py"]
    assert empty.ok and empty.data["matches"] == [] and not empty.truncated


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": "  "},
        {"query": "x\ny"},
        {"query": "x\ry"},
        {"query": "\x00"},
        {"query": "x" * 4097},
        {"query": None},
        {"query": 1},
        {"query": "\ud800"},
        {"query": "é" * 2049},
        {"query": "x", "extra": 1},
        {"query": "x", "glob": "!*.py"},
        {"query": "x", "glob": "../*.py"},
        {"query": "x", "glob": "C:\\*.py"},
        {"query": "x", "glob": ""},
        {"query": "x", "max_results": True},
        {"query": "x", "max_results": None},
        {"query": "x", "max_results": "1"},
        {"query": "x", "max_results": 0},
        {"query": "x", "max_results": 201},
        {"query": "x", "glob": None},
        {"query": "x", "glob": 1},
    ],
)
def test_search_rejects_invalid_requests_before_repository_access(tmp_path, arguments):
    result = search(tmp_path, **arguments)

    assert not result.ok and result.error_code == "invalid_arguments"


def test_search_invalid_regex_is_invalid_arguments(tmp_path):
    result = search(tmp_path, query="(")

    assert not result.ok and result.error_code == "invalid_arguments"


def test_search_query_validation_precedes_workspace_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "harness.tools.dispatcher._workspace",
        lambda _root: pytest.fail("workspace was resolved before validation"),
    )

    result = search(tmp_path / "missing", query="bad\nquery")

    assert not result.ok and result.error_code == "invalid_arguments"


def test_search_accepts_query_at_exact_utf8_byte_limit(monkeypatch, tmp_path):
    calls = iter([(b"", b"", False, 1), (b"", b"", False, 1)])
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", lambda *_a, **_k: next(calls))

    result = search(tmp_path, query="é" * 2048)

    assert result.ok


def test_search_handles_flag_like_query_as_pattern_data(tmp_path):
    (tmp_path / "flags.txt").write_text("--version\n", encoding="utf-8")

    result = search(tmp_path, query="--version")

    assert result.ok
    assert result.data["matches"] == [
        {"path": "flags.txt", "line": 1, "preview": "--version"}
    ]


def test_search_does_not_hide_stderr_as_empty_results(tmp_path, monkeypatch):
    calls = iter([(b"", b"", False, 1), (b"", b"ripgrep failed", False, 0)])
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", lambda *_a, **_k: next(calls))

    result = search(tmp_path, query="needle")

    assert not result.ok and result.error_code == "io_error"


def test_search_missing_rg_is_a_controlled_error(tmp_path, monkeypatch):
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: None)

    result = search(tmp_path, query="needle")

    assert not result.ok and result.error_code == "missing_rg"


def test_search_uses_one_deadline_for_regex_preflight_and_search(tmp_path, monkeypatch):
    deadlines = []

    def fake_run(_command, _root, _env, deadline, **_kwargs):
        deadlines.append(deadline)
        return b"", b"", False, 1 if len(deadlines) == 1 else 1

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", fake_run)

    result = search(tmp_path, query="needle")

    assert result.ok
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]


def _capture_processes(monkeypatch):
    processes = []
    real_popen = search_module.subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(search_module.subprocess, "Popen", capture)
    return processes


@pytest.mark.parametrize("descriptor", [1, 2])
def test_run_stream_caps_bound_both_pipes_and_reap_child(
    tmp_path, monkeypatch, descriptor
):
    processes = _capture_processes(monkeypatch)
    limit = 32
    script = f"import os,time; os.write({descriptor}, b'x'*500); time.sleep(10)"

    stdout, stderr, capped, code = search_module._run(
        [sys.executable, "-c", script],
        tmp_path,
        os.environ.copy(),
        time.monotonic() + 2,
        stdout_limit=limit if descriptor == 1 else 1024,
        stderr_limit=limit if descriptor == 2 else 1024,
    )

    process = processes[0]
    assert capped and code != 0
    assert len(stdout) <= (limit if descriptor == 1 else 1024)
    assert len(stderr) <= (limit if descriptor == 2 else 1024)
    assert process.returncode is not None
    assert process.stdout.closed and process.stderr.closed


def test_run_count_callback_stops_and_reaps_active_child(tmp_path, monkeypatch):
    processes = _capture_processes(monkeypatch)
    records = []

    def stop_after_record(line):
        records.append(line)
        return True

    result = search_module._run(
        [
            sys.executable,
            "-c",
            "import time; print('event', flush=True); time.sleep(10)",
        ],
        tmp_path,
        os.environ.copy(),
        time.monotonic() + 2,
        stdout_limit=1024,
        stderr_limit=1024,
        on_stdout_line=stop_after_record,
    )

    process = processes[0]
    assert records == [b"event"]
    assert result[2] and result[3] != 0
    assert process.returncode is not None
    assert process.stdout.closed and process.stderr.closed


def test_run_callback_exception_reaps_child(tmp_path, monkeypatch):
    processes = _capture_processes(monkeypatch)

    with pytest.raises(ValueError, match="parse failed"):
        search_module._run(
            [
                sys.executable,
                "-c",
                "import time; print('event', flush=True); time.sleep(10)",
            ],
            tmp_path,
            os.environ.copy(),
            time.monotonic() + 2,
            stdout_limit=1024,
            stderr_limit=1024,
            on_stdout_line=lambda _line: (_ for _ in ()).throw(
                ValueError("parse failed")
            ),
        )

    process = processes[0]
    assert process.returncode is not None
    assert process.stdout.closed and process.stderr.closed


def test_run_deadline_uses_fake_clock_and_reaps_child(tmp_path, monkeypatch):
    processes = _capture_processes(monkeypatch)
    monkeypatch.setattr(search_module.time, "monotonic", lambda: 2.0)

    with pytest.raises(subprocess.TimeoutExpired):
        search_module._run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path,
            os.environ.copy(),
            1.0,
            stdout_limit=1024,
            stderr_limit=1024,
        )

    process = processes[0]
    assert process.returncode is not None
    assert process.stdout.closed and process.stderr.closed


def test_search_rejects_malformed_complete_event_as_io_error(tmp_path, monkeypatch):
    calls = iter([(b"", b"", False, 1), (b"{bad\n", b"", False, 0)])

    def fake_run(*_args, **kwargs):
        result = next(calls)
        callback = kwargs.get("on_stdout_line")
        if callback is not None:
            callback(b"{bad")
        return result

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", fake_run)

    result = search(tmp_path, query="needle")

    assert not result.ok and result.error_code == "io_error"


def test_search_discards_only_cut_event_and_marks_partial_output_truncated(
    tmp_path, monkeypatch
):
    calls = iter([(b"", b"", False, 1), (b'{"type":"match"', b"", True, -9)])

    def fake_run(*_args, **kwargs):
        result = next(calls)
        # No complete JSON line reaches the callback; the final event is cut.
        return result

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", fake_run)

    result = search(tmp_path, query="needle")

    assert result.ok and result.truncated
    assert result.data == {"query": "needle", "matches": []}


def test_exit_code_two_is_error_even_when_output_was_capped(tmp_path, monkeypatch):
    calls = iter([(b"", b"", False, 1), (b"", b"rg error", True, 2)])
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/rg")
    monkeypatch.setattr(search_module, "_run", lambda *_a, **_k: next(calls))

    result = search(tmp_path, query="needle")

    assert not result.ok and result.error_code == "io_error"


def test_search_skips_hidden_ignored_generated_external_and_large_files(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("needle\n", encoding="utf-8")
    for folder in ("dist", "node_modules", ".venv"):
        generated = tmp_path / folder
        generated.mkdir()
        (generated / "generated.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "large.py").write_bytes(b"needle\n" + b"x" * (1024 * 1024))
    outside = tmp_path.parent / "external.py"
    outside.write_text("needle\n", encoding="utf-8")
    (tmp_path / "escape.py").symlink_to(outside)

    result = search(tmp_path, query="needle")

    assert [item["path"] for item in result.data["matches"]] == ["visible.py"]


def test_search_caps_match_count_and_preview_bytes(tmp_path):
    (tmp_path / "many.txt").write_text("needle " + "é" * 400 + "\nneedle 2\nneedle 3\n")

    result = search(tmp_path, query="needle", max_results=1)

    assert result.truncated
    assert len(result.data["matches"]) == 1
    assert len(result.data["matches"][0]["preview"].encode("utf-8")) <= 512


def test_search_skips_non_utf8_matching_preview(tmp_path):
    (tmp_path / "invalid.txt").write_bytes(b"needle \xff\n")
    (tmp_path / "valid.txt").write_text("needle valid\n", encoding="utf-8")

    result = search(tmp_path, query="needle")

    assert result.ok
    assert [item["path"] for item in result.data["matches"]] == ["valid.txt"]


def test_tool_result_trims_search_matches_and_keeps_query_metadata():
    result = ToolResult(
        "search_code",
        True,
        {
            "query": "q" * 4096,
            "matches": [
                {"path": f"src/{index:03}.py", "line": 1, "preview": "x" * 512}
                for index in range(200)
            ],
        },
    )

    encoded = json.dumps(result.to_dict(), ensure_ascii=True).encode()
    assert len(encoded) <= MAX_RESULT_BYTES
    assert result.truncated
    assert result.data["query"] == "q" * 4096
    assert result.data["matches"]


def test_search_is_dispatched_read_only(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("needle\n", encoding="utf-8")
    before = source.read_bytes()

    result = execute_tool(ToolAction("search_code", {"query": "needle"}), tmp_path)

    assert result.ok and source.read_bytes() == before
