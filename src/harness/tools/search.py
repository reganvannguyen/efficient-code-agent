"""Bounded, read-only lexical search using ripgrep."""

from __future__ import annotations

import base64
import json
import os
import selectors
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from harness.tools.dispatcher import ToolFailure, ToolResult
from harness.tools.repository import (
    EXCLUDED_DIRECTORIES,
    MAX_FILE_BYTES,
    _accept_path,
    _limit_utf8,
)

MAX_SEARCH_OUTPUT = 64 * 1024
MAX_SEARCH_STDERR = 4 * 1024
MAX_PREVIEW_BYTES = 512
SEARCH_TIMEOUT_SECONDS = 5


def search_code(root: Path, arguments: dict[str, object]) -> ToolResult:
    """Return bounded matching lines, with regex query and filename glob distinct."""
    query = arguments["query"]
    glob = arguments.get("glob")
    maximum = arguments.get("max_results", 50)
    rg = shutil.which("rg")
    if rg is None:
        raise ToolFailure("missing_rg", "ripgrep is not installed")
    env = os.environ.copy()
    env.pop("RIPGREP_CONFIG_PATH", None)
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
    base = [rg, "--no-config", "--no-require-git", "--no-ignore-global"]
    try:
        _, preflight_stderr, preflight_capped, preflight_code = _run(
            base + [f"--regexp={query}", "-"],
            root,
            env,
            deadline,
            stdout_limit=1024,
            stderr_limit=MAX_SEARCH_STDERR,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise ToolFailure("timeout", "search timed out") from None
    except FileNotFoundError:
        raise ToolFailure("missing_rg", "ripgrep is not installed") from None
    # rg exits 2 for a malformed regex; no matches from empty stdin is expected.
    if preflight_code == 2 or (preflight_capped and preflight_stderr):
        raise ToolFailure(
            "invalid_arguments", "query is not a valid ripgrep regular expression"
        )
    if preflight_code not in {0, 1}:
        raise ToolFailure("io_error", "ripgrep could not validate the query")

    command = base + ["--json", f"--max-filesize={MAX_FILE_BYTES}", f"--regexp={query}"]
    for directory in sorted(EXCLUDED_DIRECTORIES):
        command += ["--glob", f"!{directory}/**", "--glob", f"!**/{directory}/**"]
    command += ["--glob", "!.*", "--glob", "!**/.*"]
    command += ["--", "."]
    matches: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    clipped = False

    def collect(raw: bytes) -> bool:
        nonlocal clipped
        if _collect_match(raw, root, glob, seen, matches):
            clipped = True
        return len(matches) > int(maximum)

    try:
        output, stderr, capped, code = _run(
            command,
            root,
            env,
            deadline,
            stdout_limit=MAX_SEARCH_OUTPUT,
            stderr_limit=MAX_SEARCH_STDERR,
            on_stdout_line=collect,
        )
    except subprocess.TimeoutExpired:
        raise ToolFailure("timeout", "search timed out") from None
    except FileNotFoundError:
        raise ToolFailure("missing_rg", "ripgrep is not installed") from None
    if stderr or code == 2 or (code not in {0, 1} and not capped):
        raise ToolFailure("io_error", "ripgrep could not search repository files")
    matches = matches[: int(maximum)]
    matches.sort(key=lambda item: (str(item["path"]), int(item["line"])))
    return ToolResult(
        "search_code",
        True,
        {"query": query, "matches": matches},
        truncated=capped or clipped,
    )


def _collect_match(
    raw: bytes,
    root: Path,
    glob: object,
    seen: set[tuple[str, int]],
    matches: list[dict[str, object]],
) -> bool:
    try:
        event = json.loads(raw)
        if event.get("type") in {"begin", "end", "summary", "context"}:
            return False
        if event.get("type") != "match":
            raise ValueError("unknown search event")
        data = event["data"]
        path_data = data["path"]
        path = path_data.get("text")
        if path is None:
            path = os.fsdecode(base64.b64decode(path_data["bytes"], validate=True))
        line = data["line_number"]
        line_data = data["lines"]
        text = line_data.get("text")
        if text is None:
            text = base64.b64decode(line_data["bytes"], validate=True).decode("utf-8")
        if (
            not isinstance(line, int)
            or isinstance(line, bool)
            or line < 1
            or not isinstance(path, str)
            or not isinstance(text, str)
        ):
            raise ValueError("invalid matching line")
        relative = _accept_path(os.fsencode(root / path), root, None)
        if relative is None or (
            glob is not None and not Path(relative).match(str(glob))
        ):
            return False
        if (root / relative).stat().st_size > MAX_FILE_BYTES:
            return False
        key = (relative, line)
        if key in seen:
            return False
        seen.add(key)
        preview, clipped = _limit_utf8(text.rstrip("\r\n"), MAX_PREVIEW_BYTES)
        matches.append({"path": relative, "line": line, "preview": preview})
        return clipped
    except UnicodeDecodeError:
        return False
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError):
        raise ToolFailure(
            "io_error", "ripgrep returned malformed search output"
        ) from None


def _run(
    command: list[str],
    root: Path,
    env: dict[str, str],
    deadline: float,
    *,
    stdout_limit: int,
    stderr_limit: int,
    stdin: int = subprocess.DEVNULL,
    on_stdout_line: Callable[[bytes], bool] | None = None,
) -> tuple[bytes, bytes, bool, int]:
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    streams = {
        process.stdout: (bytearray(), stdout_limit),
        process.stderr: (bytearray(), stderr_limit),
    }
    selector = selectors.DefaultSelector()
    capped = False
    pending = bytearray()
    try:
        for stream in streams:
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, SEARCH_TIMEOUT_SECONDS)
            for key, _ in selector.select(min(remaining, 0.1)):
                stream = key.fileobj
                data, limit = streams[stream]
                chunk = os.read(stream.fileno(), min(8192, limit - len(data) + 1))
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                room = max(0, limit - len(data))
                data.extend(chunk[:room])
                if stream is process.stdout and on_stdout_line is not None:
                    pending.extend(chunk[:room])
                    while b"\n" in pending:
                        record, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        if on_stdout_line(bytes(record)):
                            capped = True
                            raise _OutputCap
                if len(chunk) > room:
                    capped = True
                    raise _OutputCap
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if pending and on_stdout_line is not None:
            capped = on_stdout_line(bytes(pending)) or capped
        return (
            bytes(streams[process.stdout][0]),
            bytes(streams[process.stderr][0]),
            capped,
            process.returncode,
        )
    except _OutputCap:
        if process.poll() is None:
            process.kill()
        process.wait()
        stderr_data, stderr_limit = streams[process.stderr]
        if not process.stderr.closed:
            while len(stderr_data) < stderr_limit:
                chunk = os.read(
                    process.stderr.fileno(), stderr_limit - len(stderr_data)
                )
                if not chunk:
                    break
                stderr_data.extend(chunk)
        return (
            bytes(streams[process.stdout][0]),
            bytes(streams[process.stderr][0]),
            True,
            process.returncode,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


class _OutputCap(Exception):
    """Internal signal for intentional output-bound termination."""
