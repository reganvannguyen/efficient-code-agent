"""Read-only repository discovery and bounded file reading."""

from __future__ import annotations

import hashlib
import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path

from harness.tools.dispatcher import ToolFailure, ToolResult

MAX_FILE_BYTES = 1024 * 1024
MAX_CONTENT_BYTES = 16 * 1024
MAX_LINES = 200
MAX_LIST_OUTPUT = 64 * 1024
MAX_LIST_PATHS = 200
LIST_TIMEOUT_SECONDS = 5
EXCLUDED_DIRECTORIES = {"node_modules", ".venv", "__pycache__", "dist", "build"}


def list_files(root: Path, arguments: dict[str, object]) -> ToolResult:
    """List sorted workspace-relative files using ripgrep's ignore rules."""
    directory = arguments.get("directory", ".")
    pattern = arguments.get("pattern")
    try:
        base = _resolve(root, directory)
    except ToolFailure as exc:
        if exc.code == "not_found" and directory != ".":
            raise ToolFailure(
                "not_found",
                "directory does not exist; omit directory (uses '.') to list workspace root",
            ) from None
        raise
    if not base.is_dir():
        raise ToolFailure("wrong_type", "directory must name a directory")
    rg = shutil.which("rg")
    if rg is None:
        raise ToolFailure("missing_rg", "ripgrep is not installed")
    command = [
        rg,
        "--no-config",
        "--no-require-git",
        "--no-ignore-global",
        "--files",
        "--null",
    ]
    command.append(str(base))
    env = os.environ.copy()
    env.pop("RIPGREP_CONFIG_PATH", None)
    try:
        output, capped, returncode = _run_bounded(command, root, env, pattern)
    except subprocess.TimeoutExpired:
        raise ToolFailure("timeout", "file listing timed out") from None
    except FileNotFoundError:
        raise ToolFailure("missing_rg", "ripgrep is not installed") from None
    candidates = output.split(b"\0")
    if candidates and candidates[-1] == b"":
        candidates.pop()
    paths = [os.fsdecode(raw) for raw in candidates]
    if returncode not in {0, 1} and not capped:
        raise ToolFailure("io_error", "ripgrep could not list repository files")
    paths = sorted(set(paths))
    truncated = capped or len(paths) > MAX_LIST_PATHS
    paths = paths[:MAX_LIST_PATHS]
    message = None
    if not paths and pattern is not None:
        message = (
            "No visible files matched this Path.match glob. The '|' character is "
            "literal, not OR. Try separate filename patterns, a simpler glob, or "
            "omit pattern. An empty filtered result does not prove file absence."
        )
    return ToolResult(
        "list_files",
        True,
        {"directory": directory, "paths": paths},
        truncated=truncated,
        message=message,
    )


def read_file(root: Path, arguments: dict[str, object]) -> ToolResult:
    """Read a line range from a regular UTF-8 text file within the workspace."""
    path = arguments["path"]
    start = int(arguments.get("start_line", 1))
    end = int(arguments.get("end_line", start + MAX_LINES - 1))
    target = _resolve(root, path)
    try:
        if not target.is_file():
            raise ToolFailure("wrong_type", "path must name a regular file")
        with target.open("rb") as source:
            raw = source.read(MAX_FILE_BYTES + 1)
    except ToolFailure:
        raise
    except FileNotFoundError:
        raise ToolFailure("not_found", "file does not exist") from None
    except PermissionError:
        raise ToolFailure("unreadable", "file cannot be read") from None
    if len(raw) > MAX_FILE_BYTES:
        raise ToolFailure("file_too_large", "file exceeds the 1 MiB read limit")
    if b"\x00" in raw:
        raise ToolFailure("binary_file", "binary files cannot be read")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolFailure("encoding_error", "file is not valid UTF-8") from None
    lines = text.splitlines()
    total_lines = len(lines)
    selected = lines[start - 1 : end]
    numbered = "\n".join(
        f"{start + index}: {line}" for index, line in enumerate(selected)
    )
    numbered, output_cut = _limit_utf8(numbered, MAX_CONTENT_BYTES)
    returned_lines = numbered.splitlines()
    actual_start = start if returned_lines else None
    actual_end = start + len(returned_lines) - 1 if returned_lines else None
    range_omitted = total_lines > 0 and (start > 1 or end < total_lines)
    omitted_by_output = output_cut or len(returned_lines) < len(selected)
    relative = target.relative_to(root).as_posix()
    return ToolResult(
        "read_file",
        True,
        {
            "path": relative,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "total_lines": total_lines,
            "start_line": actual_start,
            "end_line": actual_end,
            "content": numbered,
        },
        truncated=range_omitted or omitted_by_output,
    )


def _run_bounded(
    command: list[str], root: Path, env: dict[str, str], pattern: str | None
) -> tuple[bytes, bool, int]:
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    output = bytearray()
    pending = bytearray()
    raw_bytes = 0
    accepted = 0
    capped = False
    deadline = time.monotonic() + LIST_TIMEOUT_SECONDS
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                raise subprocess.TimeoutExpired(command, LIST_TIMEOUT_SECONDS)
            events = selector.select(min(remaining, 0.1))
            if events:
                read_size = min(8192, MAX_LIST_OUTPUT - raw_bytes)
                if read_size <= 0:
                    capped = True
                    process.kill()
                    break
                chunk = os.read(process.stdout.fileno(), read_size)
                if not chunk:
                    break
                raw_bytes += len(chunk)
                pending.extend(chunk)
                while b"\0" in pending:
                    record, _, remainder = pending.partition(b"\0")
                    pending = bytearray(remainder)
                    accepted_path = _accept_path(record, root, pattern)
                    if accepted_path is not None:
                        output.extend(os.fsencode(accepted_path))
                        output.append(0)
                        accepted += 1
                        if accepted > MAX_LIST_PATHS:
                            break
                if accepted > MAX_LIST_PATHS or raw_bytes >= MAX_LIST_OUTPUT:
                    capped = True
                    process.kill()
                    break
            elif process.poll() is not None:
                break
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        process.wait()
        process.stdout.close()
    return bytes(output), capped, process.returncode


def _accept_path(raw: bytes, root: Path, pattern: str | None) -> str | None:
    try:
        candidate = Path(os.fsdecode(bytes(raw))).resolve(strict=True)
        relative = candidate.relative_to(root).as_posix()
        if not candidate.is_file():
            return None
        parts = Path(relative).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRECTORIES for part in parts):
            return None
        return relative if pattern is None or Path(relative).match(pattern) else None
    except PermissionError:
        raise ToolFailure("unreadable", "path cannot be inspected") from None
    except (OSError, RuntimeError, ValueError):
        return None


def _resolve(root: Path, path: str) -> Path:
    try:
        candidate = (root / path).resolve(strict=True)
        candidate.relative_to(root)
        return candidate
    except FileNotFoundError:
        raise ToolFailure("not_found", "path does not exist") from None
    except PermissionError:
        raise ToolFailure("unreadable", "path cannot be inspected") from None
    except (OSError, RuntimeError, ValueError):
        raise ToolFailure(
            "path_denied", "path must stay within the workspace"
        ) from None


def _limit_utf8(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True
