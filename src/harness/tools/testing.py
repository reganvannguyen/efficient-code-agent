"""Bounded pytest execution for explicitly configured workspace targets."""

from __future__ import annotations

import math
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

from harness.tools.dispatcher import MAX_RESULT_BYTES, ToolResult, _json_size
from harness.tools.policy import ToolPolicy, normalize_target

MAX_STREAM_BYTES = 16 * 1024


def validate_test_policy(root: Path, policy: ToolPolicy) -> tuple[str, ...]:
    """Validate and canonicalize configured pytest targets before inference."""
    if not isinstance(policy, ToolPolicy):
        raise ValueError("policy must be a ToolPolicy")
    if not isinstance(policy.test_targets, tuple) or not policy.test_targets:
        raise ValueError("test_targets must be a non-empty tuple")
    targets: list[str] = []
    for target in policy.test_targets:
        if (
            not isinstance(target, str)
            or not target
            or target.startswith("-")
            or len(target.encode("utf-8", errors="replace")) > 1024
            or "::" in target
            or "\x00" in target
            or "\\" in target
        ):
            raise ValueError("test targets must be safe relative paths")
        candidate = Path(target)
        if (
            candidate.is_absolute()
            or PureWindowsPath(target).is_absolute()
            or bool(PureWindowsPath(target).drive)
            or ".." in candidate.parts
            or (candidate.parts and candidate.parts[0] == "-")
        ):
            raise ValueError("test targets must be safe relative paths")
        path = root
        for part in candidate.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("test targets cannot contain symlinks")
        try:
            canonical = (root / candidate).resolve(strict=True)
            canonical.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("test target must exist inside the workspace") from None
        if not canonical.is_file() and not canonical.is_dir():
            raise ValueError("test target must be a file or directory")
        targets.append(normalize_target(target))
    timeout = policy.test_timeout
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("test_timeout must be finite and positive")
    return tuple(targets)


def run_tests(
    root: Path,
    target: str,
    timeout: float,
    *,
    deadline: float,
    cancelled: object = None,
) -> ToolResult:
    """Run the fixed pytest preset, retaining bounded output and reaping groups."""
    argv = (sys.executable, "-m", "pytest", "-q", "--", target)
    base = {"preset": "pytest", "target": target, "argv": list(argv)}
    metadata = {
        "tool": "run_tests",
        "ok": False,
        "data": {
            **base,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        },
        "truncated": False,
        "error_code": "execution_error",
        "message": "pytest could not be started",
    }
    if _json_size(metadata) > MAX_RESULT_BYTES:
        return ToolResult(
            "run_tests",
            False,
            {
                "preset": "pytest",
                "target": target[:128],
                "argv": ["<omitted: metadata exceeds result limit>"],
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
            error_code="result_metadata_limit",
            message="pytest result metadata exceeds the output limit",
        )
    if os.name != "posix":
        return _test_result(
            base,
            None,
            bytearray(),
            bytearray(),
            False,
            False,
            False,
            forced_error_code="unsupported_platform",
        )
    remaining = min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        return _test_result(base, None, bytearray(), bytearray(), True, False, False)
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    out = bytearray()
    err = bytearray()
    truncated = False
    timed_out = False
    cancelled_run = False
    try:
        execution_deadline = time.monotonic() + remaining
        process = subprocess.Popen(
            argv,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        for stream, output in ((process.stdout, out), (process.stderr, err)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, output)
        kill_reason: str | None = None
        while selector.get_map() or process.poll() is None:
            if cancelled is not None and callable(cancelled) and cancelled():
                cancelled_run = True
                kill_reason = "cancelled"
            elif time.monotonic() >= execution_deadline or time.monotonic() >= deadline:
                timed_out = True
                kill_reason = "timeout"
            if kill_reason is not None:
                _kill_group(process)
                break
            for key, _ in selector.select(
                timeout=min(0.05, max(0.0, deadline - time.monotonic()))
            ):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer: bytearray = key.data
                remaining_cap = MAX_STREAM_BYTES - len(buffer)
                if len(chunk) > remaining_cap:
                    buffer.extend(chunk[:remaining_cap])
                    truncated = True
                    kill_reason = "output_limit"
                    _kill_group(process)
                    break
                buffer.extend(chunk)
            if kill_reason is not None:
                break
        _drain_and_reap(process, selector)
        selector = None
        return _test_result(
            base, process.returncode, out, err, timed_out, cancelled_run, truncated
        )
    except KeyboardInterrupt:
        if process is not None:
            _kill_group(process)
            _drain_and_reap(process, selector)
            selector = None
        return _test_result(base, None, out, err, False, True, truncated)
    except Exception as error:
        if process is not None:
            _kill_group(process)
            _drain_and_reap(process, selector)
            selector = None
        code = (
            "executable_unavailable"
            if isinstance(error, FileNotFoundError)
            else "execution_error"
        )
        return _test_result(
            base, None, out, err, False, False, truncated, forced_error_code=code
        )
    finally:
        if process is not None:
            _close_readers(process, selector)


def _test_result(
    base: dict[str, object],
    code: int | None,
    out: bytearray,
    err: bytearray,
    timed_out: bool,
    cancelled_run: bool,
    truncated: bool,
    forced_error_code: str | None = None,
) -> ToolResult:
    stdout = _clip_utf8(_decode(out), MAX_STREAM_BYTES)
    stderr = _clip_utf8(_decode(err), MAX_STREAM_BYTES)
    if stdout != _decode(out) or stderr != _decode(err):
        truncated = True
    data = {
        **base,
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
    }
    error_code = forced_error_code or (
        "cancelled"
        if cancelled_run
        else "timeout"
        if timed_out
        else "output_limit"
        if truncated
        else "tests_failed"
        if code
        else None
    )
    while (
        _json_size(
            {
                "tool": "run_tests",
                "ok": error_code is None and code == 0,
                "data": data,
                "truncated": truncated,
                **(
                    {"error_code": error_code, "message": error_code.replace("_", " ")}
                    if error_code
                    else {}
                ),
            }
        )
        > MAX_RESULT_BYTES
    ):
        if not data["stdout"] and not data["stderr"]:
            data["argv"] = ["<omitted: metadata exceeds result limit>"]
            data["target"] = str(data["target"])[:128]
            truncated = True
            if (
                _json_size(
                    {
                        "tool": "run_tests",
                        "ok": error_code is None and code == 0,
                        "data": data,
                        "truncated": True,
                    }
                )
                > MAX_RESULT_BYTES
            ):
                raise RuntimeError("test result metadata cannot fit its result budget")
            break
        key = "stdout" if len(data["stdout"]) >= len(data["stderr"]) else "stderr"
        value = data[key]
        assert isinstance(value, str)
        data[key] = _clip_utf8(value, max(0, len(value.encode("utf-8")) // 2))
        truncated = True
    return ToolResult(
        "run_tests",
        error_code is None and code == 0,
        data,
        truncated=truncated,
        error_code=error_code,
        message=None if error_code is None else error_code.replace("_", " "),
    )


def _decode(value: bytearray) -> str:
    return bytes(value).decode("utf-8", errors="replace")


def _clip_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")[:limit]
    return encoded.decode("utf-8", errors="ignore")


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _drain_and_reap(
    process: subprocess.Popen[bytes], selector: selectors.BaseSelector | None
) -> None:
    if selector is not None:
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
                key.fileobj.close()
            except OSError:
                pass
        selector.close()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill_group(process)


def _close_readers(
    process: subprocess.Popen[bytes], selector: selectors.BaseSelector | None
) -> None:
    if selector is not None:
        try:
            for key in list(selector.get_map().values()):
                try:
                    selector.unregister(key.fileobj)
                except (KeyError, OSError):
                    pass
            selector.close()
        except (OSError, AttributeError):
            pass
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
