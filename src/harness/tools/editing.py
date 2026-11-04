"""Authorized, bounded, single-file text replacement."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import secrets
import stat
from pathlib import Path

from harness.tools.dispatcher import (
    MAX_RESULT_BYTES,
    ToolFailure,
    ToolResult,
    _json_size,
)
from harness.tools.repository import MAX_FILE_BYTES

MAX_DIFF_BYTES = 16 * 1024
PROTECTED = {
    ".git",
    ".codex",
    ".ssh",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
}


def validate_edit_arguments(arguments: dict[str, object]) -> None:
    """Validate edit fields and canonical relative path before workspace access."""
    path = arguments.get("path")
    expected = arguments.get("expected_sha256")
    old_text = arguments.get("old_text")
    new_text = arguments.get("new_text")
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ToolFailure("invalid_arguments", "path must be a canonical relative path")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        raise ToolFailure("invalid_arguments", "path must be valid UTF-8") from None
    parts = Path(path).parts
    if any(part == ".." for part in parts):
        raise ToolFailure("path_denied", "path must be a canonical relative path")
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None
    ):
        raise ToolFailure("invalid_arguments", "expected_sha256 must be 64 hex digits")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ToolFailure("invalid_arguments", "old_text and new_text must be strings")
    if "\x00" in old_text or "\x00" in new_text:
        raise ToolFailure("invalid_arguments", "edit text cannot contain NUL")
    try:
        old_bytes = old_text.encode("utf-8")
        new_bytes = new_text.encode("utf-8")
    except UnicodeEncodeError:
        raise ToolFailure(
            "invalid_arguments", "edit text must be valid UTF-8"
        ) from None
    if len(old_bytes) > MAX_FILE_BYTES or len(new_bytes) > MAX_FILE_BYTES:
        raise ToolFailure("invalid_arguments", "edit text exceeds the 1 MiB limit")
    for part in parts:
        lower = part.casefold()
        if lower in PROTECTED or lower == ".env" or lower.startswith(".env."):
            raise ToolFailure(
                "path_denied", "protected workspace paths cannot be edited"
            )


def edit_file(root: Path, arguments: dict[str, object]) -> ToolResult:
    """Replace one unique exact text fragment after checking its content hash."""
    path = str(arguments["path"])
    expected = str(arguments["expected_sha256"]).lower()
    old_text = str(arguments["old_text"])
    new_text = str(arguments["new_text"])
    target = root / path
    relative = Path(path).as_posix()
    try:
        _reject_links(root, target)
        info = target.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ToolFailure("wrong_type", "path must name a regular file")
        if info.st_nlink != 1:
            raise ToolFailure("path_denied", "hard-linked files cannot be edited")
        parent_info = target.parent.stat()
        if (
            not info.st_mode & 0o222
            or not os.access(target, os.W_OK)
            or not parent_info.st_mode & 0o222
            or not os.access(target.parent, os.W_OK)
        ):
            raise ToolFailure("unwritable", "file is not writable")
        if not info.st_size <= MAX_FILE_BYTES:
            raise ToolFailure("file_too_large", "file exceeds the 1 MiB edit limit")
        with target.open("rb") as stream:
            original = stream.read(MAX_FILE_BYTES + 1)
    except ToolFailure:
        raise
    except FileNotFoundError:
        raise ToolFailure("not_found", "file does not exist") from None
    except PermissionError:
        raise ToolFailure("unreadable", "file cannot be read") from None
    except OSError:
        raise ToolFailure("io_error", "file could not be inspected") from None
    if len(original) > MAX_FILE_BYTES:
        raise ToolFailure("file_too_large", "file exceeds the 1 MiB edit limit")
    if b"\x00" in original:
        raise ToolFailure("binary_file", "binary files cannot be edited")
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolFailure("encoding_error", "file is not valid UTF-8") from None
    actual_hash = hashlib.sha256(original).hexdigest()
    if actual_hash != expected:
        raise ToolFailure("conflict", "file content no longer matches expected_sha256")
    if old_text == "":
        if text != "":
            raise ToolFailure("conflict", "empty old_text only matches an empty file")
        start, end = 0, 0
    else:
        start = text.find(old_text)
        if start < 0 or text.find(old_text, start + 1) >= 0:
            raise ToolFailure("conflict", "old_text must occur exactly once")
        end = start + len(old_text)
    replacement = (text[:start] + new_text + text[end:]).encode("utf-8")
    if len(replacement) > MAX_FILE_BYTES:
        raise ToolFailure("invalid_arguments", "replacement exceeds the 1 MiB limit")
    if replacement == original:
        return _edit_result(relative, "unchanged", "", True)

    diff = _unified_diff(relative, text, replacement.decode("utf-8"))
    result = _edit_result(relative, "applied", diff, True)
    mode = stat.S_IMODE(info.st_mode)
    temp: Path | None = None
    mutation_attempted = False
    try:
        temp = _create_temp(target, replacement, mode)
        _reject_links(root, target)
        current_info = target.lstat()
        current = _read_target_bounded(target)
        if not _same_identity(info, current_info) or current != original:
            raise ToolFailure("conflict", "file changed while edit was being prepared")
        mutation_attempted = True
        try:
            os.replace(temp, target)
        except BaseException as error:
            return _reconcile(
                relative, original, replacement, target, error, mutation_attempted=True
            )
        temp = None
        return result
    except ToolFailure:
        raise
    except (OSError, KeyboardInterrupt) as error:
        if mutation_attempted:
            return _reconcile(
                relative, original, replacement, target, error, mutation_attempted=True
            )
        if isinstance(error, OSError):
            try:
                current = _read_target_bounded(target)
            except OSError:
                current = None
            if current is not None and current != original:
                return _reconcile(
                    relative,
                    original,
                    replacement,
                    target,
                    error,
                    mutation_attempted=False,
                )
        if isinstance(error, KeyboardInterrupt):
            return ToolResult(
                "edit_file",
                False,
                {},
                error_code="cancelled",
                message="edit was cancelled",
            )
        raise
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def _reject_links(root: Path, target: Path) -> None:
    current = root
    try:
        for part in target.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise ToolFailure("path_denied", "symbolic links cannot be edited")
    except ValueError:
        raise ToolFailure(
            "path_denied", "path must stay within the workspace"
        ) from None


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (before.st_dev, before.st_ino, before.st_mode, before.st_nlink) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
    )


def _create_temp(target: Path, contents: bytes, mode: int) -> Path:
    temp = target.parent / f".harness-edit-{secrets.token_hex(8)}"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise
    return temp


def _read_target_bounded(target: Path) -> bytes:
    try:
        _reject_links(Path(target.anchor), target)
    except ToolFailure as error:
        raise OSError("target cannot be safely reconciled") from error
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise OSError("target cannot be safely reconciled")
    with target.open("rb") as stream:
        value = stream.read(MAX_FILE_BYTES + 1)
    if len(value) > MAX_FILE_BYTES:
        raise OSError("target changed beyond the edit limit")
    return value


def _reconcile(
    relative: str,
    original: bytes,
    replacement: bytes,
    target: Path,
    error: BaseException,
    *,
    mutation_attempted: bool,
) -> ToolResult:
    try:
        actual = _read_target_bounded(target)
    except OSError:
        actual = None
    if actual == original:
        return _error_result(relative, error, "unchanged", "", False)
    if actual is not None:
        try:
            before_text = original.decode("utf-8")
            after_text = actual.decode("utf-8")
            diff = _unified_diff(relative, before_text, after_text)
        except UnicodeDecodeError:
            diff = ""
        state = "applied" if mutation_attempted and actual == replacement else "unknown"
        return _error_result(relative, error, state, diff, True)
    return _error_result(relative, error, "unknown", "", False)


def _error_result(
    relative: str,
    error: BaseException,
    state: str | None = None,
    diff: str = "",
    confirmed: bool = False,
) -> ToolResult:
    data, truncated = _audit_data(relative, state or "unknown", diff, confirmed)
    code = "cancelled" if isinstance(error, KeyboardInterrupt) else "io_error"
    message = "edit was cancelled" if code == "cancelled" else "file replacement failed"
    return ToolResult(
        "edit_file", False, data, truncated=truncated, error_code=code, message=message
    )


def _edit_result(relative: str, state: str, diff: str, confirmed: bool) -> ToolResult:
    data, truncated = _audit_data(relative, state, diff, confirmed)
    return ToolResult("edit_file", True, data, truncated=truncated)


def _audit_data(
    relative: str, state: str, diff: str, confirmed: bool
) -> tuple[dict[str, object], bool]:
    changed_paths = [relative] if confirmed and state in {"applied", "unknown"} else []
    data: dict[str, object] = {
        "path": relative,
        "changed_paths": changed_paths,
        "write_state": state,
        "diff": "",
    }
    if (
        _json_size({"tool": "edit_file", "ok": True, "data": data, "truncated": False})
        > MAX_RESULT_BYTES - 256
    ):
        raise ToolFailure(
            "result_too_large", "edit audit metadata exceeds the result budget"
        )
    candidate = diff
    was_truncated = False
    while candidate:
        encoded = candidate.encode("utf-8")
        if len(encoded) <= MAX_DIFF_BYTES:
            data["diff"] = candidate
            result = {
                "tool": "edit_file",
                "ok": True,
                "data": data,
                "truncated": was_truncated,
            }
            if _json_size(result) <= MAX_RESULT_BYTES - 256:
                break
        was_truncated = True
        candidate = encoded[: max(0, len(encoded) * 3 // 4)].decode("utf-8", "ignore")
    if candidate != diff:
        was_truncated = True
    data["diff"] = candidate
    return data, was_truncated


def _unified_diff(path: str, before: str, after: str) -> str:
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    generated = list(
        difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", n=3
        )
    )
    output: list[str] = []
    for line in generated:
        output.append(line)
        if line[:1] in {"+", "-", " "} and not line.endswith("\n"):
            output.append("\n\\ No newline at end of file\n")
    return "".join(output)
