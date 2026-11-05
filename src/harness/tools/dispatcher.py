"""Validate and dispatch the repository tools available to an agent."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from harness.agent.protocol import ToolAction
from harness.tools.policy import ApprovalRequest, ToolMode, ToolPolicy

MAX_RESULT_BYTES = 32 * 1024


@dataclass(frozen=True)
class ToolResult:
    """A JSON-ready result from one repository tool."""

    tool: str
    ok: bool
    data: dict[str, object]
    truncated: bool = False
    error_code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """Keep the in-memory and serialized truncation states consistent."""
        tool_truncated = len(self.tool) > 64
        if tool_truncated:
            object.__setattr__(self, "tool", self.tool[:64])
        data = dict(self.data)
        truncated = self.truncated or tool_truncated
        result = self._as_dict(data, truncated)
        while _json_size(result) > MAX_RESULT_BYTES:
            matches = data.get("matches")
            content = data.get("content")
            paths = data.get("paths")
            if isinstance(content, str) and content:
                lines = content.splitlines(keepends=True)
                if len(lines) > 1:
                    content = "".join(lines[:-1])
                else:
                    content = content[: max(0, len(content) // 2)]
                data["content"] = content
                if "start_line" in data:
                    if not content:
                        data["start_line"] = None
                    data["end_line"] = (
                        data["start_line"] + len(content.splitlines()) - 1
                        if content and isinstance(data["start_line"], int)
                        else None
                    )
            elif isinstance(matches, list) and matches:
                data["matches"] = matches[: len(matches) // 2]
            elif isinstance(paths, list) and paths:
                data["paths"] = paths[: len(paths) // 2]
            else:
                data = {}
                result = self._as_dict(data, True)
                truncated = True
                break
            truncated = True
            result = self._as_dict(data, truncated)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "truncated", truncated)

    def to_dict(self) -> dict[str, object]:
        """Serialize the result while enforcing the final JSON byte limit."""
        return self._as_dict(self.data, self.truncated)

    def _as_dict(self, data: dict[str, object], truncated: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "tool": self.tool,
            "ok": self.ok,
            "data": data,
            "truncated": truncated,
        }
        if self.error_code is not None:
            result["error_code"] = self.error_code
        if self.message is not None:
            result["message"] = self.message
        return result


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))


def execute_tool(
    action: ToolAction,
    workspace: Path,
    *,
    allow_edits: bool = False,
    policy: ToolPolicy | None = None,
    approve: Callable[[ApprovalRequest], bool] | None = None,
    deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ToolResult:
    """Validate and run one repository tool request; edits require authorization."""
    if not isinstance(allow_edits, bool):
        return _error("", "invalid_arguments", "allow_edits must be a boolean")
    if policy is not None and not isinstance(policy, ToolPolicy):
        return _error("", "invalid_arguments", "policy must be a ToolPolicy or None")
    if policy is not None and allow_edits:
        return _error("", "invalid_arguments", "allow_edits conflicts with policy")
    if not isinstance(action, ToolAction):
        return _error("", "invalid_arguments", "expected a ToolAction")
    if not isinstance(action.name, str) or not action.name.strip():
        return _error("", "invalid_arguments", "tool name must be a non-blank string")
    if action.name not in {
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_tests",
    }:
        return _error(action.name, "unknown_tool", "unsupported tool")
    if not isinstance(action.arguments, dict):
        return _error(action.name, "invalid_arguments", "arguments must be an object")
    try:
        _validate_schema(action.name, action.arguments)
        if action.name == "edit_file" and not allow_edits and policy is None:
            raise ToolFailure("permission_denied", "file edits are not authorized")
        if action.name == "edit_file" and policy is not None:
            if policy.edit_file is ToolMode.DENY:
                raise ToolFailure(
                    "permission_denied", "file edits are denied by policy"
                )
            from harness.tools.editing import apply_prepared_edit, prepare_edit

            root = _workspace(workspace)
            prepared = prepare_edit(root, action.arguments)
            if isinstance(prepared, ToolResult):
                return prepared
            if policy.edit_file is ToolMode.ASK:
                preview = prepared.diff.encode("utf-8")
                diff = preview[: 16 * 1024].decode("utf-8", "ignore")
                request = ApprovalRequest(
                    "edit_file",
                    root,
                    prepared.path,
                    prepared.expected_sha256,
                    prepared.old_text,
                    prepared.new_text,
                    diff,
                    len(preview) > 16 * 1024,
                )
                if not _approve(approve, request):
                    return _error(
                        action.name, "permission_denied", "operation was not approved"
                    )
            stopped = _operation_stop(cancelled, deadline)
            if stopped is not None:
                return ToolResult(
                    "edit_file",
                    False,
                    {},
                    error_code=stopped,
                    message="operation stopped before edit",
                )
            return apply_prepared_edit(prepared)
        if action.name == "run_tests" and policy is None:
            raise ToolFailure("permission_denied", "test execution requires a policy")
        if (
            action.name == "run_tests"
            and policy is not None
            and policy.run_tests is ToolMode.DENY
        ):
            raise ToolFailure("permission_denied", "test execution is denied by policy")
        if action.name == "run_tests" and policy is not None:
            from harness.tools.testing import run_tests, validate_test_policy

            root = _workspace(workspace)
            targets = validate_test_policy(root, policy)
            target = str(action.arguments["target"])
            if action.arguments["preset"] != "pytest" or target not in targets:
                raise ToolFailure("permission_denied", "test target is not permitted")
            executable = sys.executable
            argv = (executable, "-m", "pytest", "-q", "--", target)
            if policy.run_tests is ToolMode.ASK and not _approve(
                approve,
                ApprovalRequest(
                    "run_tests",
                    root,
                    preset="pytest",
                    target=target,
                    executable=executable,
                    argv=argv,
                ),
            ):
                return _error(
                    action.name, "permission_denied", "operation was not approved"
                )
            stopped = _operation_stop(cancelled, deadline)
            if stopped is not None:
                return ToolResult(
                    "run_tests",
                    False,
                    {},
                    error_code=stopped,
                    message="operation stopped before tests",
                )
            if target not in validate_test_policy(root, policy):
                return _error(
                    action.name,
                    "permission_denied",
                    "test target changed during approval",
                )
            return run_tests(
                root,
                target,
                policy.test_timeout,
                deadline=deadline or float("inf"),
                cancelled=cancelled,
            )
        root = _workspace(workspace)
        from harness.tools.repository import list_files, read_file
        from harness.tools.search import search_code

        if action.name == "edit_file":
            from harness.tools.editing import edit_file

            return edit_file(root, action.arguments)
        if action.name == "run_tests":
            raise ToolFailure("permission_denied", "test execution requires a policy")

        if action.name == "list_files":
            return list_files(root, action.arguments)
        if action.name == "search_code":
            return search_code(root, action.arguments)
        return read_file(root, action.arguments)
    except ToolFailure as error:
        return _error(action.name, error.code, error.message)
    except OSError:
        return _error(action.name, "io_error", "repository operation failed")
    except ValueError:
        return _error(action.name, "invalid_arguments", "tool configuration is invalid")


class ToolFailure(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


def _workspace(workspace: Path) -> Path:
    if not isinstance(workspace, Path):
        raise ToolFailure("invalid_arguments", "workspace must be a path")
    try:
        root = workspace.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ToolFailure("workspace_error", "workspace is unavailable") from None
    if not root.is_dir():
        raise ToolFailure("workspace_error", "workspace must be a directory")
    return root


def _error(tool: str, code: str, message: str) -> ToolResult:
    return ToolResult(tool, False, {}, error_code=code, message=message)


def _approve(
    callback: Callable[[ApprovalRequest], bool] | None, request: ApprovalRequest
) -> bool:
    if callback is None:
        return False
    try:
        return callback(request) is True
    except (KeyboardInterrupt, Exception):
        return False


def _operation_stop(
    cancelled: Callable[[], bool] | None, deadline: float | None
) -> str | None:
    try:
        if cancelled is not None and cancelled():
            return "cancelled"
    except Exception:
        return "cancelled"
    if deadline is not None and time.monotonic() >= deadline:
        return "timeout"
    return None


def _validate_schema(tool: str, arguments: dict[str, object]) -> None:
    allowed = (
        {"directory", "pattern"}
        if tool == "list_files"
        else {"query", "glob", "max_results"}
        if tool == "search_code"
        else {"path", "start_line", "end_line"}
        if tool == "read_file"
        else {"preset", "target"}
        if tool == "run_tests"
        else {"path", "expected_sha256", "old_text", "new_text"}
    )
    required = (
        {"path", "expected_sha256", "old_text", "new_text"}
        if tool == "edit_file"
        else {"preset", "target"}
        if tool == "run_tests"
        else {"path"}
        if tool == "read_file"
        else set()
    )
    if arguments.keys() - allowed or required - arguments.keys():
        raise ToolFailure("invalid_arguments", "arguments do not match the tool schema")
    if tool == "list_files":
        directory = arguments.get("directory", ".")
        pattern = arguments.get("pattern")
        if not isinstance(directory, str) or not directory.strip():
            raise ToolFailure("invalid_arguments", "directory must be a non-blank path")
        if "pattern" in arguments and (
            not isinstance(pattern, str)
            or not pattern.strip()
            or pattern.startswith("!")
            or "\x00" in pattern
        ):
            raise ToolFailure("invalid_arguments", "pattern must be a positive glob")
        _validate_relative(directory)
        return
    if tool == "search_code":
        allowed = {"query", "glob", "max_results"}
        if arguments.keys() - allowed or "query" not in arguments:
            raise ToolFailure(
                "invalid_arguments", "arguments do not match the tool schema"
            )
        query = arguments["query"]
        if (
            not isinstance(query, str)
            or not query.strip()
            or "\x00" in query
            or "\n" in query
            or "\r" in query
        ):
            raise ToolFailure("invalid_arguments", "query must be one non-blank line")
        try:
            if len(query.encode("utf-8")) > 4096:
                raise ToolFailure("invalid_arguments", "query exceeds 4096 UTF-8 bytes")
        except UnicodeEncodeError:
            raise ToolFailure(
                "invalid_arguments", "query must be valid UTF-8"
            ) from None
        if "glob" in arguments:
            glob = arguments["glob"]
            if (
                not isinstance(glob, str)
                or not glob.strip()
                or glob.startswith("!")
                or "\x00" in glob
                or Path(glob).is_absolute()
                or PureWindowsPath(glob).is_absolute()
                or PureWindowsPath(glob).drive
                or ".." in Path(glob).parts
                or ".." in PureWindowsPath(glob).parts
            ):
                raise ToolFailure(
                    "invalid_arguments",
                    "glob must be a positive workspace-relative glob",
                )
        if "max_results" in arguments:
            maximum = arguments["max_results"]
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 1 <= maximum <= 200
            ):
                raise ToolFailure(
                    "invalid_arguments", "max_results must be an integer from 1 to 200"
                )
        return
    if tool == "run_tests":
        if (
            arguments.get("preset") != "pytest"
            or not isinstance(arguments.get("target"), str)
            or not arguments["target"]
        ):
            raise ToolFailure(
                "invalid_arguments", "run_tests requires preset pytest and a target"
            )
        return
    path = arguments["path"]
    if not isinstance(path, str) or not path.strip():
        raise ToolFailure("invalid_arguments", "path must be a non-blank string")
    _validate_relative(path)
    if tool == "edit_file":
        from harness.tools.editing import validate_edit_arguments

        validate_edit_arguments(arguments)
        return
    start = _line_number(arguments.get("start_line", 1), "start_line")
    end = _line_number(arguments.get("end_line", start + 199), "end_line")
    if end < start or end - start + 1 > 200:
        raise ToolFailure("invalid_arguments", "line range must contain 1 to 200 lines")


def _validate_relative(path: str) -> None:
    if "\x00" in path:
        raise ToolFailure("invalid_arguments", "path contains a NUL byte")
    value = Path(path)
    if (
        value.is_absolute()
        or PureWindowsPath(path).is_absolute()
        or bool(PureWindowsPath(path).drive)
        or any(part == ".." for part in value.parts)
    ):
        raise ToolFailure("path_denied", "path must stay within the workspace")


def _line_number(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolFailure("invalid_arguments", f"{name} must be a positive integer")
    return value
