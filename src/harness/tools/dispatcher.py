"""Validate and dispatch the repository tools available to an agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from harness.agent.protocol import ToolAction

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


def execute_tool(action: ToolAction, workspace: Path) -> ToolResult:
    """Validate and run one supported read-only tool request."""
    if not isinstance(action, ToolAction):
        return _error("", "invalid_arguments", "expected a ToolAction")
    if not isinstance(action.name, str) or not action.name.strip():
        return _error("", "invalid_arguments", "tool name must be a non-blank string")
    if action.name not in {"list_files", "read_file", "search_code"}:
        return _error(action.name, "unknown_tool", "unsupported tool")
    if not isinstance(action.arguments, dict):
        return _error(action.name, "invalid_arguments", "arguments must be an object")
    try:
        _validate_schema(action.name, action.arguments)
        root = _workspace(workspace)
        from harness.tools.repository import list_files, read_file
        from harness.tools.search import search_code

        if action.name == "list_files":
            return list_files(root, action.arguments)
        if action.name == "search_code":
            return search_code(root, action.arguments)
        return read_file(root, action.arguments)
    except ToolFailure as error:
        return _error(action.name, error.code, error.message)
    except OSError:
        return _error(action.name, "io_error", "repository operation failed")


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


def _validate_schema(tool: str, arguments: dict[str, object]) -> None:
    allowed = (
        {"directory", "pattern"}
        if tool == "list_files"
        else {"query", "glob", "max_results"}
        if tool == "search_code"
        else {"path", "start_line", "end_line"}
    )
    required = {"path"} if tool == "read_file" else set()
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
    path = arguments["path"]
    if not isinstance(path, str) or not path.strip():
        raise ToolFailure("invalid_arguments", "path must be a non-blank string")
    _validate_relative(path)
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
