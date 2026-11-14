"""Strict parsing for normalized model actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True)
class ToolAction:
    """A model request to invoke a named tool."""

    name: str
    arguments: dict[str, object]
    kind: Literal["tool"] = field(default="tool", init=False)


@dataclass(frozen=True)
class ContextEditAction:
    """A normalized model request to edit working-memory notes."""

    operation: str
    target: str
    content: str
    run_id: str | None = None
    kind: Literal["context_edit"] = field(default="context_edit", init=False)


@dataclass(frozen=True)
class CompleteAction:
    """A model proposal that the task is complete."""

    summary: str
    kind: Literal["complete"] = field(default="complete", init=False)


@dataclass(frozen=True)
class InvalidAction:
    """A malformed or unsupported model response."""

    reason: str
    kind: Literal["invalid"] = field(default="invalid", init=False)


Action = Union[ToolAction, ContextEditAction, CompleteAction, InvalidAction]


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_context_edit_fields(fields: dict[str, object]) -> Action:
    allowed = {"operation", "target", "content", "run_id"}
    required = {"operation", "target", "content"}
    keys = set(fields.keys())
    if not required.issubset(keys) or not keys.issubset(allowed):
        return InvalidAction(
            "update_context requires operation, target, and content (optional run_id)"
        )
    operation = fields["operation"]
    target = fields["target"]
    content = fields["content"]
    run_id = fields.get("run_id")
    if not isinstance(operation, str) or not operation.strip():
        return InvalidAction("update_context operation must be a non-blank string")
    if not isinstance(target, str) or not target.strip():
        return InvalidAction("update_context target must be a non-blank string")
    if not isinstance(content, str):
        return InvalidAction("update_context content must be a string")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        return InvalidAction("update_context run_id must be a non-blank string")
    return ContextEditAction(
        operation=operation,
        target=target,
        content=content,
        run_id=run_id,
    )


def parse_action(text: str) -> Action:
    """Parse one exact action object; malformed responses become InvalidAction."""
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as error:
        return InvalidAction(str(error))

    if not isinstance(payload, dict):
        return InvalidAction("response must be one JSON object")

    kind = payload.get("kind")
    if kind == "tool":
        if payload.keys() != {"kind", "name", "arguments"}:
            return InvalidAction(
                "tool action must contain exactly kind, name, arguments"
            )
        name = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(name, str) or not name.strip():
            return InvalidAction("tool name must be a non-blank string")
        if not isinstance(arguments, dict):
            return InvalidAction("tool arguments must be an object")
        if name == "update_context":
            return _parse_context_edit_fields(arguments)
        return ToolAction(name=name, arguments=arguments)

    if kind == "context_edit":
        body = {k: v for k, v in payload.items() if k != "kind"}
        return _parse_context_edit_fields(body)

    if kind == "complete":
        if payload.keys() != {"kind", "summary"}:
            return InvalidAction("complete action must contain exactly kind, summary")
        summary = payload["summary"]
        if not isinstance(summary, str) or not summary.strip():
            return InvalidAction("completion summary must be a non-blank string")
        return CompleteAction(summary=summary)

    return InvalidAction(f"unknown action kind: {kind!r}")
