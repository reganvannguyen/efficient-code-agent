"""Strict parsing for normalized model actions."""

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
class CompleteAction:
    """A model proposal that the task is complete."""

    summary: str
    kind: Literal["complete"] = field(default="complete", init=False)


@dataclass(frozen=True)
class InvalidAction:
    """A malformed or unsupported model response."""

    reason: str
    kind: Literal["invalid"] = field(default="invalid", init=False)


Action = Union[ToolAction, CompleteAction, InvalidAction]


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


def parse_action(text: str) -> Action:
    """Parse one exact action object; malformed responses become InvalidAction."""
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
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
        return ToolAction(name=name, arguments=arguments)

    if kind == "complete":
        if payload.keys() != {"kind", "summary"}:
            return InvalidAction("complete action must contain exactly kind, summary")
        summary = payload["summary"]
        if not isinstance(summary, str) or not summary.strip():
            return InvalidAction("completion summary must be a non-blank string")
        return CompleteAction(summary=summary)

    return InvalidAction(f"unknown action kind: {kind!r}")
