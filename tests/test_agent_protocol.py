from pathlib import Path

import pytest

from harness.agent import (
    AgentRun,
    CompleteAction,
    InvalidAction,
    RunStatus,
    ToolAction,
    parse_action,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            '{"kind":"tool","name":"read_file","arguments":{"path":"a.py"}}',
            ToolAction(name="read_file", arguments={"path": "a.py"}),
        ),
        (
            ' { "kind": "complete", "summary": "Done." } \n',
            CompleteAction(summary="Done."),
        ),
    ],
)
def test_parse_valid_actions(text: str, expected: object) -> None:
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        '{"kind":"complete",',
        "[]",
        "null",
        '{"kind":"tool","name":"x","arguments":{}} {"kind":"complete","summary":"done"}',
        '```json\n{"kind":"complete","summary":"done"}\n```',
        'Here is the action: {"kind":"complete","summary":"done"}',
        '{"kind":"tool","name":"x","arguments":{},"extra":1}',
        '{"kind":"tool","arguments":{}}',
        '{"kind":"tool","name":"  ","arguments":{}}',
        '{"kind":"tool","name":"x","arguments":[]}',
        '{"kind":"complete","summary":"done","extra":1}',
        '{"kind":"complete"}',
        '{"kind":"complete","summary":"  "}',
        '{"kind":"context_edit","data":{}}',
        '{"kind":"invalid","reason":"bad"}',
        '{"kind":"tool","name":"x","arguments":{"v":1,"v":2}}',
        '{"kind":"tool","name":"x","arguments":{"nested":{"v":1,"v":2}}}',
        '{"kind":"tool","kind":"complete","name":"x","arguments":{}}',
        '{"kind":"tool","name":"x","arguments":{"v":NaN}}',
        '{"kind":"tool","name":"x","arguments":{"v":Infinity}}',
        '{"kind":"tool","name":"x","arguments":{"v":-Infinity}}',
        '{"name":"x","arguments":{}}',
        '{"kind":"tool","name":1,"arguments":{}}',
        '{"kind":"complete","summary":false}',
    ],
)
def test_malformed_responses_return_invalid_action(text: str) -> None:
    assert isinstance(parse_action(text), InvalidAction)


def test_unknown_tool_name_is_accepted_as_envelope() -> None:
    action = parse_action('{"kind":"tool","name":"future_tool","arguments":{}}')
    assert action == ToolAction(name="future_tool", arguments={})


def test_completion_parsing_does_not_mutate_run() -> None:
    run = AgentRun(task="inspect", workspace=Path("/tmp/repo"))
    before = (run.status, run.steps, run.model_calls, run.final_response)

    action = parse_action('{"kind":"complete","summary":"done"}')

    assert isinstance(action, CompleteAction)
    assert (run.status, run.steps, run.model_calls, run.final_response) == before
    assert run.status is RunStatus.PENDING
