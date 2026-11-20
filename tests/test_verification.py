import hashlib
import json
from pathlib import Path

from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.agent.verification import VerificationQuality
from harness.llama_client import ChatResponse
from harness.tools.policy import ToolMode, ToolPolicy


def _response(value: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(value))


def _tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"kind": "tool", "name": name, "arguments": arguments}


def test_unverified_completion_rejected_when_edits_lack_post_edit_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("print('hello')\n")
    digest = hashlib.sha256(b"print('hello')\n").hexdigest()
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=(".",),
    )

    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "hello.py",
                    "expected_sha256": digest,
                    "old_text": "print('hello')",
                    "new_text": "print('world')",
                },
            ),
            {"kind": "complete", "summary": "Premature complete"},
            {"kind": "complete", "summary": "Premature complete 2"},
            {"kind": "complete", "summary": "Premature complete 3"},
            {"kind": "complete", "summary": "Premature complete 4"},
        ]
    )

    run = run_agent(
        "Edit and complete",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.FAILED
    assert run.stop_reason == "no_progress"
    assert run.verification_quality is VerificationQuality.UNVERIFIED
    assert any(
        "Repeated unverified completion" in issue for issue in run.unresolved_issues
    )


def test_post_edit_passing_test_verifies_completion(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\ndef test_add(): assert add(1, 2) == 3\n"
    )
    digest = hashlib.sha256(b"def add(a, b): return a + b\n").hexdigest()
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_calc.py",),
    )

    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "calc.py",
                    "expected_sha256": digest,
                    "old_text": "return a + b",
                    "new_text": "return a + b  # optimized",
                },
            ),
            _tool("run_tests", {"preset": "pytest", "target": "test_calc.py"}),
            {"kind": "complete", "summary": "Tests verified and passing"},
        ]
    )

    run = run_agent(
        "Optimize add and test",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == "model_complete"
    assert run.verification_quality is VerificationQuality.VERIFIED
    assert run.last_edit_step is not None
    assert run.last_test_step is not None
    assert run.last_test_step > run.last_edit_step
    assert len(run.test_invocations) == 1
    assert run.candidate_diff != ""


def test_test_before_edit_does_not_verify_completion(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("val = 1\n")
    (tmp_path / "test_foo.py").write_text(
        "from foo import val\ndef test_val(): assert val == 1\n"
    )
    digest = hashlib.sha256(b"val = 1\n").hexdigest()
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_foo.py",),
    )

    # Sequence: run_tests (step 1) -> edit_file (step 2) -> complete (step 3: rejected) -> ...
    actions = iter(
        [
            _tool("run_tests", {"preset": "pytest", "target": "test_foo.py"}),
            _tool(
                "edit_file",
                {
                    "path": "foo.py",
                    "expected_sha256": digest,
                    "old_text": "val = 1",
                    "new_text": "val = 2",
                },
            ),
            {"kind": "complete", "summary": "Done"},
            {"kind": "complete", "summary": "Done 2"},
            {"kind": "complete", "summary": "Done 3"},
            {"kind": "complete", "summary": "Done 4"},
        ]
    )

    run = run_agent(
        "Test then edit",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.FAILED
    assert run.stop_reason == "no_progress"
    assert run.verification_quality is VerificationQuality.UNVERIFIED


def test_failing_test_after_edit_rejects_completion(tmp_path: Path) -> None:
    (tmp_path / "bar.py").write_text("x = 1\n")
    (tmp_path / "test_bar.py").write_text(
        "from bar import x\ndef test_x(): assert x == 99\n"
    )
    digest = hashlib.sha256(b"x = 1\n").hexdigest()
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_bar.py",),
    )

    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "bar.py",
                    "expected_sha256": digest,
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                },
            ),
            _tool("run_tests", {"preset": "pytest", "target": "test_bar.py"}),
            {"kind": "complete", "summary": "Completed despite test failure"},
            {"kind": "complete", "summary": "Completed 2"},
            {"kind": "complete", "summary": "Completed 3"},
            {"kind": "complete", "summary": "Completed 4"},
        ]
    )

    run = run_agent(
        "Failing test",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.FAILED
    assert run.stop_reason == "no_progress"
    assert run.verification_quality is VerificationQuality.PARTIALLY_VERIFIED


def test_identical_tool_calls_stop_with_no_progress(tmp_path: Path) -> None:
    run = run_agent(
        "Loop forever",
        tmp_path,
        model=lambda _p, *, timeout: _response(_tool("list_files", {})),
        max_model_calls=10,
    )

    assert run.status is RunStatus.LIMIT_REACHED
    assert run.stop_reason == "no_progress"
    assert any(
        "Repeated identical tool call" in issue for issue in run.unresolved_issues
    )


def test_read_only_completion_is_unverified(tmp_path: Path) -> None:
    run = run_agent(
        "Explain repo",
        tmp_path,
        model=lambda _p, *, timeout: _response(
            {"kind": "complete", "summary": "Done explanation"}
        ),
    )

    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == "model_complete"
    assert run.verification_quality is VerificationQuality.UNVERIFIED


def test_coding_mode_without_edits_completes_as_unverified(tmp_path: Path) -> None:
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=(".",),
    )
    run = run_agent(
        "Answer question without edits",
        tmp_path,
        model=lambda _p, *, timeout: _response(
            {"kind": "complete", "summary": "Answer"}
        ),
        policy=policy,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == "model_complete"
    assert run.verification_quality is VerificationQuality.UNVERIFIED


def test_duplicate_tool_call_nudges_blocks_temporarily_and_reenables(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    seen_schemas: list[list[str]] = []
    actions = iter(
        [
            _tool("list_files", {}),
            _tool("list_files", {}),  # repeat #2 -> blocked duplicate nudge
            _tool("list_files", {}),  # repeat #3 -> temp block list_files on next turn
            _tool("read_file", {"path": "app.py"}),  # different tool -> clears block
            _tool("list_files", {}),  # re-enabled after using another tool
            {"kind": "complete", "summary": "Recovered after nudge and block."},
        ]
    )

    def _model(
        prompt: str,
        *,
        timeout: object = None,
        response_schema: object = None,
    ) -> ChatResponse:
        del prompt, timeout
        if isinstance(response_schema, dict):
            names = [
                v["properties"]["name"]["const"]
                for v in response_schema.get("oneOf", [])  # type: ignore[union-attr]
                if isinstance(v, dict)
                and isinstance(v.get("properties"), dict)
                and "name" in v["properties"]
            ]
            seen_schemas.append(names)
        return _response(next(actions))

    from functools import partial

    from harness.agent.runtime import READ_ONLY_ACTION_SCHEMA

    wrapped_model = partial(_model, response_schema=READ_ONLY_ACTION_SCHEMA)
    run = run_agent(
        "Inspect app.py",
        tmp_path,
        model=wrapped_model,
        max_model_calls=10,
        max_steps=10,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.final_response == "Recovered after nudge and block."
    # On turn 4 (index 3), list_files was temporarily omitted from response_schema
    assert "list_files" in seen_schemas[0]
    assert "list_files" not in seen_schemas[3]
    # On turn 5 (index 4), after calling read_file, list_files is restored
    assert "list_files" in seen_schemas[4]

    error_codes = [
        ev.data.get("error_code")
        for ev in run.events
        if ev.event_type.value == "tool_result"
    ]
    assert "duplicate_tool_call" in error_codes
    assert "tool_temporarily_blocked" in error_codes
