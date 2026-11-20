import hashlib
import json
import os
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from harness.agent.protocol import ToolAction
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.llama_client import ChatResponse
from harness.tools import testing
from harness.tools.dispatcher import execute_tool
from harness.tools.policy import ApprovalRequest, ToolMode, ToolPolicy
from harness.tools.testing import run_tests


def _response(value: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(value))


def _tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"kind": "tool", "name": name, "arguments": arguments}


def test_approved_edit_and_tests_use_exact_prepared_actions(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("old\n")
    (tmp_path / "test_sample.py").write_text("def test_pass():\n    assert True\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "source.txt",
                    "expected_sha256": digest,
                    "old_text": "old",
                    "new_text": "new",
                },
            ),
            _tool("run_tests", {"preset": "pytest", "target": "test_sample.py"}),
            {"kind": "complete", "summary": "done"},
        ]
    )
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        with pytest.raises(FrozenInstanceError):
            request.path = "other"  # type: ignore[misc]
        return True

    run = run_agent(
        "edit and test",
        tmp_path,
        model=lambda _prompt, *, timeout: _response(next(actions)),
        policy=ToolPolicy(test_targets=("./test_sample.py/",)),
        approve=approve,
    )
    assert run.status is RunStatus.COMPLETED
    assert (tmp_path / "source.txt").read_text() == "new\n"
    assert run.changed_paths == ["source.txt"]
    assert [request.tool for request in requests] == ["edit_file", "run_tests"]
    assert requests[0].old_text == "old" and requests[0].new_text == "new"
    assert requests[1].argv[-1] == "test_sample.py"
    assert run.last_tool_result is not None
    assert run.last_tool_result["data"]["exit_code"] == 0  # type: ignore[index]


def test_edit_is_revalidated_after_approval(tmp_path: Path) -> None:
    target = tmp_path / "source.txt"
    target.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "source.txt",
                    "expected_sha256": digest,
                    "old_text": "old",
                    "new_text": "new",
                },
            ),
            {"kind": "complete", "summary": "done"},
        ]
    )

    def approve(_request: ApprovalRequest) -> bool:
        target.write_text("concurrent\n")
        return True

    run = run_agent(
        "edit",
        tmp_path,
        model=lambda _prompt, *, timeout: _response(next(actions)),
        policy=ToolPolicy(),
        approve=approve,
    )
    assert target.read_text() == "concurrent\n"
    assert run.last_tool_result is not None
    assert run.last_tool_result["error_code"] == "conflict"
    assert run.changed_paths == []


@pytest.mark.parametrize("stop_kind", ["cancel", "deadline"])
def test_stop_during_approval_prevents_edit(
    tmp_path: Path, monkeypatch, stop_kind: str
) -> None:
    import harness.agent.runtime as runtime
    import harness.tools.dispatcher as dispatcher

    target = tmp_path / "source.txt"
    target.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    cancelled = False
    now = 0.0
    monkeypatch.setattr(runtime, "monotonic", lambda: now)
    monkeypatch.setattr(dispatcher.time, "monotonic", lambda: now)
    action = _tool(
        "edit_file",
        {
            "path": "source.txt",
            "expected_sha256": digest,
            "old_text": "old",
            "new_text": "new",
        },
    )

    def approve(_request: ApprovalRequest) -> bool:
        nonlocal cancelled, now
        if stop_kind == "cancel":
            cancelled = True
        else:
            now = 2.0
        return True

    run = run_agent(
        "edit",
        tmp_path,
        model=lambda _prompt, *, timeout: _response(action),
        policy=ToolPolicy(),
        approve=approve,
        cancelled=lambda: cancelled,
        run_timeout=1,
    )
    assert target.read_text() == "old\n"
    assert run.status is (
        RunStatus.CANCELLED if stop_kind == "cancel" else RunStatus.LIMIT_REACHED
    )


def test_applied_edit_audit_is_recorded_before_post_tool_cancel(
    tmp_path: Path, monkeypatch
) -> None:
    import harness.tools.editing as editing

    target = tmp_path / "source.txt"
    target.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    cancelled = False
    apply = editing.apply_prepared_edit

    def apply_then_cancel(prepared):
        nonlocal cancelled
        result = apply(prepared)
        cancelled = True
        return result

    monkeypatch.setattr(editing, "apply_prepared_edit", apply_then_cancel)
    action = _tool(
        "edit_file",
        {
            "path": "source.txt",
            "expected_sha256": digest,
            "old_text": "old",
            "new_text": "new",
        },
    )
    run = run_agent(
        "edit",
        tmp_path,
        model=lambda _prompt, *, timeout: _response(action),
        policy=ToolPolicy(edit_file=ToolMode.AUTO),
        cancelled=lambda: cancelled,
    )
    assert target.read_text() == "new\n"
    assert run.status is RunStatus.CANCELLED
    assert run.changed_paths == ["source.txt"]
    assert run.last_tool_result is not None
    assert run.last_tool_result["data"]["write_state"] == "applied"  # type: ignore[index]


def test_unknown_write_state_and_path_survive_later_completion(
    tmp_path: Path, monkeypatch
) -> None:
    import harness.tools.editing as editing
    from harness.tools.dispatcher import ToolResult

    target = tmp_path / "source.txt"
    target.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    apply = editing.apply_prepared_edit

    def apply_uncertain(prepared):
        applied = apply(prepared)
        return ToolResult(
            "edit_file",
            False,
            {
                **applied.data,
                "write_state": "unknown",
                "changed_paths": ["source.txt"],
            },
            error_code="io_error",
        )

    monkeypatch.setattr(editing, "apply_prepared_edit", apply_uncertain)
    actions = iter(
        [
            _tool(
                "edit_file",
                {
                    "path": "source.txt",
                    "expected_sha256": digest,
                    "old_text": "old",
                    "new_text": "new",
                },
            ),
            {"kind": "complete", "summary": "done"},
        ]
    )
    run = run_agent(
        "edit",
        tmp_path,
        model=lambda _prompt, *, timeout: _response(next(actions)),
        policy=ToolPolicy(edit_file=ToolMode.AUTO),
    )
    assert run.status is RunStatus.COMPLETED
    assert run.changed_paths == ["source.txt"]
    assert run.last_write_state == "unknown"
    assert run.last_tool_result is not None
    assert run.last_tool_result["data"]["write_state"] == "unknown"  # type: ignore[index]


def test_coding_prompt_and_schema_follow_enabled_policy(tmp_path: Path) -> None:
    from harness.agent.runtime import _build_prompt, action_schema

    policy = ToolPolicy(
        edit_file=ToolMode.DENY,
        test_targets=("./tests/",),
    )
    prompt = _build_prompt("task", [], 20_000, policy=policy)
    assert prompt is not None
    assert "edit_file is disabled by policy" in prompt
    assert 'Configured targets: ["tests"]' in prompt
    schema = action_schema(policy)
    names = [
        variant["properties"]["name"].get("const")
        for variant in schema["oneOf"]
        if "name" in variant["properties"]
    ]
    assert "edit_file" not in names
    assert "run_tests" in names


def test_ask_policy_without_approval_denies_direct_dispatch(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("a")
    action = ToolAction(
        "edit_file",
        {
            "path": "x.txt",
            "expected_sha256": hashlib.sha256(b"a").hexdigest(),
            "old_text": "a",
            "new_text": "b",
        },
    )
    result = execute_tool(action, tmp_path, policy=ToolPolicy())
    assert result.error_code == "permission_denied"
    assert target.read_text() == "a"


@pytest.mark.parametrize("decision", [False, None, 1, "raise"])
def test_edit_ask_refusal_and_callback_errors_never_write(
    tmp_path: Path, decision: object
) -> None:
    target = tmp_path / "x.txt"
    target.write_text("a")
    action = ToolAction(
        "edit_file",
        {
            "path": "x.txt",
            "expected_sha256": hashlib.sha256(b"a").hexdigest(),
            "old_text": "a",
            "new_text": "b",
        },
    )

    def approve(_request: ApprovalRequest) -> object:
        if decision == "raise":
            raise RuntimeError("approval failed")
        return decision

    result = execute_tool(action, tmp_path, policy=ToolPolicy(), approve=approve)  # type: ignore[arg-type]
    assert result.error_code == "permission_denied"
    assert target.read_text() == "a"


def test_deny_and_auto_policy_are_enforced_by_direct_dispatch(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("a")
    action = ToolAction(
        "edit_file",
        {
            "path": "x.txt",
            "expected_sha256": hashlib.sha256(b"a").hexdigest(),
            "old_text": "a",
            "new_text": "b",
        },
    )
    denied = execute_tool(
        action,
        tmp_path,
        policy=ToolPolicy(edit_file=ToolMode.DENY),
        approve=lambda _request: True,
    )
    assert denied.error_code == "permission_denied"
    applied = execute_tool(action, tmp_path, policy=ToolPolicy(edit_file=ToolMode.AUTO))
    assert applied.ok
    assert target.read_text() == "b"


def test_auto_edit_deadline_expiring_during_prepare_does_not_write(
    tmp_path: Path, monkeypatch
) -> None:
    import harness.tools.dispatcher as dispatcher
    import harness.tools.editing as editing

    target = tmp_path / "x.txt"
    target.write_text("a")
    action = ToolAction(
        "edit_file",
        {
            "path": "x.txt",
            "expected_sha256": hashlib.sha256(b"a").hexdigest(),
            "old_text": "a",
            "new_text": "b",
        },
    )
    now = 0.0
    monkeypatch.setattr(dispatcher.time, "monotonic", lambda: now)
    prepare = editing.prepare_edit

    def prepare_then_expire(root, arguments):
        nonlocal now
        prepared = prepare(root, arguments)
        now = 2.0
        return prepared

    monkeypatch.setattr(editing, "prepare_edit", prepare_then_expire)
    result = execute_tool(
        action,
        tmp_path,
        policy=ToolPolicy(edit_file=ToolMode.AUTO),
        deadline=1.0,
    )
    assert result.error_code == "timeout"
    assert target.read_text() == "a"


def test_test_dispatch_denies_unconfigured_and_arbitrary_targets(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_allowed.py").write_text("def test_ok():\n    assert True\n")
    policy = ToolPolicy(run_tests=ToolMode.AUTO, test_targets=("test_allowed.py",))
    denied = execute_tool(
        ToolAction(
            "run_tests", {"preset": "pytest", "target": "test_allowed.py::test_ok"}
        ),
        tmp_path,
        policy=policy,
    )
    assert denied.error_code in {"invalid_arguments", "permission_denied"}
    result = execute_tool(
        ToolAction("run_tests", {"preset": "pytest", "target": "test_allowed.py"}),
        tmp_path,
        policy=policy,
    )
    assert result.ok
    assert (
        execute_tool(
            ToolAction("run_tests", {"preset": "pytest", "target": "test_allowed.py"}),
            tmp_path,
        ).error_code
        == "permission_denied"
    )


def test_configured_test_target_is_checked_before_inference(tmp_path: Path) -> None:
    called = False

    def model(_prompt: str, *, timeout: float) -> ChatResponse:
        nonlocal called
        called = True
        return ChatResponse("{}")

    with pytest.raises(ValueError, match="test target"):
        run_agent(
            "task",
            tmp_path,
            model=model,
            policy=ToolPolicy(test_targets=("missing",)),
        )
    assert not called


@pytest.mark.parametrize("value", [True, 0, float("inf"), float("nan")])
def test_policy_timeout_rejects_bool_nonpositive_and_nonfinite(value: object) -> None:
    with pytest.raises(ValueError, match="test_timeout"):
        ToolPolicy(test_timeout=value)  # type: ignore[arg-type]


def test_cli_policy_modes_are_distinct() -> None:
    assert ToolPolicy().edit_file is ToolMode.ASK
    assert ToolPolicy().run_tests is ToolMode.ASK
    assert ToolPolicy(edit_file=ToolMode.AUTO).edit_file is ToolMode.AUTO


def test_pytest_failure_is_a_bounded_tool_observation(tmp_path: Path) -> None:
    (tmp_path / "test_fail.py").write_text("def test_fail():\n    assert False\n")
    result = run_tests(tmp_path, "test_fail.py", 10, deadline=time.monotonic() + 10)
    assert not result.ok
    assert result.error_code == "tests_failed"
    assert result.data["exit_code"] != 0
    assert result.data["preset"] == "pytest"
    assert result.data["target"] == "test_fail.py"


def test_pytest_timeout_retains_metadata_and_partial_output(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    print('started', flush=True)\n    time.sleep(5)\n"
    )
    result = run_tests(tmp_path, "test_slow.py", 0.2, deadline=time.monotonic() + 2)
    assert result.error_code == "timeout"
    assert result.data["timed_out"] is True
    assert result.data["preset"] == "pytest"
    assert len(json.dumps(result.to_dict(), ensure_ascii=True).encode()) <= 32 * 1024


def test_pytest_output_cap_sets_truncated_and_bounds_serialization(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_output.py").write_text(
        "def test_output():\n    print('x' * 50000)\n    assert False\n"
    )
    result = run_tests(tmp_path, "test_output.py", 10, deadline=time.monotonic() + 10)
    assert result.error_code == "output_limit"
    assert result.truncated
    assert len(json.dumps(result.to_dict(), ensure_ascii=True).encode()) <= 32 * 1024


def test_pytest_cancellation_stops_process_group(tmp_path: Path) -> None:
    (tmp_path / "test_wait.py").write_text(
        "import time\ndef test_wait():\n    print('started', flush=True)\n    time.sleep(10)\n"
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    result = run_tests(
        tmp_path,
        "test_wait.py",
        10,
        deadline=time.monotonic() + 10,
        cancelled=cancelled,
    )
    assert result.error_code == "cancelled"
    assert result.data["timed_out"] is False


def test_reader_failure_kills_and_reaps_pytest(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "test_wait.py").write_text(
        "import time\ndef test_wait():\n    time.sleep(10)\n"
    )
    original_popen = testing.subprocess.Popen
    processes = []
    original_selector = testing.selectors.DefaultSelector

    class BrokenSelector:
        def __init__(self):
            self.inner = original_selector()

        def register(self, *args):
            return self.inner.register(*args)

        def get_map(self):
            return self.inner.get_map()

        def select(self, **kwargs):
            return [
                (key, testing.selectors.EVENT_READ) for key in self.get_map().values()
            ]

        def unregister(self, *args):
            return self.inner.unregister(*args)

        def close(self):
            return self.inner.close()

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        monkeypatch.setattr(
            testing.os,
            "read",
            lambda *_args: (_ for _ in ()).throw(OSError("reader broke")),
        )
        return process

    monkeypatch.setattr(testing.subprocess, "Popen", popen)
    monkeypatch.setattr(testing.selectors, "DefaultSelector", BrokenSelector)
    monkeypatch.setattr(testing.selectors, "DefaultSelector", BrokenSelector)
    result = run_tests(tmp_path, "test_wait.py", 10, deadline=time.monotonic() + 10)
    assert result.error_code == "execution_error"
    assert processes and processes[0].poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_descendant_that_keeps_pipe_open(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    (tmp_path / "test_child.py").write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    result = run_tests(tmp_path, "test_child.py", 0.8, deadline=time.monotonic() + 2)
    assert result.error_code == "timeout"
    child_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_decoded_test_streams_keep_utf8_expansion_bounded() -> None:
    result = testing._test_result(
        {"preset": "pytest", "target": ".", "argv": ["python"]},
        1,
        bytearray(b"\xff" * (16 * 1024)),
        bytearray(),
        False,
        False,
        False,
    )
    assert len(result.data["stdout"].encode("utf-8")) <= 16 * 1024
    assert result.truncated
    assert len(json.dumps(result.to_dict(), ensure_ascii=True).encode()) <= 32 * 1024


def test_edit_file_auto_fills_sha256_and_rejects_unread_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import x\ndef test_x(): assert x == 3\n", encoding="utf-8"
    )
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_app.py",),
        auto_test_after_edit=False,
    )
    actions = iter(
        [
            # Attempt edit without reading first -> rejected
            _tool(
                "edit_file",
                {"path": "app.py", "old_text": "x = 1", "new_text": "x = 2"},
            ),
            # Read file -> harness records sha256
            _tool("read_file", {"path": "app.py"}),
            # Edit without expected_sha256 -> succeeds via auto-fill
            _tool(
                "edit_file",
                {"path": "app.py", "old_text": "x = 1", "new_text": "x = 2"},
            ),
            # Second chained edit without expected_sha256 -> succeeds via updated sha256
            _tool(
                "edit_file",
                {"path": "app.py", "old_text": "x = 2", "new_text": "x = 3"},
            ),
            _tool("run_tests", {"preset": "pytest", "target": "test_app.py"}),
            {"kind": "complete", "summary": "Updated x to 3"},
        ]
    )
    run = run_agent(
        "Update x",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )
    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality.value == "verified"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 3\n"


def test_auto_test_after_edit_verifies_without_extra_run_tests_turn(
    tmp_path: Path,
) -> None:
    (tmp_path / "calc.py").write_text("def val(): return 1\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import val\ndef test_val(): assert val() == 2\n",
        encoding="utf-8",
    )
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_calc.py",),
        auto_test_after_edit=True,
    )
    actions = iter(
        [
            _tool("read_file", {"path": "calc.py"}),
            _tool(
                "edit_file",
                {"path": "calc.py", "old_text": "return 1", "new_text": "return 2"},
            ),
            # Complete immediately without an explicit run_tests step
            {"kind": "complete", "summary": "Fixed val()"},
        ]
    )
    run = run_agent(
        "Fix val",
        tmp_path,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )
    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality.value == "verified"
    assert len(run.test_invocations) == 1
    assert run.test_invocations[0]["exit_code"] == 0


def test_edit_file_line_range_with_auto_sha_and_auto_test(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def val():\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text(
        "from app import val\n\ndef test_val():\n    assert val() == 2\n",
        encoding="utf-8",
    )
    actions = iter(
        [
            {"kind": "tool", "name": "read_file", "arguments": {"path": "app.py"}},
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "app.py",
                    "start_line": 1,
                    "end_line": 2,
                    "new_text": "def val():\n    return 2",
                },
            },
            {"kind": "complete", "summary": "Fixed val() using line range."},
        ]
    )
    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_app.py",),
        auto_test_after_edit=True,
    )
    run = run_agent(
        "Fix app.py",
        tmp_path,
        model=lambda prompt, **kwargs: ChatResponse(
            text=json.dumps(next(actions)),
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        ),
        policy=policy,
    )
    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality.value == "verified"
    assert (tmp_path / "app.py").read_text(
        encoding="utf-8"
    ) == "def val():\n    return 2\n"
