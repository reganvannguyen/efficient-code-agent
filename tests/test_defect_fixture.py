import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from harness import cli
from harness.agent.runtime import run_agent
from harness.agent.state import RunStatus
from harness.agent.verification import VerificationQuality
from harness.llama_client import ChatResponse
from harness.tools.policy import ToolMode, ToolPolicy
from harness.tools.testing import run_tests

BROKEN_FIXTURE = Path(__file__).parent / "fixtures" / "broken_repo"


def copy_broken_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(BROKEN_FIXTURE, repo)
    return repo


def _response(value: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(value))


def _tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {"kind": "tool", "name": name, "arguments": arguments}


def test_defect_fixture_fails_initially(tmp_path: Path) -> None:
    repo = copy_broken_fixture(tmp_path)
    result = run_tests(repo, "test_calculator.py", timeout=30.0, deadline=1e9)
    assert not result.ok
    assert result.data["exit_code"] == 1
    assert "assert 95.0 == 105.0" in result.data["stdout"]


def test_agent_fixes_defect_through_edit_test_retry_loop(tmp_path: Path) -> None:
    repo = copy_broken_fixture(tmp_path)
    before_pristine = {
        p.name: p.read_bytes() for p in BROKEN_FIXTURE.iterdir() if p.is_file()
    }
    calc_path = repo / "calculator.py"
    initial_digest = hashlib.sha256(calc_path.read_bytes()).hexdigest()

    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_calculator.py",),
    )

    actions = iter(
        [
            # Step 1: Run tests (fails)
            _tool("run_tests", {"preset": "pytest", "target": "test_calculator.py"}),
            # Step 2: Read file to inspect code and SHA
            _tool("read_file", {"path": "calculator.py"}),
            # Step 3: Apply fix
            _tool(
                "edit_file",
                {
                    "path": "calculator.py",
                    "expected_sha256": initial_digest,
                    "old_text": "return round(subtotal - (subtotal * tax_rate), 2)",
                    "new_text": "return round(subtotal + (subtotal * tax_rate), 2)",
                },
            ),
            # Step 4: Re-run tests (passes)
            _tool("run_tests", {"preset": "pytest", "target": "test_calculator.py"}),
            # Step 5: Complete verified
            {
                "kind": "complete",
                "summary": "Fixed tax subtraction bug in calculate_total",
            },
        ]
    )

    run = run_agent(
        "Fix calculate_total to add tax instead of subtracting",
        repo,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.stop_reason == "model_complete"
    assert run.verification_quality is VerificationQuality.VERIFIED
    assert run.changed_paths == ["calculator.py"]
    assert (
        "return round(subtotal + (subtotal * tax_rate), 2)"
        in (repo / "calculator.py").read_text()
    )
    assert (
        "-    return round(subtotal - (subtotal * tax_rate), 2)" in run.candidate_diff
    )
    assert (
        "+    return round(subtotal + (subtotal * tax_rate), 2)" in run.candidate_diff
    )
    assert len(run.test_invocations) == 2
    assert run.test_invocations[0]["exit_code"] == 1
    assert run.test_invocations[1]["exit_code"] == 0

    # Ensure original fixture files were not modified
    after_pristine = {
        p.name: p.read_bytes() for p in BROKEN_FIXTURE.iterdir() if p.is_file()
    }
    assert before_pristine == after_pristine


def test_fixture_reset_and_isolation_prevents_leakage(tmp_path: Path) -> None:
    repo1 = copy_broken_fixture(tmp_path / "run1")
    (repo1 / "calculator.py").write_text("corrupted\n")

    repo2 = copy_broken_fixture(tmp_path / "run2")
    assert (repo2 / "calculator.py").read_text() != "corrupted\n"
    assert (repo2 / "calculator.py").read_bytes() == (
        BROKEN_FIXTURE / "calculator.py"
    ).read_bytes()


def test_invalid_patch_and_denial_recovery_in_fixture(tmp_path: Path) -> None:
    repo = copy_broken_fixture(tmp_path)
    calc_path = repo / "calculator.py"
    real_digest = hashlib.sha256(calc_path.read_bytes()).hexdigest()

    policy = ToolPolicy(
        edit_file=ToolMode.AUTO,
        run_tests=ToolMode.AUTO,
        test_targets=("test_calculator.py",),
    )

    actions = iter(
        [
            # Attempt 1: Edit with bad hash -> rejected (conflict)
            _tool(
                "edit_file",
                {
                    "path": "calculator.py",
                    "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "old_text": "return round(subtotal - (subtotal * tax_rate), 2)",
                    "new_text": "return round(subtotal + (subtotal * tax_rate), 2)",
                },
            ),
            # Attempt 2: Valid edit
            _tool(
                "edit_file",
                {
                    "path": "calculator.py",
                    "expected_sha256": real_digest,
                    "old_text": "return round(subtotal - (subtotal * tax_rate), 2)",
                    "new_text": "return round(subtotal + (subtotal * tax_rate), 2)",
                },
            ),
            # Step 3: Verify with tests
            _tool("run_tests", {"preset": "pytest", "target": "test_calculator.py"}),
            # Step 4: Complete
            {"kind": "complete", "summary": "Recovered and fixed"},
        ]
    )

    run = run_agent(
        "Recover from bad hash and fix",
        repo,
        model=lambda _p, *, timeout: _response(next(actions)),
        policy=policy,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.verification_quality is VerificationQuality.VERIFIED
    assert run.changed_paths == ["calculator.py"]


def test_cli_coding_run_on_defect_fixture(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = copy_broken_fixture(tmp_path)
    calc_path = repo / "calculator.py"
    initial_digest = hashlib.sha256(calc_path.read_bytes()).hexdigest()

    actions = iter(
        [
            _response(
                _tool("run_tests", {"preset": "pytest", "target": "test_calculator.py"})
            ),
            _response(
                _tool(
                    "edit_file",
                    {
                        "path": "calculator.py",
                        "expected_sha256": initial_digest,
                        "old_text": "return round(subtotal - (subtotal * tax_rate), 2)",
                        "new_text": "return round(subtotal + (subtotal * tax_rate), 2)",
                    },
                )
            ),
            _response(
                _tool("run_tests", {"preset": "pytest", "target": "test_calculator.py"})
            ),
            _response({"kind": "complete", "summary": "Fixed defect in calculator"}),
        ]
    )

    monkeypatch.setattr(cli, "chat", lambda _p, **_kw: next(actions))
    cli.main(
        [
            "run",
            "Fix tax calculation in calculator.py",
            "--repo",
            str(repo),
            "--coding",
            "--auto-approve",
            "--test-target",
            "test_calculator.py",
        ]
    )
    captured = capsys.readouterr()
    assert "status=completed" in captured.err
    assert "verification=verified" in captured.err
    assert "(verified completion)" in captured.err
    assert "candidate_diff:" in captured.err
    assert "tests_executed=2" in captured.err
    assert captured.out.strip() == "Fixed defect in calculator"


@pytest.mark.skipif(
    os.getenv("HARNESS_LIVE_SMOKE") != "1", reason="opt-in local model smoke"
)
def test_live_smoke_fixes_defect_fixture(tmp_path: Path, capsys) -> None:
    repo = copy_broken_fixture(tmp_path)
    cli.main(
        [
            "run",
            "Run tests to find the failing test in test_calculator.py, inspect calculator.py, "
            "fix the bug so calculate_total adds tax instead of subtracting, and re-run tests to verify.",
            "--repo",
            str(repo),
            "--coding",
            "--auto-approve",
            "--test-target",
            "test_calculator.py",
        ]
    )
    captured = capsys.readouterr()
    assert "status=completed" in captured.err
    assert "verification=verified" in captured.err
    assert "calculator.py" in captured.err
