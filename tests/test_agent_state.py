from pathlib import Path

import pytest

from harness.agent import AgentRun, RunStatus


def test_run_defaults_and_instances_are_independent() -> None:
    first = AgentRun(task="inspect", workspace=Path("/tmp/one"))
    second = AgentRun(task="explain", workspace=Path("/tmp/two"))

    assert first.status is RunStatus.PENDING
    assert first.steps == first.model_calls == 0
    assert first.final_response is None
    first.steps += 1
    first.model_calls += 1
    assert second.steps == second.model_calls == 0


def test_run_allows_only_declared_transitions() -> None:
    run = AgentRun(task="inspect", workspace=Path("/tmp/repo"))
    run.transition_to(RunStatus.RUNNING)
    run.transition_to(RunStatus.COMPLETED)
    assert run.status is RunStatus.COMPLETED


_ALLOWED_TRANSITIONS = {
    (RunStatus.PENDING, RunStatus.RUNNING),
    (RunStatus.RUNNING, RunStatus.COMPLETED),
    (RunStatus.RUNNING, RunStatus.FAILED),
    (RunStatus.RUNNING, RunStatus.LIMIT_REACHED),
    (RunStatus.RUNNING, RunStatus.CANCELLED),
}


@pytest.mark.parametrize(
    "source,target",
    [
        (source, target)
        for source in RunStatus
        for target in RunStatus
        if (source, target) not in _ALLOWED_TRANSITIONS
    ],
)
def test_invalid_transition_does_not_change_status(
    source: RunStatus, target: RunStatus
) -> None:
    run = AgentRun(task="inspect", workspace=Path("/tmp/repo"), status=source)

    with pytest.raises(ValueError):
        run.transition_to(target)

    assert run.status is source


@pytest.mark.parametrize(
    "target",
    [
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.LIMIT_REACHED,
        RunStatus.CANCELLED,
    ],
)
def test_running_run_can_reach_each_terminal_state(target: RunStatus) -> None:
    run = AgentRun(task="inspect", workspace=Path("/tmp/repo"))
    run.transition_to(RunStatus.RUNNING)
    run.transition_to(target)
    assert run.status is target
