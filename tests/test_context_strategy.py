"""Deterministic tests for swappable bounded-history and editable-context strategies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agent.events import RunEventType
from harness.agent.runtime import (
    _INSTRUCTIONS,
    _NEXT_ACTION,
    _build_prompt,
    run_agent,
)
from harness.agent.state import RunStatus
from harness.context import (
    BoundedHistoryStrategy,
    ContextInput,
    EditableContextStrategy,
    resolve_context_strategy,
)
from harness.llama_client import ChatResponse
from harness.storage.database import SqliteRunRecorder, load_run_record


def _response(payload: dict[str, object]) -> ChatResponse:
    return ChatResponse(json.dumps(payload))


def test_resolve_context_strategy_defaults_to_baseline_and_rejects_unknown() -> None:
    assert isinstance(resolve_context_strategy(None), BoundedHistoryStrategy)
    assert isinstance(
        resolve_context_strategy("bounded_history"), BoundedHistoryStrategy
    )
    assert isinstance(
        resolve_context_strategy("editable_context"), EditableContextStrategy
    )
    with pytest.raises(ValueError, match="Unknown context_strategy"):
        resolve_context_strategy("magic_vector_db")


def test_bounded_history_strategy_matches_baseline_byte_for_byte() -> None:
    history_a = ['{"action":"a","observation":"b"}']
    history_b = list(history_a)

    baseline = _build_prompt("Explain main.py", history_a, 24_576)
    via_strategy = BoundedHistoryStrategy().build_prompt(
        ContextInput(
            instructions=_INSTRUCTIONS,
            directive=_NEXT_ACTION,
            task="Explain main.py",
            history=history_b,
            byte_limit=24_576,
        )
    )
    assert baseline == via_strategy


def test_editable_context_strategy_includes_notes_excerpts_and_respects_budget() -> (
    None
):
    strategy = EditableContextStrategy(max_notes_bytes=32, max_excerpts_bytes=64)
    prompt = strategy.build_prompt(
        ContextInput(
            instructions="RULES",
            directive="NEXT",
            task="Fix bug",
            history=['{"action":1,"observation":2}'],
            byte_limit=4096,
            task_notes="Confirmed root cause in parser.py line 42.",
            repo_excerpts=("def parse():\n    return None",),
        )
    )
    assert prompt is not None
    assert "RULES\nOriginal task (immutable):\nFix bug" in prompt
    assert "Working notes:\nConfirmed root cause in parser.p" in prompt
    assert "Repository excerpts:\ndef parse():" in prompt
    assert len(prompt.encode("utf-8")) <= 4096

    # Tight budget drops oldest history while keeping pinned task/instructions
    tight = strategy.build_prompt(
        ContextInput(
            instructions="RULES",
            directive="NEXT",
            task="Fix bug",
            history=["x" * 500, "y" * 20],
            byte_limit=260,
            task_notes="short note",
        )
    )
    assert tight is not None
    assert len(tight.encode("utf-8")) <= 260
    assert "y" * 20 in tight


def test_same_transcript_runs_under_either_strategy_and_records_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text("print('hi')\n", encoding="utf-8")

    transcript = [
        {"kind": "tool", "name": "list_files", "arguments": {}},
        {"kind": "tool", "name": "read_file", "arguments": {"path": "hello.py"}},
        {"kind": "complete", "summary": "hello.py prints hi."},
    ]

    db_path = tmp_path / "traces.sqlite3"
    recorder = SqliteRunRecorder(db_path)

    for strategy_name in ("bounded_history", "editable_context"):
        actions = iter(transcript)
        run = run_agent(
            "Explain hello.py",
            repo,
            model=lambda _p, *, timeout: _response(next(actions)),
            recorder=recorder,
            context_strategy=strategy_name,
        )
        assert run.status == RunStatus.COMPLETED
        assert run.final_response == "hello.py prints hi."
        started_events = [
            e for e in run.events if e.event_type == RunEventType.RUN_STARTED
        ]
        assert len(started_events) == 1
        assert started_events[0].data["context_strategy"] == strategy_name

        record = load_run_record(db_path, run.run_id)
        assert record is not None
        manifest_path = Path(record["run"]["artifacts_dir"]) / "manifest.json"
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_config = saved_manifest["config"]
        assert saved_config["context_strategy"] == strategy_name
        assert saved_config["context_strategy_parameters"]["strategy"] == strategy_name


def test_editable_context_strategy_adds_soft_nudge_only_when_needed() -> None:
    strategy = EditableContextStrategy()
    empty_template = (
        "# Working Context\n\n## Goal\nTask\n\n"
        "## Confirmed Findings\n\n## Current Hypothesis\n\n"
        "## Next Action\n\n## Open Questions\n"
    )

    early = strategy.build_prompt(
        ContextInput(
            instructions="RULES",
            directive="NEXT",
            task="Task",
            history=[],
            byte_limit=4096,
            steps=1,
            task_notes=empty_template,
        )
    )
    assert early is not None
    assert "[Context hint:" not in early

    nudged = strategy.build_prompt(
        ContextInput(
            instructions="RULES",
            directive="NEXT",
            task="Task",
            history=[],
            byte_limit=4096,
            steps=2,
            task_notes=empty_template,
        )
    )
    assert nudged is not None
    assert "[Context hint: working.md has no saved findings yet" in nudged

    populated = strategy.build_prompt(
        ContextInput(
            instructions="RULES",
            directive="NEXT",
            task="Task",
            history=[],
            byte_limit=4096,
            steps=5,
            task_notes=empty_template.replace(
                "## Confirmed Findings\n",
                "## Confirmed Findings\n- Found bug in a.py:1\n",
            ),
        )
    )
    assert populated is not None
    assert "[Context hint:" not in populated
