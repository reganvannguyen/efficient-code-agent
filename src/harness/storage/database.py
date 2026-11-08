"""SQLite-backed trace and event persistence with degraded error handling."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from harness.agent.events import RunEvent
from harness.agent.state import AgentRun
from harness.storage.artifacts import write_run_artifacts
from harness.storage.recorder import LoggingPolicy, RunArtifacts, StorageError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    workspace TEXT NOT NULL,
    status TEXT NOT NULL,
    stop_reason TEXT,
    benchmark_outcome TEXT,
    verification_quality TEXT,
    model_calls INTEGER DEFAULT 0,
    steps INTEGER DEFAULT 0,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    start_time REAL NOT NULL,
    end_time REAL,
    duration_seconds REAL DEFAULT 0.0,
    artifacts_dir TEXT,
    patch_path TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    duration_ms REAL,
    data_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, sequence);
"""


class SqliteRunRecorder:
    """Records run lifecycle, ordered events, and artifacts to SQLite and disk."""

    def __init__(
        self,
        db_path: Path,
        artifacts_dir: Path | None = None,
        policy: LoggingPolicy = LoggingPolicy.DEGRADED,
    ) -> None:
        self.db_path = Path(db_path)
        self.artifacts_dir = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else self.db_path.parent / "artifacts"
        )
        self.policy = policy
        self.is_degraded = False
        self._manifest: dict[str, object] | None = None
        self._current_run: AgentRun | None = None

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.executescript(_SCHEMA)
        except Exception as exc:
            self._handle_error(exc, context="initializing SQLite schema")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level="DEFERRED",
        )
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _handle_error(
        self,
        exc: Exception,
        context: str,
        run: AgentRun | None = None,
    ) -> None:
        self.is_degraded = True
        msg = f"storage error during {context}: {exc}"
        if self.policy is LoggingPolicy.FAIL:
            raise StorageError(msg) from exc

        # Degraded policy: warn and record in run's unresolved issues
        target_run = run or self._current_run
        if target_run is not None:
            warning = f"sqlite_logging_degraded: {context}: {exc}"
            if warning not in target_run.unresolved_issues:
                target_run.unresolved_issues.append(warning)
        print(f"harness [storage warning]: {msg}", file=sys.stderr)

    def on_run_started(
        self,
        run: AgentRun,
        manifest: dict[str, object] | None = None,
    ) -> None:
        """Insert the initial run record and cache run metadata."""
        self._current_run = run
        self._manifest = manifest
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, task, workspace, status, stop_reason,
                        benchmark_outcome, verification_quality,
                        model_calls, steps, prompt_tokens,
                        completion_tokens, total_tokens,
                        start_time, end_time, duration_seconds,
                        artifacts_dir, patch_path, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?
                    )
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        workspace=excluded.workspace
                    """,
                    (
                        run.run_id,
                        run.task,
                        str(run.workspace),
                        run.status.value,
                        run.stop_reason,
                        (
                            run.benchmark_outcome.value
                            if run.benchmark_outcome
                            else None
                        ),
                        run.verification_quality.value,
                        run.model_calls,
                        run.steps,
                        run.total_prompt_tokens,
                        run.total_completion_tokens,
                        run.total_tokens,
                        run.start_time,
                        run.end_time,
                        run.duration_seconds,
                        None,
                        None,
                        time.time(),
                    ),
                )
        except Exception as exc:
            self._handle_error(exc, context="on_run_started", run=run)

    def on_event(self, event: RunEvent) -> None:
        """Insert an immutable event row into the events table."""
        try:
            data_str = json.dumps(event.data, sort_keys=True)
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, run_id, sequence, step,
                        timestamp, event_type, duration_ms, data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event.sequence,
                        event.step,
                        event.timestamp,
                        event.event_type.value,
                        event.duration_ms,
                        data_str,
                    ),
                )
        except Exception as exc:
            self._handle_error(exc, context=f"on_event({event.event_type.value})")

    def on_run_finished(self, run: AgentRun) -> RunArtifacts | None:
        """Write artifacts and update terminal metrics and status."""
        self._current_run = run
        artifacts: RunArtifacts | None = None
        try:
            run_artifacts_dir = self.artifacts_dir / run.run_id
            artifacts = write_run_artifacts(
                artifacts_dir=run_artifacts_dir,
                run=run,
                manifest=self._manifest,
            )
        except Exception as exc:
            self._handle_error(exc, context="write_run_artifacts", run=run)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE runs SET
                        status = ?,
                        stop_reason = ?,
                        benchmark_outcome = ?,
                        verification_quality = ?,
                        model_calls = ?,
                        steps = ?,
                        prompt_tokens = ?,
                        completion_tokens = ?,
                        total_tokens = ?,
                        start_time = ?,
                        end_time = ?,
                        duration_seconds = ?,
                        artifacts_dir = ?,
                        patch_path = ?
                    WHERE run_id = ?
                    """,
                    (
                        run.status.value,
                        run.stop_reason,
                        (
                            run.benchmark_outcome.value
                            if run.benchmark_outcome
                            else None
                        ),
                        run.verification_quality.value,
                        run.model_calls,
                        run.steps,
                        run.total_prompt_tokens,
                        run.total_completion_tokens,
                        run.total_tokens,
                        run.start_time,
                        run.end_time,
                        run.duration_seconds,
                        str(artifacts.artifacts_dir) if artifacts else None,
                        str(artifacts.patch_path)
                        if artifacts and artifacts.patch_path
                        else None,
                        run.run_id,
                    ),
                )
        except Exception as exc:
            self._handle_error(exc, context="on_run_finished", run=run)

        return artifacts


def load_run_record(db_path: Path, run_id: str) -> dict[str, Any]:
    """Reconstruct a run record and its ordered events from SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        run_row = cur.fetchone()
        if run_row is None:
            raise KeyError(f"Run {run_id} not found in {db_path}")

        cur.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence ASC",
            (run_id,),
        )
        event_rows = cur.fetchall()

        events: list[dict[str, Any]] = []
        for r in event_rows:
            events.append(
                {
                    "event_id": r["event_id"],
                    "run_id": r["run_id"],
                    "sequence": r["sequence"],
                    "step": r["step"],
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "duration_ms": r["duration_ms"],
                    "data": json.loads(r["data_json"]),
                }
            )

        return {
            "run": dict(run_row),
            "events": events,
        }
    finally:
        conn.close()
