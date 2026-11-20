"""Local-only stdlib HTTP server for running and inspecting agent sessions."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from harness.agent.events import RunEvent
from harness.agent.runtime import (
    CHAT_ONLY_SYSTEM_PROMPT,
    CODING_SYSTEM_PROMPT,
    READ_ONLY_SYSTEM_PROMPT,
    Model,
    action_schema,
    run_agent,
)
from harness.agent.state import AgentRun
from harness.config import (
    DEFAULT_BASE_URL,
    DEFAULT_CONFIG_PATH,
    save_user_config,
    validate_local_base_url,
)
from harness.context import (
    LlamaTokenCounter,
    TokenBudget,
    estimate_tokens,
    validate_scope_id,
)
from harness.llama_client import chat, probe_server
from harness.storage.database import SqliteRunRecorder, load_run_record
from harness.tools.policy import ApprovalRequest, ToolMode, ToolPolicy

_INDEX_HTML_PATH = Path(__file__).with_name("index.html")
_START_LLAMA_CMD = "llama-server -m /path/to/model.gguf --port 8080 -c 8192"
_MAX_SESSION_TURNS = 6
_MAX_TURN_REPLY_CHARS = 600
_MAX_CHAT_HISTORY_CHARS = 2400


def _read_session_memory_files(memory_dir: Path, scope: str) -> list[dict[str, str]]:
    """Return working.md, task.md, and findings/*.md files for a session scope."""
    base = memory_dir / scope
    files: list[dict[str, str]] = []
    for rel in ("working.md", "task.md"):
        p = base / rel
        if p.is_file():
            try:
                files.append(
                    {
                        "path": rel,
                        "content": p.read_text(encoding="utf-8", errors="replace"),
                    }
                )
            except OSError:
                pass
    findings_dir = base / "findings"
    if findings_dir.is_dir():
        for p in sorted(findings_dir.glob("*.md")):
            if p.is_file():
                try:
                    files.append(
                        {
                            "path": f"findings/{p.name}",
                            "content": p.read_text(encoding="utf-8", errors="replace"),
                        }
                    )
                except OSError:
                    pass
    return files


@dataclass
class _UIRunState:
    ui_run_id: str
    repo: Path | None
    task: str
    memory_dir: Path
    session_id: str | None = None
    status: str = "running"
    stop_reason: str | None = None
    steps: int = 0
    model_calls: int = 0
    total_tokens: int | None = None
    context_tokens: int | None = None
    context_window: int = 8192
    final_response: str | None = None
    candidate_diff: str = ""
    events: list[dict[str, object]] = field(default_factory=list)
    pending_approval: dict[str, object] | None = None
    cancel_requested: bool = False
    _approval_event: threading.Event = field(default_factory=threading.Event)
    _approval_decision: bool = False
    _agent_run_id: str | None = None

    def working_md_text(self) -> str:
        scope = self.session_id or self._agent_run_id or self.ui_run_id
        target = self.memory_dir / scope / "working.md"
        if target.is_file():
            try:
                return target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        return ""

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.ui_run_id,
            "session_id": self.session_id,
            "agent_run_id": self._agent_run_id,
            "repo": str(self.repo) if self.repo is not None else "",
            "task": self.task,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "model_calls": self.model_calls,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
            "context_window": self.context_window,
            "final_response": self.final_response,
            "candidate_diff": self.candidate_diff,
            "working_md": self.working_md_text(),
            "memory_files": _read_session_memory_files(
                self.memory_dir, self.session_id or self._agent_run_id or self.ui_run_id
            ),
            "pending_approval": self.pending_approval,
            "events": list(self.events),
        }


class UISessionManager:
    """Thread-safe coordinator for browser-started agent runs and approvals."""

    def __init__(
        self,
        *,
        default_repo: Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = "",
        db_path: Path = Path(".runs/traces.db"),
        memory_dir: Path = Path(".harness/runs"),
        config_path: Path = DEFAULT_CONFIG_PATH,
        model_callable: Model | None = None,
    ) -> None:
        self.default_repo = default_repo.resolve() if default_repo is not None else None
        self.base_url = validate_local_base_url(base_url)
        self.model_name = model_name.strip()
        self.db_path = db_path
        self.memory_dir = memory_dir
        self.config_path = config_path
        self.model_callable = model_callable
        self._lock = threading.Lock()
        self._runs: dict[str, _UIRunState] = {}
        self._active_run_id: str | None = None
        self._session_turns: dict[str, list[tuple[str, str]]] = {}
        self._ensure_session_tables()

    def _ensure_session_tables(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    model_calls INTEGER NOT NULL,
                    total_tokens INTEGER,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def check_health(self) -> dict[str, object]:
        """Probe the configured localhost model server via /v1/models or /health."""
        with self._lock:
            base_url = self.base_url
            configured_model = self.model_name
        ok, models = probe_server(base_url, timeout=1.5)
        effective_model = configured_model or (models[0] if models else "")
        return {
            "server_ok": ok,
            "llama_server_ok": ok,
            "base_url": base_url,
            "model": effective_model,
            "configured_model": configured_model,
            "models": models,
            "default_repo": str(self.default_repo)
            if self.default_repo is not None
            else "",
            "start_command": _START_LLAMA_CMD,
        }

    def get_config(self) -> dict[str, object]:
        """Return the active localhost server URL and model configuration."""
        with self._lock:
            return {"base_url": self.base_url, "model": self.model_name}

    def update_config(self, payload: dict[str, Any]) -> dict[str, object]:
        """Validate, apply, and persist updated localhost server URL and model."""
        raw_url = payload.get("base_url")
        raw_model = payload.get("model")
        with self._lock:
            new_url = (
                validate_local_base_url(raw_url)
                if raw_url is not None
                else self.base_url
            )
            if raw_model is not None:
                if not isinstance(raw_model, str):
                    raise ValueError("model must be a string")
                new_model = raw_model.strip()
            else:
                new_model = self.model_name
            self.base_url = new_url
            self.model_name = new_model
            config_path = self.config_path
        save_user_config(base_url=new_url, model=new_model, path=config_path)
        return self.check_health()

    def start_run(self, payload: dict[str, Any]) -> dict[str, object]:
        """Validate payload and start an agent run in a background thread."""
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-blank string")

        raw_repo = payload.get("repo")
        if raw_repo is None:
            repo = self.default_repo
        elif not isinstance(raw_repo, str):
            raise ValueError("repo must be a string path or empty")
        elif not raw_repo.strip():
            repo = None
        else:
            repo = Path(raw_repo).expanduser().resolve()
            if not repo.is_dir():
                raise ValueError(f"repo directory does not exist: {repo}")

        coding = bool(payload.get("coding", False)) and repo is not None
        auto_approve = bool(payload.get("auto_approve", False))
        context_strategy = str(payload.get("context_strategy", "bounded_history"))
        if context_strategy not in {"bounded_history", "editable_context"}:
            raise ValueError(
                "context_strategy must be bounded_history or editable_context"
            )
        retrieval_mode = str(payload.get("retrieval_mode", "lexical_only"))
        if retrieval_mode not in {"lexical_only", "code_aware"}:
            raise ValueError("retrieval_mode must be lexical_only or code_aware")

        raw_ctx = payload.get("context_window", 8192)
        if not isinstance(raw_ctx, int) or isinstance(raw_ctx, bool) or raw_ctx < 512:
            raise ValueError("context_window must be an integer >= 512")

        def _pos_int(key: str, default: int) -> int:
            val = payload.get(key, default)
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise ValueError(f"{key} must be a positive integer")
            return val

        unlimited_steps = bool(payload.get("unlimited_steps", False))
        max_model_calls = 10_000 if unlimited_steps else _pos_int("max_model_calls", 30)
        max_steps = 10_000 if unlimited_steps else _pos_int("max_steps", 30)
        max_invalid = _pos_int("max_consecutive_invalid_actions", 6)
        raw_session = payload.get("session_id")
        session_id = (
            raw_session.strip()
            if isinstance(raw_session, str) and raw_session.strip()
            else None
        )

        with self._lock:
            if self._active_run_id is not None:
                active = self._runs.get(self._active_run_id)
                if active is not None and active.status in {
                    "running",
                    "pending",
                    "waiting_approval",
                }:
                    raise RuntimeError("Another run is already in progress")

            ui_run_id = uuid.uuid4().hex[:12]
            state = _UIRunState(
                ui_run_id=ui_run_id,
                repo=repo,
                task=task.strip(),
                memory_dir=self.memory_dir,
                session_id=session_id,
            )
            self._runs[ui_run_id] = state
            self._active_run_id = ui_run_id

        thread = threading.Thread(
            target=self._execute_run,
            args=(
                state,
                coding,
                auto_approve,
                context_strategy,
                retrieval_mode,
                raw_ctx,
                max_model_calls,
                max_steps,
                max_invalid,
            ),
            daemon=True,
        )
        thread.start()
        return {
            "run_id": ui_run_id,
            "session_id": session_id,
            "status": state.status,
        }

    def _load_session_turns(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            cached = self._session_turns.get(session_id)
            if cached is not None:
                return list(cached)
        if not self.db_path.is_file():
            return []
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                rows = conn.execute(
                    "SELECT user_prompt, assistant_reply FROM ui_turns "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, _MAX_SESSION_TURNS),
                ).fetchall()
        except sqlite3.Error:
            return []
        turns = [(str(r[0]), str(r[1])) for r in reversed(rows)]
        with self._lock:
            self._session_turns[session_id] = list(turns)
        return turns

    def _build_session_task(self, session_id: str | None, task: str) -> str:
        """Prepend bounded prior (user, assistant) turns for follow-up awareness."""
        if not session_id:
            return task
        turns = self._load_session_turns(session_id)
        if not turns:
            return task
        lines: list[str] = []
        total_chars = 0
        for user_text, reply_text in reversed(turns[-_MAX_SESSION_TURNS:]):
            block = f"User: {user_text}\nAssistant: {reply_text}"
            if lines and total_chars + len(block) > _MAX_CHAT_HISTORY_CHARS:
                break
            lines.append(block)
            total_chars += len(block)
        lines.reverse()
        history_block = "\n\n".join(lines)
        return (
            "Current user request (answer THIS request, not prior turns):\n"
            f"{task.strip()}\n\n"
            "Prior conversation (background context only):\n"
            f"{history_block}"
        )

    def _record_session_turn(
        self,
        session_id: str | None,
        state: _UIRunState,
        assistant_reply: str,
    ) -> None:
        """Store a compact (user, assistant) turn in memory and SQLite."""
        if not session_id:
            return
        user_task = state.task.strip()
        full_reply = assistant_reply.strip()
        compact_reply = full_reply
        if len(compact_reply) > _MAX_TURN_REPLY_CHARS:
            compact_reply = compact_reply[: _MAX_TURN_REPLY_CHARS - 3].rstrip() + "..."
        with self._lock:
            turns = self._session_turns.setdefault(session_id, [])
            turns.append((user_task, compact_reply))
            if len(turns) > _MAX_SESSION_TURNS:
                del turns[:-_MAX_SESSION_TURNS]
        now = datetime.now(timezone.utc).isoformat()
        title = user_task[:60] + ("..." if len(user_task) > 60 else "")
        repo_str = str(state.repo) if state.repo is not None else ""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute(
                    """
                    INSERT INTO ui_sessions (session_id, title, repo, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        repo = excluded.repo,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, title, repo_str, now),
                )
                conn.execute(
                    """
                    INSERT INTO ui_turns (
                        session_id, run_id, user_prompt, assistant_reply,
                        steps, model_calls, total_tokens, status, stop_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        state.ui_run_id,
                        user_task,
                        full_reply,
                        state.steps,
                        state.model_calls,
                        state.total_tokens,
                        state.status,
                        state.stop_reason,
                        now,
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def _execute_run(
        self,
        state: _UIRunState,
        coding: bool,
        auto_approve: bool,
        context_strategy: str,
        retrieval_mode: str,
        context_window: int,
        max_model_calls: int,
        max_steps: int,
        max_consecutive_invalid_actions: int,
    ) -> None:
        chat_only = state.repo is None
        policy: ToolPolicy | None = None
        approve_fn = None
        system_prompt = (
            CHAT_ONLY_SYSTEM_PROMPT if chat_only else READ_ONLY_SYSTEM_PROMPT
        )
        if coding and not chat_only:
            mode = ToolMode.AUTO if auto_approve else ToolMode.ASK
            policy = ToolPolicy(edit_file=mode, run_tests=mode, test_targets=(".",))
            system_prompt = CODING_SYSTEM_PROMPT
            if not auto_approve:

                def _approve(req: ApprovalRequest) -> bool:
                    with self._lock:
                        state._approval_event.clear()
                        state._approval_decision = False
                        state.status = "waiting_approval"
                        state.pending_approval = {
                            "tool": req.tool,
                            "path": str(req.path),
                            "target": req.target,
                        }
                    while not state.cancel_requested:
                        if state._approval_event.wait(timeout=0.25):
                            break
                    with self._lock:
                        state.pending_approval = None
                        if state.cancel_requested:
                            return False
                        state.status = "running"
                        return state._approval_decision

                approve_fn = _approve

        if context_strategy == "editable_context" and not coding:
            system_prompt += " You also have update_context and read_memory tools for session memory notes."

        if self.model_callable is not None:
            model = self.model_callable
            token_counter = estimate_tokens
        else:
            with self._lock:
                run_base_url = self.base_url
                run_model_name = self.model_name
            if not run_model_name:
                _, detected_models = probe_server(run_base_url, timeout=1.5)
                run_model_name = (
                    detected_models[0] if detected_models else "local-model"
                )
            model = partial(
                chat,
                base_url=run_base_url,
                model=run_model_name,
                max_tokens=1024,
                response_schema=action_schema(
                    policy,
                    include_context_edit=(context_strategy == "editable_context"),
                    chat_only=chat_only,
                ),
                system_prompt=system_prompt,
            )
            token_counter = LlamaTokenCounter(base_url=run_base_url)

        recorder = SqliteRunRecorder(db_path=self.db_path)
        with self._lock:
            state.context_window = context_window

        def _on_event(event: RunEvent) -> None:
            with self._lock:
                state._agent_run_id = event.run_id
                state.events.append(event.to_dict())
                state.steps = max(state.steps, event.step)
                if event.event_type.value == "model_request":
                    state.model_calls += 1
                elif event.event_type.value == "model_response":
                    pt = event.data.get("prompt_tokens")
                    if isinstance(pt, int):
                        state.context_tokens = pt

        try:
            workspace = state.repo
            if workspace is None:
                workspace = self.memory_dir / "_chat_workspace"
                workspace.mkdir(parents=True, exist_ok=True)
            effective_task = self._build_session_task(state.session_id, state.task)
            run: AgentRun = run_agent(
                effective_task,
                workspace,
                model=model,
                max_model_calls=max_model_calls,
                max_steps=max_steps,
                max_consecutive_invalid_actions=max_consecutive_invalid_actions,
                policy=policy,
                approve=approve_fn,
                cancelled=lambda: state.cancel_requested,
                on_event=_on_event,
                recorder=recorder,
                context_strategy=context_strategy,
                session_id=state.session_id,
                memory_dir=(
                    self.memory_dir
                    if (context_strategy == "editable_context" or state.session_id)
                    else None
                ),
                token_budget=TokenBudget(
                    context_window=context_window,
                    output_reserve=min(1024, context_window // 4),
                ),
                token_counter=token_counter,
                max_tool_output_bytes=max(4096, min(16384, context_window)),
                retrieval_backend=(
                    None
                    if chat_only or retrieval_mode == "lexical_only"
                    else retrieval_mode
                ),
                chat_only=chat_only,
                record_turn_summary=False,
            )
            with self._lock:
                state._agent_run_id = run.run_id
                state.status = run.status.value
                state.stop_reason = run.stop_reason
                state.steps = run.steps
                state.model_calls = run.model_calls
                state.total_tokens = run.total_tokens
                state.final_response = run.final_response
                state.candidate_diff = run.candidate_diff
            self._record_session_turn(
                state.session_id,
                state,
                run.final_response or f"[{run.status.value}: {run.stop_reason or ''}]",
            )
        except Exception as exc:
            with self._lock:
                state.status = "failed"
                state.stop_reason = str(exc)
        finally:
            with self._lock:
                state.pending_approval = None
                if self._active_run_id == state.ui_run_id:
                    self._active_run_id = None

    def get_run(self, ui_run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._runs.get(ui_run_id)
            if state is None:
                raise KeyError(f"Unknown run_id: {ui_run_id}")
            return state.to_dict()

    def approve_run(self, ui_run_id: str, approved: bool) -> dict[str, object]:
        with self._lock:
            state = self._runs.get(ui_run_id)
            if state is None:
                raise KeyError(f"Unknown run_id: {ui_run_id}")
            if state.pending_approval is None:
                raise ValueError("No pending approval for this run")
            state._approval_decision = bool(approved)
            state._approval_event.set()
            return {"ok": True, "approved": bool(approved)}

    def cancel_run(self, ui_run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._runs.get(ui_run_id)
            if state is None:
                raise KeyError(f"Unknown run_id: {ui_run_id}")
            state.cancel_requested = True
            state._approval_decision = False
            state._approval_event.set()
            return {"ok": True, "cancelled": True}

    def pick_repo(self) -> dict[str, object]:
        """Open a native OS folder picker and return the chosen directory path."""
        if sys.platform == "darwin":
            proc = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'POSIX path of (choose folder with prompt "Select repository folder")',
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                return {"cancelled": True, "path": None}
            chosen = proc.stdout.strip().rstrip("/")
            return {"cancelled": False, "path": chosen or "/"}

        if sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$d.Description = 'Select repository folder'; "
                "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
            )
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if ps is not None:
                proc = subprocess.run(
                    [ps, "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                chosen = proc.stdout.strip()
                if proc.returncode != 0 or not chosen:
                    return {"cancelled": True, "path": None}
                return {"cancelled": False, "path": chosen}

        for cmd in (
            [
                "zenity",
                "--file-selection",
                "--directory",
                "--title=Select repository folder",
            ],
            ["kdialog", "--getexistingdirectory", str(Path.home())],
        ):
            if shutil.which(cmd[0]) is not None:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if proc.returncode != 0:
                    return {"cancelled": True, "path": None}
                chosen = proc.stdout.strip().rstrip("/")
                return {"cancelled": False, "path": chosen or "/"}

        try:
            import tkinter as tk
            from tkinter import filedialog

            root_tk = tk.Tk()
            root_tk.withdraw()
            chosen = filedialog.askdirectory(title="Select repository folder")
            root_tk.destroy()
            if not chosen:
                return {"cancelled": True, "path": None}
            return {"cancelled": False, "path": chosen}
        except Exception:
            return {
                "cancelled": False,
                "path": None,
                "unsupported": True,
                "error": "No native folder picker is available on this system.",
            }

    def list_history(self, limit: int = 25) -> list[dict[str, object]]:
        if not self.db_path.is_file():
            return []
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT run_id, task, status, stop_reason, steps, model_calls, "
                    "total_tokens, duration_seconds, start_time "
                    "FROM runs ORDER BY start_time DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def get_history_record(self, run_id: str) -> dict[str, object]:
        if not self.db_path.is_file():
            raise KeyError("Trace database does not exist")
        return load_run_record(self.db_path, run_id)

    def list_sessions(self, limit: int = 50) -> list[dict[str, object]]:
        if not self.db_path.is_file():
            return []
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT s.session_id, s.title, s.repo, s.updated_at, "
                    "COUNT(t.id) AS turn_count "
                    "FROM ui_sessions s "
                    "LEFT JOIN ui_turns t ON t.session_id = s.session_id "
                    "GROUP BY s.session_id "
                    "ORDER BY s.updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def get_session_detail(self, session_id: str) -> dict[str, object]:
        if not self.db_path.is_file():
            raise KeyError(f"Unknown session_id: {session_id}")
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            session_row = conn.execute(
                "SELECT session_id, title, repo, updated_at "
                "FROM ui_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            turn_rows = conn.execute(
                "SELECT run_id, user_prompt, assistant_reply, steps, model_calls, "
                "total_tokens, status, stop_reason, created_at "
                "FROM ui_turns WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        memory_files = _read_session_memory_files(self.memory_dir, session_id)
        working_md = next(
            (f["content"] for f in memory_files if f["path"] == "working.md"), ""
        )
        return {
            **dict(session_row),
            "turns": [dict(r) for r in turn_rows],
            "working_md": working_md,
            "memory_files": memory_files,
        }

    def delete_session(self, session_id: str) -> dict[str, object]:
        safe_id = validate_scope_id(session_id)
        if not self.db_path.is_file():
            raise KeyError(f"Unknown session_id: {session_id}")
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cur = conn.execute(
                "DELETE FROM ui_sessions WHERE session_id = ?", (safe_id,)
            )
            if cur.rowcount == 0:
                raise KeyError(f"Unknown session_id: {session_id}")
            conn.execute("DELETE FROM ui_turns WHERE session_id = ?", (safe_id,))
            conn.commit()
        with self._lock:
            self._session_turns.pop(safe_id, None)
        mem_path = (self.memory_dir / safe_id).resolve()
        if mem_path.parent == self.memory_dir.resolve() and mem_path.is_dir():
            shutil.rmtree(mem_path, ignore_errors=True)
        return {"ok": True, "deleted": safe_id}


def _make_handler(manager: UISessionManager) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(min(length, 262_144)) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                html_bytes = _INDEX_HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, manager.check_health())
                return
            if path == "/api/config":
                self._send_json(HTTPStatus.OK, manager.get_config())
                return
            if path == "/api/history":
                self._send_json(HTTPStatus.OK, {"runs": manager.list_history()})
                return
            if path == "/api/sessions":
                self._send_json(HTTPStatus.OK, {"sessions": manager.list_sessions()})
                return
            if path.startswith("/api/sessions/"):
                session_id = path[len("/api/sessions/") :]
                try:
                    self._send_json(
                        HTTPStatus.OK, manager.get_session_detail(session_id)
                    )
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            if path.startswith("/api/history/"):
                run_id = path[len("/api/history/") :]
                try:
                    self._send_json(HTTPStatus.OK, manager.get_history_record(run_id))
                except (KeyError, ValueError) as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            if path.startswith("/api/runs/"):
                run_id = path[len("/api/runs/") :]
                try:
                    self._send_json(HTTPStatus.OK, manager.get_run(run_id))
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/config":
                try:
                    self._send_json(HTTPStatus.OK, manager.update_config(body))
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/pick-repo":
                try:
                    self._send_json(HTTPStatus.OK, manager.pick_repo())
                except RuntimeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path == "/api/runs":
                try:
                    created = manager.start_run(body)
                    self._send_json(HTTPStatus.CREATED, created)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except RuntimeError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/approve"):
                run_id = path[len("/api/runs/") : -len("/approve")]
                try:
                    res = manager.approve_run(run_id, bool(body.get("approved", False)))
                    self._send_json(HTTPStatus.OK, res)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                run_id = path[len("/api/runs/") : -len("/cancel")]
                try:
                    res = manager.cancel_run(run_id)
                    self._send_json(HTTPStatus.OK, res)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return

            if path.startswith("/api/sessions/") and path.endswith("/delete"):
                session_id = path[len("/api/sessions/") : -len("/delete")]
                try:
                    res = manager.delete_session(session_id)
                    self._send_json(HTTPStatus.OK, res)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    return _Handler


def create_ui_server(
    manager: UISessionManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a localhost-only ThreadingHTTPServer for the agent UI."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("UI server must bind to 127.0.0.1 or localhost only")
    return ThreadingHTTPServer((host, port), _make_handler(manager))


def serve_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    default_repo: Path | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = "",
    db_path: Path = Path(".runs/traces.db"),
    memory_dir: Path = Path(".harness/runs"),
    open_browser: bool = True,
) -> None:
    """Start the local UI server and block until interrupted."""
    manager = UISessionManager(
        default_repo=default_repo,
        base_url=base_url,
        model_name=model_name,
        db_path=db_path,
        memory_dir=memory_dir,
    )
    server = create_ui_server(manager, host=host, port=port)
    url = f"http://{host}:{server.server_port}/"
    print(f"harness ui listening at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
