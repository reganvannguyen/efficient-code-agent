"""Tests for the local-only web UI server and interactive approval flow (Issue #77)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from harness.llama_client import ChatResponse
from harness.ui import UISessionManager, create_ui_server


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_ui_server_binds_localhost_only_and_validates_inputs(tmp_path: Path) -> None:
    manager = UISessionManager(
        default_repo=tmp_path,
        base_url="http://127.0.0.1:19999",
        db_path=tmp_path / "traces.db",
        memory_dir=tmp_path / "mem",
    )
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_ui_server(manager, host="0.0.0.0", port=0)

    server = create_ui_server(manager, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/", timeout=5.0) as resp:
            html = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "Efficient Code Agent" in html
        assert "https://" not in html  # zero external CDN dependencies

        status, health = _http_json(f"{base}/api/health")
        assert status == 200
        assert health["llama_server_ok"] is False
        assert "llama-server" in str(health["start_command"])

        bad_status, bad_body = _http_json(
            f"{base}/api/runs",
            method="POST",
            payload={"task": "   ", "repo": str(tmp_path)},
        )
        assert bad_status == 400
        assert "task" in str(bad_body["error"])

        # Native Finder picker endpoint works when mocked
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("harness.ui.server.sys.platform", "darwin")
            mp.setattr(
                "harness.ui.server.subprocess.run",
                lambda *_a, **_kw: type(
                    "Proc", (), {"returncode": 0, "stdout": f"{tmp_path}/\n"}
                )(),
            )
            pick_status, pick_body = _http_json(
                f"{base}/api/pick-repo", method="POST", payload={}
            )
            assert pick_status == 200
            assert pick_body == {"cancelled": False, "path": str(tmp_path)}
    finally:
        server.shutdown()
        server.server_close()


def test_ui_server_interactive_approval_and_completion(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "app.py"
    original = "def val():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import val\n\ndef test_val():\n    assert val() == 2\n",
        encoding="utf-8",
    )
    digest = _sha256(original)

    actions = iter(
        [
            {
                "kind": "context_edit",
                "operation": "replace",
                "target": "working.md",
                "content": "## Confirmed Findings\nval() returns 1 instead of 2.",
            },
            {
                "kind": "tool",
                "name": "edit_file",
                "arguments": {
                    "path": "app.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                    "expected_sha256": digest,
                },
            },
            {
                "kind": "tool",
                "name": "run_tests",
                "arguments": {"preset": "pytest", "target": "."},
            },
            {"kind": "complete", "summary": "Updated val() to return 2."},
        ]
    )

    manager = UISessionManager(
        default_repo=repo,
        db_path=tmp_path / "traces.db",
        memory_dir=tmp_path / "mem",
        model_callable=lambda _p, *, timeout: ChatResponse(
            json.dumps(next(actions)), prompt_tokens=50, completion_tokens=10
        ),
    )
    server = create_ui_server(manager, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        status, created = _http_json(
            f"{base}/api/runs",
            method="POST",
            payload={
                "repo": str(repo),
                "task": "Change val() to return 2",
                "coding": True,
                "auto_approve": False,
                "context_strategy": "editable_context",
                "retrieval_mode": "code_aware",
            },
        )
        assert status == 201
        run_id = str(created["run_id"])

        # Wait for interactive approval pause
        deadline = time.monotonic() + 5.0
        run_state: dict[str, object] = {}
        # Approve both edit_file and run_tests interactively
        for expected_tool in ("edit_file", "run_tests"):
            while time.monotonic() < deadline:
                _, run_state = _http_json(f"{base}/api/runs/{run_id}")
                if run_state.get("pending_approval") is not None:
                    break
                time.sleep(0.05)
            assert run_state.get("status") == "waiting_approval"
            assert run_state["pending_approval"]["tool"] == expected_tool  # type: ignore[index]

            app_status, _ = _http_json(
                f"{base}/api/runs/{run_id}/approve",
                method="POST",
                payload={"approved": True},
            )
            assert app_status == 200
            time.sleep(0.05)

        # Wait for completion
        while time.monotonic() < deadline:
            _, run_state = _http_json(f"{base}/api/runs/{run_id}")
            if run_state.get("status") == "completed":
                break
            time.sleep(0.05)

        assert run_state["status"] == "completed"
        assert "return 2" in target.read_text(encoding="utf-8")
        assert "val() returns 1 instead of 2." in str(run_state["working_md"])

        hist_status, hist = _http_json(f"{base}/api/history")
        assert hist_status == 200
        assert len(hist["runs"]) == 1  # type: ignore[arg-type]
    finally:
        server.shutdown()
        server.server_close()


def test_ui_server_passes_session_chat_history_on_followup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    seen_prompts: list[str] = []

    def _model(prompt: str, *, timeout: float | None = None) -> ChatResponse:
        del timeout
        seen_prompts.append(prompt)
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": f"reply-{len(seen_prompts)}"}),
            prompt_tokens=20,
            completion_tokens=10,
        )

    manager = UISessionManager(
        default_repo=repo,
        db_path=tmp_path / "traces.db",
        memory_dir=tmp_path / "mem",
        model_callable=_model,
    )
    server = create_ui_server(manager, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for task_text, sid in (
            ("first question", "sess-1"),
            ("follow up question", "sess-1"),
            ("fresh chat question", "sess-2"),
        ):
            status, created = _http_json(
                f"{base}/api/runs",
                method="POST",
                payload={"repo": str(repo), "task": task_text, "session_id": sid},
            )
            assert status == 201
            run_id = str(created["run_id"])
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                _, state = _http_json(f"{base}/api/runs/{run_id}")
                if state.get("status") == "completed":
                    break
                time.sleep(0.05)
            assert state.get("status") == "completed"

        assert len(seen_prompts) == 3
        assert "Previous conversation in this session" not in seen_prompts[0]
        assert "User: first question" in seen_prompts[1]
        assert "Assistant: reply-1" in seen_prompts[1]
        assert "Current user request:\nfollow up question" in seen_prompts[1]
        assert "Previous conversation in this session" not in seen_prompts[2]
    finally:
        server.shutdown()
        server.server_close()


def test_ui_server_no_repo_chat_only_and_persistent_session_resume(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "traces.db"
    memory_dir = tmp_path / "mem"
    (memory_dir / "s-persist" / "findings").mkdir(parents=True, exist_ok=True)
    (memory_dir / "s-persist" / "findings" / "notes.md").write_text(
        "saved finding", encoding="utf-8"
    )
    seen_prompts: list[str] = []

    def _model(prompt: str, *, timeout: float | None = None) -> ChatResponse:
        del timeout
        seen_prompts.append(prompt)
        return ChatResponse(
            json.dumps({"kind": "complete", "summary": f"chat-{len(seen_prompts)}"}),
            prompt_tokens=20,
            completion_tokens=10,
        )

    manager1 = UISessionManager(
        default_repo=None,
        db_path=db_path,
        memory_dir=memory_dir,
        model_callable=_model,
    )
    server1 = create_ui_server(manager1, host="127.0.0.1", port=0)
    t1 = threading.Thread(target=server1.serve_forever, daemon=True)
    t1.start()
    try:
        base1 = f"http://127.0.0.1:{server1.server_port}"
        _, health = _http_json(f"{base1}/api/health")
        assert health["default_repo"] == ""

        status, created = _http_json(
            f"{base1}/api/runs",
            method="POST",
            payload={
                "repo": "",
                "task": "explain quicksort",
                "session_id": "s-persist",
            },
        )
        assert status == 201
        run_id = str(created["run_id"])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, state = _http_json(f"{base1}/api/runs/{run_id}")
            if state.get("status") == "completed":
                break
            time.sleep(0.05)
        assert state.get("status") == "completed"
        assert state.get("context_tokens") == 20
        assert state.get("context_window") == 8192
        assert (
            "repository file, search, edit, and test tools are disabled"
            in seen_prompts[0]
        )
    finally:
        server1.shutdown()
        server1.server_close()

    # Simulate server restart with a brand new UISessionManager on the same DB
    manager2 = UISessionManager(
        default_repo=None,
        db_path=db_path,
        memory_dir=memory_dir,
        model_callable=_model,
    )
    server2 = create_ui_server(manager2, host="127.0.0.1", port=0)
    t2 = threading.Thread(target=server2.serve_forever, daemon=True)
    t2.start()
    try:
        base2 = f"http://127.0.0.1:{server2.server_port}"
        s_status, sessions_body = _http_json(f"{base2}/api/sessions")
        assert s_status == 200
        assert len(sessions_body["sessions"]) == 1  # type: ignore[arg-type]
        assert sessions_body["sessions"][0]["session_id"] == "s-persist"  # type: ignore[index]

        d_status, detail = _http_json(f"{base2}/api/sessions/s-persist")
        assert d_status == 200
        assert len(detail["turns"]) == 1  # type: ignore[arg-type]
        assert detail["turns"][0]["user_prompt"] == "explain quicksort"  # type: ignore[index]
        assert any(
            f["path"] == "findings/notes.md" and f["content"] == "saved finding"
            for f in detail["memory_files"]  # type: ignore[attr-defined]
        )

        status2, created2 = _http_json(
            f"{base2}/api/runs",
            method="POST",
            payload={"repo": "", "task": "give an example", "session_id": "s-persist"},
        )
        assert status2 == 201
        run_id2 = str(created2["run_id"])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, state2 = _http_json(f"{base2}/api/runs/{run_id2}")
            if state2.get("status") == "completed":
                break
            time.sleep(0.05)
        assert state2.get("status") == "completed"
        assert "User: explain quicksort" in seen_prompts[1]
        assert "Assistant: chat-1" in seen_prompts[1]

        del_status, del_body = _http_json(
            f"{base2}/api/sessions/s-persist/delete", method="POST", payload={}
        )
        assert del_status == 200
        assert del_body == {"ok": True, "deleted": "s-persist"}
        assert not (memory_dir / "s-persist").exists()
        _, after_sessions = _http_json(f"{base2}/api/sessions")
        assert after_sessions["sessions"] == []
        missing_status, _ = _http_json(f"{base2}/api/sessions/s-persist")
        assert missing_status == 404
    finally:
        server2.shutdown()
        server2.server_close()
