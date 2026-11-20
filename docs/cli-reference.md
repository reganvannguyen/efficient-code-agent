# CLI Reference

All CLI commands share the same localhost server configuration precedence:

**CLI flags (`--base-url`, `--model`) > Environment variables (`HARNESS_BASE_URL`, `HARNESS_MODEL`) > `~/.harness/config.json` > Default (`http://127.0.0.1:8080`, auto-detected model).**

Only localhost URLs (`127.0.0.1`, `localhost`, `::1`) are accepted.

## Commands

### `harness ui` — Local Web UI
Launch the localhost-only web workspace (`http://127.0.0.1:8765` by default) and open it in your browser:

```sh
harness ui
harness ui --port 8765 --repo /path/to/repo --no-browser
```

Options:
- `--host` (default `127.0.0.1`, localhost-only)
- `--port` (default `8765`)
- `--repo` (optional initial repository path; can also be picked in the browser)
- `--base-url`, `--model` (override saved `~/.harness/config.json`)
- `--db-path` (default `~/.harness/ui/traces.db`)
- `--memory-dir` (default `~/.harness/ui/runs`)
- `--no-browser` (do not auto-open browser tab)

### `harness session` — Interactive Multi-Turn Repository Session
Start a multi-turn CLI session with `editable_context` (`working.md`), tools, and optional coding edits:

```sh
harness session --repo . --retrieval-mode code_aware
harness session --repo . --coding --test-target tests/
```

### `harness run` — Single-Task Repository Agent
Run a bounded autonomous loop on a repository task. **Read-only by default** (`list_files`, `search_code`, `read_file`).

```sh
# Read-only inspection
harness run "Explain how greeting.py combines its prefix and name" --repo ./tests/fixtures/read_only_repo

# Coding mode with interactive ASK approval (default)
harness run "Fix and test the parser" --repo ./example-repo --coding

# Coding mode with explicit AUTO approval and pinned pytest targets
harness run "Fix and test the parser" --repo ./example-repo --coding \
  --auto-approve --test-target tests/test_parser.py --test-timeout 60
```

Key options:
- `--repo PATH` (defaults to current working directory)
- `--context-strategy {bounded_history,editable_context}`
- `--retrieval-mode {lexical_only,code_aware}`
- `--coding` — opt into `edit_file` and `run_tests` (default policy: `ASK` on interactive stdin/stderr)
- `--auto-approve` — explicitly set edit and test policies to `AUTO`
- `--test-target PATH` (repeatable) — allowlisted workspace-relative pytest targets (default `.`)
- `--test-timeout SECONDS` — timeout per pytest invocation
- `--context-window TOKENS`, `--max-tool-output-bytes BYTES`
- `--max-model-calls N`, `--max-steps N`, `--timeout SECONDS`, `--run-timeout SECONDS`
- `--db-path PATH` / `--no-db` — SQLite execution trace persistence
- `--session ID`, `--memory-dir PATH` — reusable working memory scope

### `harness chat` / `harness` — Direct Local Chat
Send a one-shot prompt or open a stateless interactive prompt loop (`/exit` to quit):

```sh
harness chat "Say hello in one sentence."
harness   # interactive prompt loop (no conversation history)
```

## Tool & Output Conventions

- **`list_files`:** Bounded ripgrep file listing (up to 200 paths, 64 KiB, 5s deadline). `pattern` uses Python `Path.match` glob syntax (`*.py`, `settings.py`), not regex alternation.
- **`search_code`:** Regular-expression `query` (up to 4096 UTF-8 bytes), optional `glob` (`Path.match`), and `max_results` (`1–200`). In `code_aware` mode, Tree-sitter ranks Python symbol definitions above comments and extracts bounded snippets.
- **`read_file`:** Bounded UTF-8 line-range reads (up to 200 lines per call, 16 KiB output, 1 MiB file cap), returning the file's full SHA-256 digest.
- **`edit_file`:** Workspace-confined exact-text replacement with SHA-256 precondition checks and protected-path guards.
- **`run_tests`:** Runs only the current Python executable with `-m pytest -q -- TARGET` against configured allowlisted targets (an approval boundary, not an OS sandbox).
- **Stdout/Stderr & Exit Codes:** `harness run` writes only the final answer to stdout and status/metrics to stderr. Exit codes: `0` (completed), `1` (failure or limit reached), `2` (invalid invocation), `130` (Ctrl+C cancelled).

## Live Smoke Check

With a local model server running:

```sh
HARNESS_LIVE_SMOKE=1 .venv/bin/python -m pytest tests/test_cli.py -k live_run_smoke
```
