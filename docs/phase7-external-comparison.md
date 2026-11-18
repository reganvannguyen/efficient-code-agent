# Phase 7 External Comparison Protocol: `mini-SWE-agent`

> **Scope:** Issue #41 — external baseline adapter and documented differences. This is an **external harness comparison**, not a controlled retrieval/context ablation.

## Pinned external runner

- **Package:** `mini-swe-agent==2.4.6` (PyPI requires `Python >= 3.10`; invoked in isolation via `uvx --python 3.11 mini-swe-agent==2.4.6` so the main harness stays on Python 3.9.6).
- **Local model connection:** `OPENAI_API_BASE=http://127.0.0.1:9931/v1`, `OPENAI_API_KEY=local-llama`, `--model openai/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` against the same `llama-server -c 8192` instance.
- **Shared oracle:** Both harnesses run against identical clean `agent/` checkouts and are graded in separate `evaluator/` checkouts by the same `_run_oracle_tests` oracle.

## Explicit unmatched conditions

| Dimension | This harness | `mini-swe-agent==2.4.6` |
| --- | --- | --- |
| Python runtime | `3.9.6` (`.venv`) | `>=3.10` (isolated via `uvx`) |
| Tool set | Bounded JSON-schema tools (`list_files`, `search_code`, `read_file`, `edit_file`, `run_tests`, optional `update_context` / `read_memory`) | Unrestricted `bash` tool inside checkout |
| Permissions | `ToolPolicy` + fixed pytest targets | `--yolo` autonomous shell in checkout |
| Prompt & budgeting | `CODING_SYSTEM_PROMPT` + `TokenBudget` + `max_model_calls`/`max_steps`/`max_consecutive_invalid_actions` | mini-SWE-agent default prompt + step/cost limits |

## Reproducing

With `llama-server` running on port `9931`:

```sh
.venv/bin/python scripts/compare_mini_swe_agent.py --suite-dir benchmarks/smoke
```

All setup errors, missing binaries, and timeouts are recorded as `setup_error` / `timeout` / `agent_error` rather than silently excluded.
