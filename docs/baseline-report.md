# Phase 4 Baseline Smoke Report

> **Scope:** 5-task pipeline smoke test on local Apple Silicon hardware. These are `local_targeted` smoke runs, **not** official SWE-bench Verified scores.

## Pinned configuration

- **Harness commit:** `a7433f37f8c09628530348363b73f06a889b714d` (baseline: bounded history + basic lexical tools; no CLM context editing, no Tree-sitter ranking)
- **Model:** `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`sha256: 1664fccab734674a50763490a8c6931b54806d30e5f956b6`)
- **Server:** `/opt/homebrew/bin/llama-server -m ... -c 8192 --port 9931`
- **Context window:** `8192` tokens (`tool_format: json_schema`)
- **Hardware:** `Apple M4`, 16 GiB unified RAM (`17,179,869,184` bytes), `macOS-27.0.1-arm64-arm-64bit`
- **Execution policy:** `single_model_sequential`, `max_model_calls=16`, `max_steps=20`

## Smoke suite results (5 tasks)

| Task ID | Outcome | Oracle Tier | Agent Status | Wall Time (s) | Calls / Steps | Prompt / Completion Tokens | `llama-server` RSS Before → After (MiB) | Swap Used (MiB) | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | --- |
| `pallets__flask-5014` | `timeout` | `local_targeted` | `limit_reached` | 28.40 | 3 / 3 | 2,788 / 105 | 3,195.6 → 3,266.4 | 4,079.3 | `invalid_action_limit` (`a62db19f`) |
| `pylint-dev__pylint-4551` | `timeout` | `local_targeted` | `limit_reached` | 63.15 | 6 / 6 | 11,824 / 194 | 3,248.9 → 3,233.7 | 4,241.0 | `invalid_action_limit` (`15a17924`) |
| `pytest-dev__pytest-10051` | `timeout` | `local_targeted` | `limit_reached` | 65.00 | 8 / 8 | 22,029 / 202 | 3,212.0 → 3,267.5 | 4,578.5 | `invalid_action_limit` (`65825b62`) |
| `pytest-dev__pytest-10081` | `timeout` | `local_targeted` | `limit_reached` | 59.89 | 5 / 5 | 13,392 / 145 | 3,226.1 → 3,279.9 | 4,795.1 | `invalid_action_limit` (`a35d9d62`) |
| `sympy__sympy-12096` | `setup_error` | `local_targeted` | `-` | - | - | - | - | - | `git clone --no-checkout` exceeded `setup_timeout=120s` |

## Key observations and limitations

1. **Pipeline verification:** The evaluation runner isolated clean checkouts, recorded SQLite traces (`.runs/smoke_traces.sqlite3`) and disk artifacts (`.runs/smoke_artifacts/`), captured host/model/memory metadata, and separated `setup_error` from agent `limit_reached` outcomes.
2. **Primary failure mode (`invalid_action_limit`):** All 4 cloned tasks hit 3 consecutive invalid JSON/tool actions before producing a patch (`0/5` resolved). Improving tool-call schema adherence and action recovery is a clear baseline finding before Phase 5/6 experiments.
3. **Large-repo checkout timeout (`sympy__sympy-12096`):** Full `git clone --no-checkout` of `sympy/sympy` exceeded the 120s setup timeout; future runs should either raise `setup_timeout` or use shallow/local bare mirrors.
4. **Memory & swap behavior:** `llama-server` RSS stayed steady around `3.20–3.28 GiB` across all 4 runs, while host swap used ranged from `4,079` to `4,795 MiB` (background desktop load also contributes to host swap). `memory_pressure -Q` returned `None` in this environment and was recorded honestly as unavailable rather than estimated.
