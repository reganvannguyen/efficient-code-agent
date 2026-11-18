# Phase 7 Evaluation Protocol and Suite Partitioning

> **Scope:** Issue #39 — frozen evaluation settings, leakage-resistant partitioning, and reporting rules before running Phase 7 ablations (#40) and external comparisons (#41).

## 1. Frozen runtime & model settings ([`benchmarks/protocol.json`](../benchmarks/protocol.json))

- **Model:** `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`sha256: 1664fccab734674a50763490a8c6931b54806d30e5f956b6`, `Q4_K_M`).
- **Server:** `/opt/homebrew/bin/llama-server -m ... -c 8192 --port 9931` (`tool_format: json_schema`).
- **Budgets:** `context_window=8192`, `output_reserve=1024`, `max_tokens=1024`, `max_model_calls=30`, `max_steps=30`, `max_consecutive_invalid_actions=6`, `request_timeout=120s`, `run_timeout=300s`, `oracle_timeout=60s`, `setup_timeout=300s`.
- **Workspace & approval policy:** isolated per-trial `agent/` and `evaluator/` clean git checkouts (`cleanup_workspaces=True`); `edit_file=AUTO`, `run_tests=AUTO` inside `agent/` only.

## 2. Suite partitions & 20-task evaluation plan

- `benchmarks/smoke/` (5 tasks): pipeline smoke validation (`pallets__flask-5014`, `pylint-dev__pylint-4551`, `pytest-dev__pytest-10051`, `pytest-dev__pytest-10081`, `sympy__sympy-12096`).
- `benchmarks/dev/`: development comparisons and tuning (Phase 5/6).
- `benchmarks/held_out/`: untouched held-out tasks for Phase 7 final reporting (**never tune on held-out failures**).
- **Proposed Phase 7 size:** 20 tasks total (5 smoke + 5 dev + 10 held-out), with an optional 50-task extension only if Mac ARM runtime and local bare-mirror setup allow. These are custom local subsets, **not** official SWE-bench Verified scores.
- **Mac ARM feasibility note:** Phase 4/5 runs showed full `git clone --no-checkout https://github.com/...` can exceed `setup_timeout=120s` in constrained environments; Phase 7 trials should use local bare mirrors or raise `setup_timeout`, and every `setup_error` or exclusion must be reported.

## 3. Workspace & prompt isolation

- `TaskManifest.agent_prompt()` exposes only `issue_text` (verified by `verify_agent_prompt_isolation()`). `reference_patch`, `oracle_tests`, and `metadata` (`fail_to_pass` / `pass_to_pass`) never enter the agent prompt.
- `verify_suite_partition_isolation()` enforces zero `task_id` overlap across `smoke`, `dev`, and `held_out`.

## 4. Oracle tiers and denominator rules

| Tier | Meaning | Reporting requirement |
| --- | --- | --- |
| `official` | Official SWE-bench evaluation container/harness | Never claim without running official harness |
| `local_full` | Full repository test suite run locally | Report full-suite pass/fail and setup exclusions |
| `local_targeted` | Declared targeted pytest files in clean checkout | Label explicitly as `local_targeted`, never as official SWE-bench |

Always report **both** denominators when any task is excluded or hits `setup_error`:
1. `resolved / total_frozen_tasks` (including setup errors/exclusions), and
2. `resolved / runnable_tasks` (excluding verified setup errors, with every exclusion listed).
