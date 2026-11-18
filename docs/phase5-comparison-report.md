# Phase 5 Preliminary Comparison Report: Baseline vs Editable Context

> **Scope:** Issue #33 validation of `editable_context` against `bounded_history` (Phase 4 baseline) under identical model, tool policy, token budget, and oracle settings.

## Experimental controls

- **Held constant:** model (`Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`), context window (`8192` tokens via `TokenBudget`), tool policy (basic lexical discovery + `edit_file` + `run_tests`, no Tree-sitter retrieval), execution limits, and isolated clean-checkout oracle grading.
- **Varied:** `context_strategy` (`bounded_history` vs `editable_context`).
- **Explicit capability difference:** `editable_context` enables `update_context` (normalized to `context_edit`) and `read_memory`, and allocates an isolated `memory_dir` per task trial that is wiped on workspace cleanup.

## Deterministic fixture verification (`tests/test_editable_context_e2e.py`)

| Check | Result | Notes |
| --- | --- | --- |
| Selective note edits & hypothesis replacement | Passed | Model edited `working.md` selectively (3 of 7 steps), replaced a disproven hypothesis, and reused the confirmed finding in later prompts. |
| Immutable boundary enforcement | Passed | Attempting `replace` on `task.md` was rejected with `invalid_target`; `Original task (immutable)` remained intact. |
| Benchmark reset & note isolation | Passed | Independent `evaluate_task()` / `compare_strategies()` runs received fresh per-trial `memory_dir` directories with zero cross-run leakage. |
| Head-to-head fixture comparison (`compare_strategies`) | `1/1` vs `1/1` (`local_targeted`) | `editable_context` incurred +1 model call and +1 step for its `update_context` turn (`context_edit_count=1`, `context_pruned_count=0`). |

## Observed overhead, negative results, and live smoke status

1. **Live local fixture A/B run (`local__add-tax-01`, `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`, `8192` ctx):**
   - `bounded_history`: `0/1` (`timeout`, `stop_reason=no_progress`, `8` calls, `8` tool calls, `8,226` prompt tokens, `229` completion tokens, `20.32s`).
   - `editable_context`: `1/1` (`resolved`, `local_targeted`, `9` calls, `7` tool calls, `1` `context_edit`, `0` forced prunes, `14,405` prompt tokens, `520` completion tokens, `40.34s`).
   - On this 1-task live trial, `editable_context` resolved the bug while `bounded_history` looped into `no_progress`, at a cost of `+1` model call, `+6,179` prompt tokens, and `+20.02s` wall time.
2. **Known overhead:** Every model-initiated `update_context` call consumes one model call and one step from the task budget, and `editable_context` instructions + notes increase prompt token footprint. On short tasks where baseline already succeeds, `editable_context` adds call/token overhead without improving resolution rate.
3. **External smoke suite (`benchmarks/smoke`):** In both Phase 4 (`docs/baseline-report.md`) and Phase 5 live runs in this environment, external `git clone --no-checkout https://github.com/...` hit `setup_timeout=120.0s` (`setup_error` on `5/5`). Reproducing external comparisons requires local bare mirrors or a higher `setup_timeout`:

```sh
.venv/bin/python scripts/compare_phase5.py --suite-dir benchmarks/smoke
```

This writes per-strategy artifacts and `strategy_comparison.md` under `.runs/phase5_artifacts/`.
