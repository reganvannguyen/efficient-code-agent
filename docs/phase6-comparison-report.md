# Phase 6 Preliminary Comparison Report: Lexical-Only vs Code-Aware Retrieval

> **Scope:** Issue #38 validation of `code_aware` retrieval against `lexical_only` (Phase 3–5 baseline) with `bounded_history` held constant and CLM context editing disabled.

## Experimental controls

- **Held constant:** model (`Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`), `context_strategy="bounded_history"` (no `update_context` / `read_memory`), `TokenBudget(context_window=8192)`, tool policy, execution limits, and isolated clean-checkout oracle grading.
- **Varied:** `retrieval_backend` (`lexical_only` vs `code_aware`).

## Deterministic fixture verification (`tests/test_retrieval_e2e.py`)

| Check | Result | Notes |
| --- | --- | --- |
| Multi-file symbol discovery & compact snippets | Passed | `code_aware` ranked `compute_discount` in `pricing.py` ahead of comments in `notes.py`. |
| Fallback on unparseable Python | Passed | `broken.py` with syntax errors still returned valid `lexical_match` hits via `search_code`. |
| Cache invalidation on `edit_file` & deletion | Passed | `edit_file` invalidated the cached entry; subsequent `search_code` returned updated code, and deleted files disappeared. |
| Head-to-head fixture comparison (`compare_retrieval_modes`) | `1/1` vs `1/1` (`local_targeted`) | `code_aware` recorded `retrieval_parse_count >= 1` and `retrieval_duration_ms` alongside task outcome. |

## Live local fixture comparison (`local__pricing-01`, `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`, `8192` ctx)

- `lexical_only`: `0/1` (`unresolved`, `status=limit_reached`, `12` calls, `12` tool calls, `14.2 ms` retrieval, `0` parses, `21,354` prompt tokens, `697` completion tokens, `56.02s`).
- `code_aware`: `0/1` (`unresolved`, `status=completed`, `6` calls, `5` tool calls, `30.9 ms` retrieval, `1` parse, `6,444` prompt tokens, `206` completion tokens, `16.58s`).
- **Observed trade-off:** `code_aware` cut prompt tokens by ~70% (`6,444` vs `21,354`), model calls in half (`6` vs `12`), and wall time from `56.02s` to `16.58s` at a retrieval-time cost of `+16.7 ms` (`30.9 ms` vs `14.2 ms`), but neither mode resolved the task on this 1-task trial (`0/1` vs `0/1` — recorded transparently as a non-improvement in resolution rate).

## Reproducing live comparison

With `llama-server` running on port `9931`, run:

```sh
.venv/bin/python scripts/compare_phase6.py --suite-dir benchmarks/smoke
```

This writes per-mode artifacts and `retrieval_comparison.md` under `.runs/phase6_artifacts/`.
