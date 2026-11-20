# Exercism Python 3-Way Harness Comparison

> **Scope:** Independent third-party benchmark comparing our flagship harness (`editable_context` + `code_aware` + graduated loop intervention) against **`mini-SWE-agent==2.4.6`** and **`aider-chat==0.86.2`** on **10 curated Exercism Python exercises × 3 trials = 90 total runs** using the same local `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` model on Apple Silicon (`16 GiB`, `8192` context window).

## 1. Benchmark & Shared Protocol

- **Third-party source:** [`exercism/python`](https://github.com/exercism/python) pinned at commit `479f11b44b02337b0413e805b948491c2179615a` (built via [`scripts/build_exercism_suite.py`](../scripts/build_exercism_suite.py), manifests pinned in [`benchmarks/exercism/`](../benchmarks/exercism/)).
- **Leakage prevention & oracle verification:** Every task repo excludes `.meta/` (so no harness can access `.meta/example.py`), includes a `.gitignore` (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `.aider*`) so temporary test artifacts never pollute candidate patches, and passed a 10/10 sanity check (stub fails `pytest`, reference solution passes `pytest`).
- **Model & server:** `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`Q4_K_M`, `context_window=8192`) served via local `llama-server` at `http://127.0.0.1:9931`.
- **Shared grading & limits:** Isolated clean git checkouts per trial (`cleanup_workspaces=True`), `run_timeout=240s`, `oracle_timeout=60s`, graded in a separate evaluator checkout against each exercise's full official `*_test.py` suite (`oracle_tier=local_full`).
- **Runner:** [`scripts/compare_mini_swe_agent.py`](../scripts/compare_mini_swe_agent.py) (`--trials 3 --run-timeout 240 --resume` under `caffeinate -dims`).

## 2. Overall Results (30 Runs per Harness)

| Harness | Resolved | Pass Rate | 95% Wilson CI | Mean Wall (s) | Outcomes |
| --- | ---: | ---: | --- | ---: | --- |
| **`aider`** (`0.86.2`) | **12 / 30** | **40.0%** | [24.6%, 57.7%] | **53.01s** | `resolved=12, unresolved=15, timeout=3` |
| **`internal`** (`combined`) | **7 / 30** | **23.3%** | [11.8%, 40.9%] | 190.01s | `resolved=7, unresolved=15, timeout=6, agent_error=2` |
| **`mini-swe-agent`** (`2.4.6`) | **7 / 30** | **23.3%** | [11.8%, 40.9%] | 153.13s | `resolved=7, unresolved=9, timeout=14` |

### Pairwise Per-Task Comparison (10 Tasks)

- **`internal` vs `mini-swe-agent`:** **3 wins**, **3 losses**, **4 ties** (exact tie on overall pass rate: `7/30` vs `7/30`).
- **`internal` vs `aider`:** **2 wins** (`clock`, `raindrops`), **4 losses** (`anagram`, `rotational-cipher`, `tournament`, `two-fer`), **4 ties** (`grade-school`, `hamming`, `kindergarten-garden`, `word-count`).

## 3. Per-Task Resolution Matrix (3 Trials Each)

| Task | Difficulty | `internal` (resolved · wall · avg calls · avg tokens) | `mini-swe-agent` | `aider` |
| --- | --- | ---: | ---: | ---: |
| `exercism__two-fer` | easy | 1/3 (236.5s · 22.7 calls · 77,276 tok) | 2/3 (118.8s) | **3/3 (14.6s)** |
| `exercism__raindrops` | easy | 1/3 (234.2s · 15.0 calls · 68,206 tok) | **3/3 (40.9s)** | 0/3 (22.5s) |
| `exercism__hamming` | easy | 0/3 (240.0s · 20.3 calls · 79,639 tok) | 0/3 (174.4s) | 0/3 (15.1s) |
| `exercism__word-count` | medium | 0/3 (212.6s · 23.7 calls · 78,655 tok) | 0/3 (170.8s) | 0/3 (240.0s) |
| `exercism__anagram` | medium | 2/3 (135.6s · 12.3 calls · 41,138 tok) | 0/3 (240.0s) | **3/3 (97.7s)** |
| `exercism__grade-school` | medium | 0/3 (240.0s · 14.3 calls · 47,011 tok) | 0/3 (73.0s) | 0/3 (9.2s) |
| `exercism__rotational-cipher` | medium | 2/3 (124.3s · 10.3 calls · 29,451 tok) | 1/3 (171.3s) | **3/3 (19.7s)** |
| `exercism__kindergarten-garden` | hard | 0/3 (183.9s · 14.0 calls · 58,495 tok) | **1/3 (113.4s)** | 0/3 (19.7s) |
| `exercism__clock` | hard | **1/3 (123.3s · 6.3 calls · 38,033 tok)** | 0/3 (238.7s) | 0/3 (42.7s) |
| `exercism__tournament` | hard | 0/3 (169.8s · 9.3 calls · 38,266 tok) | 0/3 (190.0s) | **3/3 (48.9s)** |

### Token & Call Efficiency (`internal`)

> **Measurement note:** Exact token and call counts are recorded per run for `internal` (`29/30` runs reported usage; 1 `agent_error` exited before returning usage). `mini-swe-agent` and `aider` run as black-box external subprocesses via `evaluate_external_task()`, so their internal token totals were not captured (`N/A`); wall-clock latency serves as the cross-harness efficiency proxy.

| `internal` Slice | Runs | Mean Model Calls | Mean Steps | Mean Total Tokens | Mean Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **All runs** | 30 (29 with tokens) | 14.83 | 14.13 | 56,223 | 190.01s |
| **Resolved runs** | 7 | 11.86 | 11.86 | **40,660** | 122.4s |
| **Unresolved / timed-out** | 23 (22 with tokens) | 15.74 | 14.83 | 61,175 | 210.6s |

## 4. Analysis & Key Takeaways

### Difficulty & Distinct-Task Breakdown

| Slice | `internal` | `mini-swe-agent` | `aider` |
| --- | ---: | ---: | ---: |
| **Easy** (`two-fer`, `raindrops`, `hamming`, 9 runs) | 2/9 (22.2%) | **5/9 (55.6%)** | 3/9 (33.3%) |
| **Medium + Hard** (7 tasks, 21 runs) | **5/21 (23.8%)** | 2/21 (9.5%) | **9/21 (42.9%)** |
| **Distinct tasks solved (≥1/3)** | **5 / 10** | 4 / 10 | 4 / 10 |

1. **`internal` matches `mini-swe-agent` overall (`23.3%` vs `23.3%`) with a complementary profile and far fewer timeouts (`6` vs `14`):**
   - **Different tasks solved:** `internal` solved **5 distinct tasks** (`anagram`, `clock`, `raindrops`, `rotational-cipher`, `two-fer`) vs. **4** for `mini-swe-agent` (`kindergarten-garden`, `raindrops`, `rotational-cipher`, `two-fer`).
   - **Medium/hard logic vs. easy consistency:** `internal` more than doubled `mini-swe-agent` on medium/hard logic tasks (`5/21 = 23.8%` vs `2/21 = 9.5%`, winning `anagram` 2/3 vs 0/3, `clock` 1/3 vs 0/3, and `rotational-cipher` 2/3 vs 1/3), while `mini-swe-agent` was faster and more consistent on easy single-file tasks (`5/9 = 55.6%` vs `2/9 = 22.2%`, especially `raindrops` 3/3 in 40.9s and `two-fer` 2/3 in 118.8s) because it doesn't require a `read_file` SHA-256 precondition or `update_context` steps before writing.
   - **Timeouts:** `mini-swe-agent` timed out on **46.7%** of its runs (`14/30`) vs. `20.0%` (`6/30`) for `internal`, frequently getting stuck in shell-escaping or multi-step bash loops on medium/hard tasks.
2. **Why `aider` won (`40.0%`, `3.6×` faster than `internal`):**
   - **Whole-file edit format + zero discovery overhead:** Aider's standard Exercism invocation passes `{solution_file}` directly into chat context and uses Aider's `whole` edit format (rewriting the file cleanly in one response without exact `old_text` / SHA-256 matching). That let Qwen2.5-Coder-7B solve `two-fer` (14.6s), `rotational-cipher` (19.7s), `tournament` (48.9s), and `anagram` (97.7s) with 3/3 consistency.
   - **Immediate `--auto-test` loop:** Aider runs `pytest` automatically after each edit without spending an extra tool-call turn deciding to invoke `run_tests`.
3. **Actionable bottlenecks revealed in `internal`:**
   - **Latency (`190.0s` avg):** Requiring separate tool turns for `list_files` → `read_file` → `update_context` → `edit_file` (with exact SHA-256 + exact `old_text` match) → `run_tests` burns 6–15+ model calls on small exercises, causing `internal` to hit the 240s timeout even on easy tasks (`two-fer`, `raindrops`, `hamming`).
   - **High-leverage next improvements (implemented in [PR #85](https://github.com/reganvannguyen/efficient-code-agent/pull/85)):**
     1. Auto-fill `expected_sha256` from the last `read_file` ([Issue #82](https://github.com/reganvannguyen/efficient-code-agent/issues/82)) so 7B models don't burn turns on SHA-256 copy errors.
     2. Optionally auto-run configured `pytest` targets after a successful `edit_file` in `AUTO` mode ([Issue #83](https://github.com/reganvannguyen/efficient-code-agent/issues/83)), saving a full model turn per edit-test cycle.

## 5. Unmatched Parameters & Threats to Validity

- **External harness differences:** This is an external harness comparison, not a controlled single-variable ablation. Each harness uses its own system prompt, edit mechanism, and loop structure.
- **Aider task-setup & architecture differences (important context for interpreting 40.0%):**
  - Each Exercism exercise has **1 solution file** (10–80 lines) and **1 test file**, and the prompt names both files for all three harnesses.
  - Following Aider's standard Exercism benchmark CLI pattern, `{solution_file}` was passed directly on Aider's command line—so Aider **pre-loaded the file contents into Turn 1** automatically, skipping the 2–4 autonomous read/discovery turns (~40–60s) required by `internal` and `mini-swe-agent`.
  - Aider also defaulted to its **`whole` edit format** (rewriting the entire file in one response), which avoids `old_text` / SHA-256 errors on tiny 20–80 line Exercism stubs, but does **not** scale to large multi-file repositories under an `8192`-token local context window (where bounded `read_file` + targeted `edit_file` + `code_aware` retrieval are necessary).
  - Because `aider-chat==0.86.2` pins `scipy==1.15.3` whose `_spropack` shared library fails `dlopen` on this macOS version during `RepoMap` PageRank, Aider was run with `--map-tokens 0`.
- **`mini-swe-agent` text-based config:** Because local `llama-server` ran without `--jinja` native tool-calling templates, `mini-swe-agent==2.4.6` was run with `--config mini_textbased.yaml --model-class litellm_textbased` and `MSWEA_CONFIGURED=true`.
- **Sample size:** 30 runs per harness yields overlapping 95% Wilson CIs (`[24.6%, 57.7%]` vs `[11.8%, 40.9%]`), so per-task patterns and latency/timeout profiles are more informative than small percentage differences alone.


## 6. Post-Benchmark Iteration (PR #85)

Follow-up runs under the same `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`8192` context window) setup measured each improvement incrementally:

- **Exercism 30-run `internal` v2 (#82 + #83, prior to line-range edits):** Matched v1 at **7 / 30** (`23.3%`, 95% Wilson CI `[11.8%, 40.9%]`, **187.05s** mean wall, **13.93** mean calls, **56,199** mean tokens across 28/30 runs with reported usage vs. v1's `190.01s` / `14.83` calls / `56,223` across 29/30 runs). Issue #82 eliminated all `invalid_arguments` edit failures (`23 → 0`) and Issue #83 sped up clean solves (**5** completed+resolved runs averaged **7.8** steps, **26,773** tokens, **95.4s** vs. v1's **11.9** steps, **40,660** tokens, **122.4s**), while exact `old_text` `conflict` errors remained unchanged (**38** across `19/30` runs).
- **Line-range `edit_file` + compact `pytest` (`-q --tb=short --maxfail=5`) sanity checks:** Across **4** single-trial checks (`kindergarten-garden`, `two-fer`, `tournament`, `anagram`), edit `conflict`/`invalid_arguments` errors and `run_tests` `output_limit` kills both dropped to **0**:
  - `exercism__two-fer`: **resolved** in **30.0s** / `6` calls / `12,602` tokens (up from `0/3` in v2).
  - `exercism__anagram` *(regression check)*: **resolved** in **45.0s** / `7` calls / `17,357` tokens.
  - `exercism__kindergarten-garden` and `exercism__tournament`: remained `unresolved` due to model logic/budget misses rather than tool failures.
- **Multi-file lookup & working-memory smoke test ([`benchmarks/memory_lookup/`](../benchmarks/memory_lookup/), 5 tasks × 1 trial, `420s` timeout):** Added root `list_files` `{}` exploration guidance and dynamic `read_memory` schema gating (`runtime.py`), comparing `internal` (`editable_context`), `internal_no_notes` (`bounded_history`), and `mini-swe-agent==2.4.6`. `read_memory` calls dropped **23 → 0**, target file localization/patching rose to **5/5** (`internal`) and **4/5** (`internal_no_notes`) vs. **1/5** (`mini-swe-agent`, 4 `agent_error`s), and `internal` used **430,665** total tokens vs. **468,223** for `internal_no_notes`. All three arms remained **0/5** resolved because Qwen 7B still jumped to editing after reading 1–3 files instead of inspecting all 4 rule modules first.
