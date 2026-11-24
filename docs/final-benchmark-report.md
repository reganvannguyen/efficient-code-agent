# Phase 7 Final Benchmark Report

> **Important:** All results reported below are **`local_targeted`** runs on a 16 GiB Apple Silicon MacBook Air (`Apple M4`). They are **not** official SWE-bench Verified scores. Official SWE-bench Docker evaluation was not run on this 16 GB ARM Mac.

## 1. Frozen Configuration ([`benchmarks/protocol.json`](../benchmarks/protocol.json))

- **Protocol version:** `phase7-v1`
- **Model:** `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`sha256: 1664fccab734674a50763490a8c6931b54806d30e5f956b6`, `Q4_K_M`)
- **Server:** `/opt/homebrew/bin/llama-server -m ... -c 8192 --port 9931` (`tool_format: json_schema`)
- **Limits:** `context_window=8192`, `output_reserve=1024`, `max_tokens=1024`, `max_model_calls=30`, `max_steps=30`, `max_consecutive_invalid_actions=6`, `request_timeout=120s`, `run_timeout=300s`, `oracle_timeout=60s`, `setup_timeout=300s`
- **Workspace isolation:** Fresh `agent/` and `evaluator/` git checkouts per trial (`cleanup_workspaces=True`); `edit_file=AUTO`, `run_tests=AUTO` inside `agent/` only.

## 2. Internal Four-Way Ablation (Live 2-Trial Check, `local__add-tax-01`)

From [`docs/phase7-preliminary-trials.md`](phase7-preliminary-trials.md) (`suite_digest: 038c16a2bd46`, `trials=2`, `seeds=[0, 1]`, `oracle_tier=local_targeted`):

| Condition | Context Strategy | Retrieval Mode | Resolved | 95% Wilson CI | Mean Calls | Mean Prompt Tokens | Mean Edits | Retrieval (ms) | Mean Wall (s) | RSS After (MiB) | Swap (MiB) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `bounded_history` | `lexical_only` | 0/2 (0.0%) | [0.0%, 65.8%] | 13.5 | 16,046 | 0.0 | 170.9 | 45.64 | 4,670.1 | 6,797.9 |
| `retrieval_only` | `bounded_history` | `code_aware` | 1/2 (50.0%) | [9.5%, 90.5%] | 9.0 | 10,750 | 0.0 | 163.0 | 38.46 | 4,656.3 | 7,485.4 |
| `clm_only` | `editable_context` | `lexical_only` | 2/2 (100.0%) | [34.2%, 100.0%] | 11.5 | 20,670 | 2.0 | 63.2 | 113.92 | 4,473.8 | 7,107.9 |
| `combined` | `editable_context` | `code_aware` | 1/2 (50.0%) | [9.5%, 90.5%] | 18.5 | 33,128 | 0.5 | 162.4 | 95.09 | 4,471.8 | 6,608.8 |

**Paired comparisons vs `baseline`:** `retrieval_only`, `clm_only`, and `combined` all improved `local__add-tax-01` (`0/2` -> `1/2` or `2/2`), with `0` regressions on this task.

## 3. External Harness Comparison (`mini-SWE-agent==2.4.6`)

> **Label:** External harness comparison (not a controlled retrieval/context ablation). Unmatched parameters: `python_runtime>=3.10_via_uvx`, `system_prompt_and_tool_schema`, `bash_execution_privileges`, `internal_context_pruning_rules`.

- **Shared fixture (`local__add-tax-01`, `local_targeted`):**
  - Internal (`combined`): **1/1 resolved** in `27.61s`.
  - External (`mini-swe-agent==2.4.6` via `uvx --python 3.11`): **0/1 (`timeout` after `120.01s`)**.
- **Third-party 3-way Exercism Python comparison (10 tasks × 3 trials = 90 runs):** See [`docs/exercism-3way-comparison.md`](exercism-3way-comparison.md) — `aider==0.86.2` (`12/30`, `40.0%`, `53.0s` avg), `internal` (`7/30`, `23.3%`, `190.0s` avg), and `mini-swe-agent==2.4.6` (`7/30`, `23.3%`, `153.1s` avg, `14` timeouts).

## 4. Exclusions, Setup Failures, and Negative Results

1. **External 5-task smoke suite ([`docs/baseline-report.md`](baseline-report.md)):**
   - Phase 4 baseline: `0/5` (`4` `invalid_action_limit` on `pallets__flask-5014`, `pylint-dev__pylint-4551`, `pytest-dev__pytest-10051`, `pytest-dev__pytest-10081`; `1` `setup_error` on `sympy__sympy-12096` due to `git clone --no-checkout` exceeding `120s`).
   - Phase 5 re-run in this environment: `5/5` `setup_error` due to `120s` GitHub clone timeouts (addressed in `benchmarks/protocol.json` by raising `setup_timeout` to `300s`).
2. **Phase 6 multi-file fixture ([`docs/phase6-comparison-report.md`](phase6-comparison-report.md)):** `code_aware` cut prompt tokens by ~70% (`6,444` vs `21,354`) and calls in half (`6` vs `12`), but both modes scored `0/1` (`unresolved`) because the 7B model still produced an incorrect patch.
3. **KV-cache RAM observation:** `llama-server` RSS stayed between `4.47–4.67 GiB` across all 4 conditions regardless of prompt token count, confirming that shorter prompts do not automatically reduce allocated KV-cache RAM for a fixed `8192` window.

## 5. Conclusions and Threats to Validity

- **Editable context (`clm_only`):** Improved resolution on `local__add-tax-01` (`2/2` vs `0/2`) when the model actively used `update_context` (`2.0` edits/run), at the cost of higher prompt tokens (`20,670` vs `16,046`) and wall time (`113.9s` vs `45.6s`).
- **Code-aware retrieval (`retrieval_only`):** Reduced calls (`9.0` vs `13.5`) and prompt tokens (`10,750` vs `16,046`) with minimal parse overhead (`~163 ms`), and cut tokens by ~70% on `local__pricing-01`, though it cannot compensate when the 7B model writes a buggy patch.
- **Combined mode (`combined`):** Inconclusive under small-sample testing (`1/2` vs `2/2` for `clm_only`, `33,128` mean prompt tokens, wide overlapping 95% Wilson CIs `[9.5%, 90.5%]` vs `[34.2%, 100.0%]`).
- **Threats to validity:** Small live sample size (`1` task × `2` trials for 4-way, `5` external smoke tasks blocked by `120s` clone timeouts), `local_targeted` oracle tier only, and 7B quantized model JSON/patch reliability limits.
