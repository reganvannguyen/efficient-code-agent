# Phase 7 Preliminary Multi-Trial Findings (Issue #42)

> **Scope:** Live 2-trial 4-way ablation check on `local__add-tax-01` (`local_targeted` tier) with `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` at `8192` context (`benchmarks/protocol.json`). Not a held-out 20-task benchmark score.

## Aggregated results (2 trials per condition, seeds `0, 1`)

- **Suite digest:** `038c16a2bd46` | **Tasks per trial:** `1` (`100.0` pp per task)
- **Hardware note:** `llama-server` RSS stayed around `4.47–4.67 GiB` regardless of prompt token count, confirming that fewer prompt tokens do not automatically reduce KV-cache allocation for a fixed context window.

| Condition | Resolved (Total) | Resolved (Eligible) | 95% Wilson CI | Mean Calls | Mean Prompt Tokens | Mean Context Edits | Retrieval (ms) | Mean Wall (s) | RSS After (MiB) | Swap (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0/2 (0.0%) | 0/2 (0.0%) | [0.0%, 65.8%] | 13.5 | 16,046 | 0.0 | 170.9 | 45.64 | 4,670.1 | 6,797.9 |
| `retrieval_only` | 1/2 (50.0%) | 1/2 (50.0%) | [9.5%, 90.5%] | 9.0 | 10,750 | 0.0 | 163.0 | 38.46 | 4,656.3 | 7,485.4 |
| `clm_only` | 2/2 (100.0%) | 2/2 (100.0%) | [34.2%, 100.0%] | 11.5 | 20,670 | 2.0 | 63.2 | 113.92 | 4,473.8 | 7,107.9 |
| `combined` | 1/2 (50.0%) | 1/2 (50.0%) | [9.5%, 90.5%] | 18.5 | 33,128 | 0.5 | 162.4 | 95.09 | 4,471.8 | 6,608.8 |

## Paired task comparisons vs `baseline`

- **`retrieval_only`:** improved=`['local__add-tax-01']`, regressed=`[]` (fewer calls and prompt tokens than `baseline`)
- **`clm_only`:** improved=`['local__add-tax-01']`, regressed=`[]` (`2.0` note edits/run; higher token & wall-time overhead)
- **`combined`:** improved=`['local__add-tax-01']`, regressed=`[]` (highest mean calls/tokens on this 1-task sample)

## Reproducing

With `llama-server` running on port `9931`:

```sh
.venv/bin/python scripts/run_repeated_ablations.py --suite-dir benchmarks/smoke --trials 3
```
