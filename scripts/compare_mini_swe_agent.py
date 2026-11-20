"""Run a multi-trial 3-way comparison: internal harness vs. mini-SWE-agent vs. Aider."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from harness.agent.events import BenchmarkOutcome
from harness.context import LlamaTokenCounter, TokenBudget
from harness.evaluation import (
    ExperimentConfig,
    FrozenEvaluationProtocol,
    ModelSpec,
    compute_file_sha256,
    default_aider_spec,
    default_mini_swe_agent_spec,
    evaluate_external_task,
    evaluate_task,
    load_suite_dir,
)
from harness.llama_client import chat


def wilson_ci(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Compute a 95% Wilson score confidence interval (percentages 0..100)."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare internal harness against mini-SWE-agent and Aider on local llama-server."
    )
    parser.add_argument("--suite-dir", type=Path, default=Path("benchmarks/exercism"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocol.json")
    )
    parser.add_argument(
        "--work-root", type=Path, default=Path(".runs/exercism_3way_work")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(".runs/exercism_3way_artifacts")
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path(".runs/exercism_3way_traces.sqlite3")
    )
    parser.add_argument("--llama-url", default="http://127.0.0.1:9931")
    parser.add_argument("--gguf-path", type=Path, default=None)
    parser.add_argument("--mini-version", default="2.4.6")
    parser.add_argument("--aider-version", default="0.86.2")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--harnesses", default="internal,mini-swe-agent,aider")
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _checkpoint_path(
    checkpoints_dir: Path, trial: int, harness: str, task_id: str
) -> Path:
    safe_task = task_id.replace("/", "_")
    return checkpoints_dir / f"trial{trial}__{harness}__{safe_task}.json"


def _summarize_report(
    *,
    records: list[dict[str, Any]],
    harnesses: list[str],
    task_ids: list[str],
    trials: int,
    protocol: FrozenEvaluationProtocol,
    mini_version: str,
    aider_version: str,
) -> tuple[dict[str, Any], str]:
    by_harness: dict[str, list[dict[str, Any]]] = {h: [] for h in harnesses}
    for rec in records:
        h = str(rec["harness"])
        if h in by_harness:
            by_harness[h].append(rec)

    aggregates: dict[str, dict[str, Any]] = {}
    for h in harnesses:
        rows = by_harness[h]
        total = len(rows)
        resolved = sum(1 for r in rows if r["resolved"])
        lo, hi = wilson_ci(resolved, total)
        mean_wall = (
            sum(float(r["wall_clock_seconds"]) for r in rows) / total if total else 0.0
        )
        outcomes = Counter(str(r["outcome"]) for r in rows)
        calls_vals = [
            r["model_calls"] for r in rows if r.get("model_calls") is not None
        ]
        tok_vals = [
            r["total_tokens"] for r in rows if r.get("total_tokens") is not None
        ]
        aggregates[h] = {
            "total": total,
            "resolved": resolved,
            "rate_pct": round((resolved / total) * 100.0, 1) if total else 0.0,
            "wilson_95_ci": [round(lo, 1), round(hi, 1)],
            "mean_wall_seconds": round(mean_wall, 2),
            "mean_model_calls": (
                round(sum(calls_vals) / len(calls_vals), 2) if calls_vals else None
            ),
            "mean_total_tokens": (
                round(sum(tok_vals) / len(tok_vals), 1) if tok_vals else None
            ),
            "outcomes": dict(sorted(outcomes.items())),
        }

    # Per-task matrix
    per_task: dict[str, dict[str, Any]] = {}
    for tid in task_ids:
        entry: dict[str, Any] = {}
        for h in harnesses:
            t_rows = [r for r in by_harness[h] if r["task_id"] == tid]
            entry[h] = {
                "resolved": sum(1 for r in t_rows if r["resolved"]),
                "total": len(t_rows),
                "mean_wall_seconds": (
                    round(
                        sum(float(r["wall_clock_seconds"]) for r in t_rows)
                        / len(t_rows),
                        2,
                    )
                    if t_rows
                    else 0.0
                ),
                "outcomes": [r["outcome"] for r in t_rows],
            }
        per_task[tid] = entry

    # Pairwise vs internal
    pairwise: dict[str, dict[str, int]] = {}
    if "internal" in harnesses:
        for other in harnesses:
            if other == "internal":
                continue
            wins = losses = ties = 0
            for tid in task_ids:
                int_r = per_task[tid]["internal"]["resolved"]
                oth_r = per_task[tid][other]["resolved"]
                if int_r > oth_r:
                    wins += 1
                elif int_r < oth_r:
                    losses += 1
                else:
                    ties += 1
            pairwise[f"internal_vs_{other}"] = {
                "internal_wins": wins,
                "other_wins": losses,
                "ties": ties,
            }

    payload = {
        "protocol_version": protocol.protocol_version,
        "model": protocol.gguf_filename,
        "context_window": protocol.context_window,
        "trials": trials,
        "task_count": len(task_ids),
        "harnesses": harnesses,
        "versions": {
            "mini-swe-agent": mini_version,
            "aider": aider_version,
        },
        "aggregates": aggregates,
        "pairwise": pairwise,
        "per_task": per_task,
        "records": records,
    }

    lines = [
        "# Exercism Python 3-Way Harness Comparison",
        "",
        f"- **Model:** `{protocol.gguf_filename}` (`{protocol.quantization}`, `context_window={protocol.context_window}`)",
        f"- **Tasks:** `{len(task_ids)}` curated Exercism Python exercises × `{trials}` trials = `{len(task_ids) * trials}` runs per harness",
        f"- **Pinned external versions:** `mini-swe-agent=={mini_version}`, `aider-chat=={aider_version}`",
        "",
        "## Overall Summary",
        "",
        "| Harness | Resolved | Rate | 95% Wilson CI | Mean Wall (s) | Outcomes |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for h in harnesses:
        agg = aggregates[h]
        ci = agg["wilson_95_ci"]
        out_str = ", ".join(f"{k}={v}" for k, v in agg["outcomes"].items())
        lines.append(
            f"| `{h}` | {agg['resolved']}/{agg['total']} | {agg['rate_pct']:.1f}% | "
            f"[{ci[0]:.1f}%, {ci[1]:.1f}%] | {agg['mean_wall_seconds']:.2f} | {out_str} |"
        )

    if pairwise:
        lines.extend(["", "## Pairwise Per-Task Comparison (vs. `internal`)", ""])
        for key, pw in pairwise.items():
            other_name = key.replace("internal_vs_", "")
            lines.append(
                f"- **`internal` vs `{other_name}`:** "
                f"`internal` wins **{pw['internal_wins']}**, `{other_name}` wins **{pw['other_wins']}**, ties **{pw['ties']}**"
            )

    lines.extend(["", "## Per-Task Resolution Matrix", ""])
    hdr = "| Task | " + " | ".join(f"`{h}`" for h in harnesses) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in harnesses) + " |"
    lines.extend([hdr, sep])
    for tid in task_ids:
        cells = []
        for h in harnesses:
            info = per_task[tid][h]
            cells.append(
                f"{info['resolved']}/{info['total']} ({info['mean_wall_seconds']:.1f}s)"
            )
        lines.append(f"| `{tid}` | " + " | ".join(cells) + " |")

    return payload, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = FrozenEvaluationProtocol.from_json_file(args.protocol)
    manifests = list(load_suite_dir(args.suite_dir))
    if args.task_filter:
        manifests = [m for m in manifests if args.task_filter in m.task_id]
    if not manifests:
        raise SystemExit("No manifests matched.")

    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    run_timeout = (
        args.run_timeout if args.run_timeout is not None else protocol.run_timeout
    )

    gguf_sha = (
        compute_file_sha256(args.gguf_path) if args.gguf_path else protocol.gguf_sha256
    )
    spec = ModelSpec(
        gguf_filename=protocol.gguf_filename,
        gguf_sha256=gguf_sha,
        context_window=protocol.context_window,
        tool_format=protocol.tool_format,
    )
    config = ExperimentConfig.capture(spec)
    model_fn = functools.partial(
        chat,
        base_url=args.llama_url,
        model=protocol.gguf_filename,
        max_tokens=protocol.max_tokens,
    )

    base_v1 = f"{args.llama_url.rstrip('/')}/v1"
    mini_spec = default_mini_swe_agent_spec(
        protocol=protocol,
        base_url=base_v1,
        pinned_version=args.mini_version,
    )
    mini_cmd = (
        "uvx",
        "--python",
        "3.11",
        "--from",
        f"mini-swe-agent=={args.mini_version}",
        "mini",
        "--config",
        "mini_textbased.yaml",
        "--model-class",
        "litellm_textbased",
        "--model",
        f"openai/{protocol.gguf_filename}",
        "--cost-limit",
        "0",
        "--task",
        "{task}",
        "--yolo",
        "--exit-immediately",
    )

    aider_spec = default_aider_spec(
        protocol=protocol,
        base_url=base_v1,
        pinned_version=args.aider_version,
    )
    aider_cmd = (
        "uvx",
        "--python",
        "3.12",
        "--from",
        f"aider-chat=={args.aider_version}",
        "--with",
        "scipy>=1.15.2",
        "aider",
        "--model",
        f"openai/{protocol.gguf_filename}",
        "--openai-api-base",
        "{base_url}",
        "--openai-api-key",
        "local-llama",
        "--yes-always",
        "--no-auto-commits",
        "--no-stream",
        "--no-show-model-warnings",
        "--no-analytics",
        "--no-check-update",
        "--map-tokens",
        "0",
        "--test-cmd",
        f"{sys.executable} -m pytest -q {{test_file}}",
        "--auto-test",
        "--message",
        "{task}",
        "{solution_file}",
    )

    checkpoints_dir = args.artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for trial in range(1, args.trials + 1):
        for manifest in manifests:
            for harness_name in harnesses:
                cp = _checkpoint_path(
                    checkpoints_dir, trial, harness_name, manifest.task_id
                )
                if args.resume and cp.is_file():
                    rec = json.loads(cp.read_text(encoding="utf-8"))
                    records.append(rec)
                    print(
                        f"[RESUME] trial={trial} {harness_name:14s} {manifest.task_id}: "
                        f"resolved={rec['resolved']} outcome={rec['outcome']} ({rec['wall_clock_seconds']}s)",
                        flush=True,
                    )
                    continue

                trial_work = args.work_root / f"trial_{trial}" / harness_name
                trial_art = args.artifacts_dir / f"trial_{trial}" / harness_name
                if harness_name == "internal":
                    res = evaluate_task(
                        manifest,
                        work_root=trial_work,
                        model=model_fn,
                        artifacts_root=trial_art,
                        max_model_calls=protocol.max_model_calls,
                        max_steps=protocol.max_steps,
                        max_consecutive_invalid_actions=protocol.max_consecutive_invalid_actions,
                        request_timeout=protocol.request_timeout,
                        run_timeout=run_timeout,
                        oracle_timeout=protocol.oracle_timeout,
                        setup_timeout=protocol.setup_timeout,
                        cleanup_workspaces=protocol.cleanup_workspaces,
                        experiment_config=config,
                        context_strategy="editable_context",
                        retrieval_backend="code_aware",
                        token_budget=TokenBudget(
                            context_window=protocol.context_window,
                            output_reserve=protocol.output_reserve,
                        ),
                        token_counter=LlamaTokenCounter(base_url=args.llama_url),
                        max_tool_output_bytes=max(
                            4096, min(16384, protocol.context_window)
                        ),
                    )
                elif harness_name == "mini-swe-agent":
                    res = evaluate_external_task(
                        manifest,
                        work_root=trial_work,
                        spec=mini_spec,
                        command_template=mini_cmd,
                        run_timeout=run_timeout,
                        oracle_timeout=protocol.oracle_timeout,
                        setup_timeout=protocol.setup_timeout,
                        cleanup_workspaces=protocol.cleanup_workspaces,
                    )
                elif harness_name == "aider":
                    res = evaluate_external_task(
                        manifest,
                        work_root=trial_work,
                        spec=aider_spec,
                        command_template=aider_cmd,
                        run_timeout=run_timeout,
                        oracle_timeout=protocol.oracle_timeout,
                        setup_timeout=protocol.setup_timeout,
                        cleanup_workspaces=protocol.cleanup_workspaces,
                    )
                else:
                    raise SystemExit(f"Unknown harness: {harness_name}")

                rec = {
                    "trial": trial,
                    "harness": harness_name,
                    "task_id": manifest.task_id,
                    "difficulty": manifest.metadata.get("difficulty", ""),
                    "resolved": res.outcome == BenchmarkOutcome.RESOLVED,
                    "outcome": res.outcome.value,
                    "agent_status": (
                        res.agent_status.value if res.agent_status is not None else None
                    ),
                    "wall_clock_seconds": round(res.duration_seconds, 2),
                    "model_calls": res.model_calls,
                    "steps": res.steps,
                    "total_tokens": res.total_tokens,
                    "patch_bytes": len((res.candidate_patch or "").encode("utf-8")),
                    "error_detail": res.error_detail,
                }
                cp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
                records.append(rec)
                print(
                    f"[DONE]   trial={trial} {harness_name:14s} {manifest.task_id}: "
                    f"resolved={rec['resolved']} outcome={rec['outcome']} ({rec['wall_clock_seconds']}s)",
                    flush=True,
                )

    payload, md = _summarize_report(
        records=records,
        harnesses=harnesses,
        task_ids=[m.task_id for m in manifests],
        trials=args.trials,
        protocol=protocol,
        mini_version=args.mini_version,
        aider_version=args.aider_version,
    )
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "multi_trial_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.artifacts_dir / "multi_trial_comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
