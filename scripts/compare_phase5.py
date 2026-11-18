"""Run a head-to-head Phase 5 comparison: bounded_history vs editable_context."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from harness.context import LlamaTokenCounter, TokenBudget
from harness.evaluation import (
    ExperimentConfig,
    ModelSpec,
    compare_strategies,
    compute_file_sha256,
    load_suite_dir,
)
from harness.llama_client import chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare bounded_history vs editable_context on a task suite."
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path("benchmarks/smoke"),
        help="Directory containing task manifest *.json files.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(".runs/phase5_work"),
        help="Temporary root for isolated agent and evaluator checkouts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".runs/phase5_artifacts"),
        help="Directory for per-strategy artifacts and comparison markdown.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(".runs/phase5_traces.sqlite3"),
        help="SQLite database path for structured run traces.",
    )
    parser.add_argument(
        "--llama-url",
        default="http://127.0.0.1:9931",
        help="Base URL for local llama-server.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        help="Model identifier passed to llama-server.",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help="Optional local GGUF file path for SHA-256 pinning.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="Configured context window size (default: 8192).",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Optional llama-server PID for RSS sampling.",
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=16,
        help="Maximum model calls per task.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent loop steps per task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = load_suite_dir(args.suite_dir)

    gguf_sha = compute_file_sha256(args.gguf_path) if args.gguf_path else None
    spec = ModelSpec(
        gguf_filename=args.gguf_path.name if args.gguf_path else args.model_name,
        gguf_sha256=gguf_sha,
        context_window=args.ctx_size,
        tool_format="json_schema",
    )
    config = ExperimentConfig.capture(spec)
    model_fn = functools.partial(chat, base_url=args.llama_url, model=args.model_name)

    report = compare_strategies(
        manifests,
        work_root=args.work_root,
        model=model_fn,
        artifacts_root=args.artifacts_dir,
        db_path=args.db_path,
        max_model_calls=args.max_model_calls,
        max_steps=args.max_steps,
        experiment_config=config,
        server_pid=args.server_pid,
        token_budget=TokenBudget(context_window=args.ctx_size),
        token_counter=LlamaTokenCounter(base_url=args.llama_url),
    )
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
