"""Run the Phase 7 controlled four-way internal ablation suite."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from harness.context import LlamaTokenCounter
from harness.evaluation import (
    ExperimentConfig,
    FrozenEvaluationProtocol,
    ModelSpec,
    compute_file_sha256,
    load_suite_dir,
    run_four_way_ablation,
)
from harness.llama_client import chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline, retrieval_only, clm_only, and combined ablations."
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path("benchmarks/smoke"),
        help="Directory containing task manifest *.json files.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmarks/protocol.json"),
        help="Path to frozen Phase 7 protocol JSON.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(".runs/phase7_ablation_work"),
        help="Temporary root for isolated agent and evaluator checkouts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".runs/phase7_ablation_artifacts"),
        help="Directory for per-condition artifacts and four_way_ablation.md/json.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(".runs/phase7_ablation_traces.sqlite3"),
        help="SQLite database path for structured run traces.",
    )
    parser.add_argument(
        "--llama-url",
        default="http://127.0.0.1:9931",
        help="Base URL for local llama-server.",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help="Optional local GGUF file path for SHA-256 verification.",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Optional llama-server PID for RSS sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = FrozenEvaluationProtocol.from_json_file(args.protocol)
    manifests = load_suite_dir(args.suite_dir)

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

    report = run_four_way_ablation(
        manifests,
        work_root=args.work_root,
        model=model_fn,
        artifacts_root=args.artifacts_dir,
        db_path=args.db_path,
        protocol=protocol,
        experiment_config=config,
        server_pid=args.server_pid,
        token_counter=LlamaTokenCounter(base_url=args.llama_url),
    )
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
