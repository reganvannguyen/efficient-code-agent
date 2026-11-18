"""Run repeated Phase 7 four-way ablations and generate multi-trial analysis."""

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
    run_repeated_ablations,
)
from harness.llama_client import chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated 4-way ablations and compute accuracy/token/RAM analysis."
    )
    parser.add_argument("--suite-dir", type=Path, default=Path("benchmarks/smoke"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocol.json")
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--work-root", type=Path, default=Path(".runs/phase7_trials_work")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(".runs/phase7_trials_artifacts")
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path(".runs/phase7_trials_traces.sqlite3")
    )
    parser.add_argument("--llama-url", default="http://127.0.0.1:9931")
    parser.add_argument("--gguf-path", type=Path, default=None)
    parser.add_argument("--server-pid", type=int, default=None)
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

    _, analysis = run_repeated_ablations(
        manifests,
        trials=args.trials,
        work_root=args.work_root,
        model=model_fn,
        artifacts_root=args.artifacts_dir,
        db_path=args.db_path,
        protocol=protocol,
        experiment_config=config,
        server_pid=args.server_pid,
        token_counter=LlamaTokenCounter(base_url=args.llama_url),
    )
    print(analysis.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
