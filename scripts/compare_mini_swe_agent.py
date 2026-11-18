"""Run an external comparison against mini-SWE-agent on the local llama-server."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from harness.context import LlamaTokenCounter, TokenBudget
from harness.evaluation import (
    ExperimentConfig,
    FrozenEvaluationProtocol,
    ModelSpec,
    compare_with_external_harness,
    compute_file_sha256,
    default_mini_swe_agent_spec,
    evaluate_suite,
    load_suite_dir,
)
from harness.llama_client import chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare internal harness against mini-SWE-agent on local llama-server."
    )
    parser.add_argument("--suite-dir", type=Path, default=Path("benchmarks/smoke"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocol.json")
    )
    parser.add_argument("--work-root", type=Path, default=Path(".runs/phase7_ext_work"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path(".runs/phase7_ext_artifacts")
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path(".runs/phase7_ext_traces.sqlite3")
    )
    parser.add_argument("--llama-url", default="http://127.0.0.1:9931")
    parser.add_argument("--gguf-path", type=Path, default=None)
    parser.add_argument("--mini-version", default="2.4.6")
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

    internal_summary = evaluate_suite(
        manifests,
        work_root=args.work_root / "internal",
        model=model_fn,
        artifacts_root=args.artifacts_dir / "internal",
        db_path=args.db_path,
        max_model_calls=protocol.max_model_calls,
        max_steps=protocol.max_steps,
        max_consecutive_invalid_actions=protocol.max_consecutive_invalid_actions,
        request_timeout=protocol.request_timeout,
        run_timeout=protocol.run_timeout,
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
    )

    ext_spec = default_mini_swe_agent_spec(
        protocol=protocol,
        base_url=f"{args.llama_url.rstrip('/')}/v1",
        pinned_version=args.mini_version,
    )
    command_template = (
        "uvx",
        "--python",
        "3.11",
        "--from",
        f"mini-swe-agent=={args.mini_version}",
        "mini",
        "--model",
        f"openai/{protocol.gguf_filename}",
        "--cost-limit",
        "0",
        "--task",
        "{task}",
        "--yolo",
        "--exit-immediately",
    )
    report = compare_with_external_harness(
        list(manifests),
        internal_label="combined",
        internal_summary=internal_summary,
        external_spec=ext_spec,
        work_root=args.work_root / "external",
        artifacts_root=args.artifacts_dir,
        command_template=command_template,
        run_timeout=protocol.run_timeout,
        oracle_timeout=protocol.oracle_timeout,
        setup_timeout=protocol.setup_timeout,
        cleanup_workspaces=protocol.cleanup_workspaces,
        limitations=(
            "mini-swe-agent requires Python >= 3.10 (invoked via uvx --python 3.11) while this harness runs on Python 3.9.6.",
            "mini-swe-agent uses unrestricted bash execution in the checkout rather than ToolPolicy-governed JSON-schema tools.",
        ),
    )
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
