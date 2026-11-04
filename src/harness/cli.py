"""Command-line interface for the harness."""

from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

from harness.agent.runtime import (
    DEFAULT_MAX_PROMPT_BYTES,
    READ_ONLY_ACTION_SCHEMA,
    READ_ONLY_SYSTEM_PROMPT,
    run_agent,
)
from harness.agent.state import RunStatus
from harness.llama_client import ChatResponse, LlamaClientError, chat


def main(argv: list[str] | None = None) -> None:
    """Run the requested command or start the independent-prompt loop."""
    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        interactive()
        return

    parser = argparse.ArgumentParser(prog="harness")
    commands = parser.add_subparsers(dest="command", required=True)
    chat_parser = commands.add_parser("chat", help="send a prompt to a local model")
    chat_parser.add_argument("prompt")
    chat_parser.add_argument("--base-url", default="http://127.0.0.1:9931")
    chat_parser.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    chat_parser.add_argument("--timeout", type=float, default=120)
    chat_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser = commands.add_parser("run", help="answer a task about a repository")
    run_parser.add_argument("task")
    run_parser.add_argument("--repo", type=Path, default=Path.cwd())
    run_parser.add_argument("--base-url", default="http://127.0.0.1:9931")
    run_parser.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.add_argument("--timeout", type=float, default=120)
    run_parser.add_argument("--run-timeout", type=float, default=300)
    run_parser.add_argument("--max-model-calls", type=int, default=20)
    run_parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args(args_list)
    if args.command == "run":
        _run(args, parser)
        return
    if not args.prompt.strip():
        parser.error("chat prompt must not be blank")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    try:
        response = chat(
            args.prompt,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
    except (LlamaClientError, ValueError) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    _print_response(response)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run a bounded read-only repository task and report its terminal state."""
    if not args.task.strip():
        parser.error("run task must not be blank")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.max_model_calls <= 0:
        parser.error("--max-model-calls must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    for name, value in (
        ("--timeout", args.timeout),
        ("--run-timeout", args.run_timeout),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be a finite positive number")

    model = partial(
        chat,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        response_schema=READ_ONLY_ACTION_SCHEMA,
        system_prompt=READ_ONLY_SYSTEM_PROMPT,
    )
    try:
        run = run_agent(
            args.task,
            args.repo,
            model=model,
            request_timeout=args.timeout,
            run_timeout=args.run_timeout,
            max_model_calls=args.max_model_calls,
            max_steps=args.max_steps,
            max_prompt_bytes=DEFAULT_MAX_PROMPT_BYTES
            - len(READ_ONLY_SYSTEM_PROMPT.encode("utf-8")),
        )
    except ValueError as exc:
        parser.error(str(exc))

    if run.status is RunStatus.COMPLETED:
        print(run.final_response or "")
    detail = _STOP_DETAILS.get(run.stop_reason, "run stopped")
    verification = (
        " (unverified model completion)" if run.status is RunStatus.COMPLETED else ""
    )
    print(
        f"harness: {detail}; status={run.status.value} "
        f"stop_reason={run.stop_reason} model_calls={run.model_calls} "
        f"steps={run.steps}{verification}",
        file=sys.stderr,
    )
    if run.status is RunStatus.CANCELLED:
        raise SystemExit(130)
    if run.status is not RunStatus.COMPLETED:
        raise SystemExit(1)


_STOP_DETAILS = {
    "model_complete": "model proposed a completion",
    "model_call_limit": "model call limit reached",
    "step_limit": "step limit reached",
    "invalid_action_limit": "invalid action limit reached",
    "prompt_budget": "prompt byte limit reached",
    "model_output_limit": "model output limit reached",
    "run_timeout": "run deadline reached",
    "model_timeout": "model request timed out",
    "model_error": "model server unavailable or returned an invalid response",
    "cancelled": "run cancelled",
}


def interactive() -> None:
    """Read independent prompts and send each to the default local model."""
    print("Interactive mode: prompts are independent. Type /exit to quit.")
    while True:
        try:
            prompt = input("You > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if prompt.strip() == "/exit":
            return
        if not prompt.strip():
            continue

        try:
            response = chat(prompt)
        except KeyboardInterrupt:
            print()
            return
        except (LlamaClientError, ValueError) as exc:
            print(f"harness: {exc}", file=sys.stderr)
            continue

        _print_response(response)


def _print_response(response: ChatResponse) -> None:
    """Print response text and available usage in the CLI's standard format."""
    print(response.text)
    usage = (
        f"prompt={response.prompt_tokens} completion={response.completion_tokens} "
        f"total={response.total_tokens}"
    )
    if any(
        count is not None
        for count in (
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
        )
    ):
        print(f"token usage: {usage}", file=sys.stderr)
