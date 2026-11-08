"""Command-line interface for the harness."""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

from harness.agent.runtime import (
    CODING_SYSTEM_PROMPT,
    DEFAULT_MAX_PROMPT_BYTES,
    READ_ONLY_SYSTEM_PROMPT,
    action_schema,
    run_agent,
)
from harness.agent.state import RunStatus
from harness.llama_client import ChatResponse, LlamaClientError, chat
from harness.tools.policy import ApprovalRequest, ToolMode, ToolPolicy


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
    run_parser.add_argument("--coding", action="store_true")
    run_parser.add_argument("--auto-approve", action="store_true")
    run_parser.add_argument("--test-target", action="append", default=None)
    run_parser.add_argument("--test-timeout", type=float, default=None)
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
    if (
        args.auto_approve
        or args.test_target is not None
        or args.test_timeout is not None
    ) and not args.coding:
        parser.error("--auto-approve and test options require --coding")
    if args.auto_approve and not args.coding:
        parser.error("--auto-approve requires --coding")
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
    if args.test_timeout is not None and (
        not math.isfinite(args.test_timeout) or args.test_timeout <= 0
    ):
        parser.error("--test-timeout must be finite and positive")

    policy = None
    approve = None
    prompt = READ_ONLY_SYSTEM_PROMPT
    if args.coding:
        mode = ToolMode.AUTO if args.auto_approve else ToolMode.ASK
        policy = ToolPolicy(
            edit_file=mode,
            run_tests=mode,
            test_targets=tuple(args.test_target or ["."]),
            test_timeout=60.0 if args.test_timeout is None else args.test_timeout,
        )
        prompt = CODING_SYSTEM_PROMPT
        approve = None if args.auto_approve else _approve_in_terminal
    model = partial(
        chat,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        response_schema=action_schema(policy),
        system_prompt=prompt,
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
            max_prompt_bytes=DEFAULT_MAX_PROMPT_BYTES - len(prompt.encode("utf-8")),
            policy=policy,
            approve=approve,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if run.status is RunStatus.COMPLETED:
        print(run.final_response or "")
    detail = _STOP_DETAILS.get(run.stop_reason, "run stopped")
    if run.status is RunStatus.COMPLETED:
        verification = (
            " (verified completion)"
            if run.verification_quality.value == "verified"
            else " (unverified model completion)"
        )
    else:
        verification = ""
    print(
        f"harness: {detail}; status={run.status.value} "
        f"stop_reason={run.stop_reason} model_calls={run.model_calls} "
        f"steps={run.steps}{verification}",
        file=sys.stderr,
    )
    print(f"harness: run_id={run.run_id}", file=sys.stderr)
    if run.benchmark_outcome is not None:
        print(
            f"harness: benchmark_outcome={run.benchmark_outcome.value}", file=sys.stderr
        )
    if run.total_tokens is not None:
        print(
            f"harness: tokens=prompt:{run.total_prompt_tokens or 0},"
            f"completion:{run.total_completion_tokens or 0},"
            f"total:{run.total_tokens}",
            file=sys.stderr,
        )
    print(f"harness: verification={run.verification_quality.value}", file=sys.stderr)
    if run.changed_paths:
        print(f"harness: changed_paths={','.join(run.changed_paths)}", file=sys.stderr)
    if run.last_write_state == "unknown":
        print(
            "harness: write_state=unknown (write outcome is uncertain)", file=sys.stderr
        )
    if run.candidate_diff:
        print(
            f"harness: candidate_diff:\n{run.candidate_diff.strip()}", file=sys.stderr
        )
    if run.test_invocations:
        print(f"harness: tests_executed={len(run.test_invocations)}", file=sys.stderr)
    if run.last_test_result:
        test_data = run.last_test_result.get("data", {})
        if isinstance(test_data, dict):
            print(
                f"harness: last_test_exit_code={test_data.get('exit_code')} "
                f"outcome={run.last_test_result.get('error_code') or 'passed'}",
                file=sys.stderr,
            )
    if run.unresolved_issues:
        print(
            f"harness: unresolved_issues={'; '.join(run.unresolved_issues)}",
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
    "no_progress": "agent stopped due to repeated actions without progress",
}


def _approve_in_terminal(request: ApprovalRequest) -> bool:
    """Ask on stderr/stdin only when both terminal channels are interactive."""
    if not sys.stdin.isatty() or not sys.stderr.isatty():
        return False
    if request.tool == "edit_file":
        print(
            f"Approve edit in {request.root}: {request.path} "
            f"(sha256 {request.expected_sha256})?",
            file=sys.stderr,
        )
        print(
            f"old_text={json.dumps(request.old_text, ensure_ascii=True)}",
            file=sys.stderr,
        )
        print(
            f"new_text={json.dumps(request.new_text, ensure_ascii=True)}",
            file=sys.stderr,
        )
        print(request.diff[:8192], file=sys.stderr)
        if request.diff_truncated or len(request.diff) > 8192:
            print("[diff preview truncated]", file=sys.stderr)
    else:
        print(
            f"Approve tests in {request.root}: preset={request.preset} "
            f"target={request.target} argv={json.dumps(request.argv)}",
            file=sys.stderr,
        )
    try:
        print("Approve? [y/N] ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline()
    except EOFError:
        return False
    if answer == "":
        return False
    return answer.strip().casefold() in {"y", "yes"}


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
