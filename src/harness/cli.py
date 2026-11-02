"""Command-line interface for the harness."""

from __future__ import annotations

import argparse
import math
import sys

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
    args = parser.parse_args(args_list)
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
