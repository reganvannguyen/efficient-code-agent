"""Token counting, prompt budget calculation, and deterministic pruning helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from harness.llama_client import LlamaClientError, tokenize


def _require_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _clip_utf8(value: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    raw = value.encode("utf-8", errors="replace")[:byte_limit]
    return raw.decode("utf-8", errors="ignore")


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in a prompt string."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


def estimate_tokens(text: str) -> int:
    """Deterministic UTF-8 byte-based token estimate fallback (4 bytes/token)."""
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8", errors="replace")) + 3) // 4)


@dataclass(frozen=True)
class LlamaTokenCounter:
    """Token counter backed by llama.cpp's ``/tokenize`` endpoint with optional fallback."""

    base_url: str = "http://127.0.0.1:9931"
    timeout: float = 5.0
    fallback: Callable[[str], int] | None = estimate_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens via ``/tokenize``, falling back deterministically if configured."""
        if not text:
            return 0
        try:
            return len(tokenize(text, base_url=self.base_url, timeout=self.timeout))
        except LlamaClientError:
            if self.fallback is not None:
                return self.fallback(text)
            raise


def resolve_token_counter(
    counter: TokenCounter | Callable[[str], int] | None,
) -> Callable[[str], int]:
    """Normalize a TokenCounter, callable, or None into a ``(str) -> int`` function."""
    if counter is None:
        return estimate_tokens
    if isinstance(counter, TokenCounter):
        return counter.count_tokens
    if callable(counter):
        return counter
    raise ValueError("token_counter must be a TokenCounter, callable, or None")


@dataclass(frozen=True)
class TokenBudget:
    """Input token budget derived from context window, reserves, overhead, and margin."""

    context_window: int
    output_reserve: int = 1024
    overhead_tokens: int = 512
    safety_margin: int = 128

    def __post_init__(self) -> None:
        _require_non_negative_int(self.context_window, "context_window")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        _require_non_negative_int(self.output_reserve, "output_reserve")
        _require_non_negative_int(self.overhead_tokens, "overhead_tokens")
        _require_non_negative_int(self.safety_margin, "safety_margin")
        if self.input_budget <= 0:
            raise ValueError(
                "input_budget must be positive "
                f"(got context_window={self.context_window}, "
                f"output_reserve={self.output_reserve}, "
                f"overhead_tokens={self.overhead_tokens}, "
                f"safety_margin={self.safety_margin})"
            )

    @property
    def input_budget(self) -> int:
        """Maximum tokens available for assembled prompt content."""
        return (
            self.context_window
            - self.output_reserve
            - self.overhead_tokens
            - self.safety_margin
        )


@dataclass(frozen=True)
class PromptPruningReport:
    """Structured record of fallback pruning applied during prompt assembly."""

    pruned: bool = False
    dropped_history_count: int = 0
    capped_excerpts_count: int = 0
    omitted_findings_count: int = 0
    trimmed_notes: bool = False
    prompt_tokens: int | None = None
    token_budget: int | None = None
    prompt_bytes: int = 0
    byte_limit: int = 0
    exceeded_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary of fallback pruning."""
        return {
            "pruned": self.pruned,
            "dropped_history_count": self.dropped_history_count,
            "capped_excerpts_count": self.capped_excerpts_count,
            "omitted_findings_count": self.omitted_findings_count,
            "trimmed_notes": self.trimmed_notes,
            "prompt_tokens": self.prompt_tokens,
            "token_budget": self.token_budget,
            "prompt_bytes": self.prompt_bytes,
            "byte_limit": self.byte_limit,
            "exceeded_reason": self.exceeded_reason,
        }


def measure_prompt(
    text: str,
    *,
    token_limit: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[int, int | None]:
    """Return ``(utf8_bytes, token_count)`` for ``text``."""
    byte_count = len(text.encode("utf-8"))
    if token_limit is None and count_tokens is None:
        return byte_count, None
    counter = count_tokens if count_tokens is not None else estimate_tokens
    return byte_count, counter(text)


def within_budget(
    text: str,
    *,
    byte_limit: int,
    token_limit: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[bool, int, int | None]:
    """Check whether ``text`` fits within ``byte_limit`` and optional ``token_limit``."""
    byte_count, token_count = measure_prompt(
        text, token_limit=token_limit, count_tokens=count_tokens
    )
    if byte_count > byte_limit:
        return False, byte_count, token_count
    if (
        token_limit is not None
        and token_count is not None
        and token_count > token_limit
    ):
        return False, byte_count, token_count
    return True, byte_count, token_count


def trim_notes_deterministically(
    notes: str,
    *,
    max_bytes: int | None = None,
    token_limit: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Deterministically trim working notes from the bottom while preserving headings."""
    stripped = notes.strip()
    if not stripped:
        return ""
    if max_bytes is not None and max_bytes <= 0:
        return ""
    if token_limit is not None and token_limit <= 0:
        return ""

    effective_bytes = (
        max_bytes if max_bytes is not None else len(stripped.encode("utf-8"))
    )
    fits, _, _ = within_budget(
        stripped,
        byte_limit=effective_bytes,
        token_limit=token_limit,
        count_tokens=count_tokens,
    )
    if fits:
        return stripped

    marker = "\n[...notes trimmed to fit budget...]"
    lines = stripped.splitlines()
    while lines:
        candidate = "\n".join(lines).rstrip()
        with_marker = f"{candidate}{marker}" if candidate else marker.lstrip()
        fits_marker, _, _ = within_budget(
            with_marker,
            byte_limit=effective_bytes,
            token_limit=token_limit,
            count_tokens=count_tokens,
        )
        if fits_marker:
            return with_marker
        lines.pop()

    clipped = _clip_utf8(stripped, effective_bytes)
    while clipped:
        fits_clip, _, _ = within_budget(
            clipped,
            byte_limit=effective_bytes,
            token_limit=token_limit,
            count_tokens=count_tokens,
        )
        if fits_clip:
            return clipped
        clipped = clipped[:-1]
    return ""
