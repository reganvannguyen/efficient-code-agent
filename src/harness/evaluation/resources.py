"""macOS unified memory, swap, and process RSS observation helpers."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass

_SWAP_PAIR_RE = re.compile(
    r"(total|used)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?B?)",
    re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}


def _to_bytes(amount: str, unit: str) -> int | None:
    multiplier = _UNIT_MULTIPLIERS.get(unit.upper())
    if multiplier is None:
        return None
    try:
        return int(float(amount) * multiplier)
    except ValueError:
        return None


def parse_sysctl_swapusage(output: str) -> tuple[int | None, int | None]:
    """Parse ``sysctl vm.swapusage`` into ``(used_bytes, total_bytes)``."""
    used_bytes: int | None = None
    total_bytes: int | None = None
    for match in _SWAP_PAIR_RE.finditer(output):
        label, amount, unit = match.group(1).lower(), match.group(2), match.group(3)
        parsed = _to_bytes(amount, unit)
        if label == "used":
            used_bytes = parsed
        elif label == "total":
            total_bytes = parsed
    return used_bytes, total_bytes


def parse_memory_pressure(output: str) -> str | None:
    """Extract normalized memory pressure level (normal/warn/critical) if present."""
    lowered = output.lower()
    for level in ("critical", "warn", "normal"):
        if re.search(rf"\b{level}\b", lowered):
            return level
    return None


def parse_ps_rss_kib(output: str) -> int | None:
    """Parse ``ps -o rss=`` KiB output into bytes, or None if unavailable."""
    stripped = output.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    try:
        kib = int(first_line)
    except ValueError:
        return None
    if kib < 0:
        return None
    return kib * 1024


def _run_probe(cmd: list[str], *, timeout: float) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time observation of host swap, memory pressure, and server RSS."""

    timestamp: float
    swap_used_bytes: int | None = None
    swap_total_bytes: int | None = None
    memory_pressure_level: str | None = None
    process_rss_bytes: int | None = None
    collection_method: str = "sysctl_vm_swapusage+memory_pressure+ps"
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize snapshot to a JSON-compatible dictionary."""
        return asdict(self)


def sample_macos_memory(
    server_pid: int | None = None,
    *,
    timeout: float = 5.0,
) -> MemorySnapshot:
    """Sample macOS swap, memory pressure, and optional llama-server RSS.

    Missing or unsupported measurements remain ``None`` and are never estimated.
    """
    notes: list[str] = []

    swap_out = _run_probe(["sysctl", "vm.swapusage"], timeout=timeout)
    if swap_out is not None:
        swap_used, swap_total = parse_sysctl_swapusage(swap_out)
    else:
        swap_used, swap_total = None, None
        notes.append("sysctl vm.swapusage unavailable")

    pressure_out = _run_probe(["memory_pressure", "-Q"], timeout=timeout)
    pressure_level = (
        parse_memory_pressure(pressure_out) if pressure_out is not None else None
    )
    if pressure_level is None:
        notes.append("memory_pressure unavailable")

    rss_bytes: int | None = None
    if server_pid is not None:
        ps_out = _run_probe(
            ["ps", "-o", "rss=", "-p", str(server_pid)], timeout=timeout
        )
        rss_bytes = parse_ps_rss_kib(ps_out) if ps_out is not None else None
        if rss_bytes is None:
            notes.append(f"ps rss unavailable for pid {server_pid}")

    return MemorySnapshot(
        timestamp=time.time(),
        swap_used_bytes=swap_used,
        swap_total_bytes=swap_total,
        memory_pressure_level=pressure_level,
        process_rss_bytes=rss_bytes,
        notes="; ".join(notes) if notes else None,
    )
