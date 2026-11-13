"""Pinned model, harness, and host configuration snapshots for reproducible runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


def compute_file_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str | None:
    """Compute a file's SHA-256 digest using bounded streaming reads."""
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def capture_git_commit(
    repo_root: Path | None = None, *, timeout: float = 5.0
) -> str | None:
    """Return the current git HEAD commit SHA, or None if unavailable."""
    cmd = ["git"]
    if repo_root is not None:
        cmd.extend(["-C", str(repo_root)])
    cmd.extend(["rev-parse", "HEAD"])
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
    value = proc.stdout.strip()
    return value or None


def _read_sysctl_str(key: str, *, timeout: float = 3.0) -> str | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", key],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def capture_host_info() -> dict[str, object]:
    """Capture observable OS and hardware details, leaving missing values as None."""
    os_version = platform.platform() or None
    hardware_chip = _read_sysctl_str("machdep.cpu.brand_string")
    if hardware_chip is None:
        machine = platform.machine().strip()
        hardware_chip = machine or None

    raw_ram = _read_sysctl_str("hw.memsize")
    total_ram_bytes: int | None = None
    if raw_ram is not None:
        try:
            parsed = int(raw_ram)
            if parsed > 0:
                total_ram_bytes = parsed
        except ValueError:
            total_ram_bytes = None

    return {
        "os_version": os_version,
        "hardware_chip": hardware_chip,
        "total_ram_bytes": total_ram_bytes,
    }


@dataclass(frozen=True)
class ModelSpec:
    """Pinned local model and inference server configuration."""

    gguf_filename: str
    gguf_sha256: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    llama_cpp_version: str | None = None
    server_args: tuple[str, ...] = ()
    context_window: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    tool_format: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize model specification to a JSON-compatible dictionary."""
        data = asdict(self)
        data["server_args"] = list(self.server_args)
        return data


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete pinned configuration manifest for reproducing an evaluation run."""

    model: ModelSpec
    harness_commit: str | None = None
    os_version: str | None = None
    hardware_chip: str | None = None
    total_ram_bytes: int | None = None
    restart_policy: str = "single_model_sequential"
    background_notes: str | None = None

    @classmethod
    def capture(
        cls,
        model: ModelSpec,
        *,
        repo_root: Path | None = None,
        restart_policy: str = "single_model_sequential",
        background_notes: str | None = None,
    ) -> ExperimentConfig:
        """Build an ExperimentConfig with observed host and git commit metadata."""
        host = capture_host_info()
        return cls(
            model=model,
            harness_commit=capture_git_commit(repo_root),
            os_version=(
                str(host["os_version"]) if host["os_version"] is not None else None
            ),
            hardware_chip=(
                str(host["hardware_chip"])
                if host["hardware_chip"] is not None
                else None
            ),
            total_ram_bytes=(
                int(host["total_ram_bytes"])
                if isinstance(host["total_ram_bytes"], int)
                else None
            ),
            restart_policy=restart_policy,
            background_notes=background_notes,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize experiment configuration to a JSON-compatible dictionary."""
        return {
            "model": self.model.to_dict(),
            "harness_commit": self.harness_commit,
            "os_version": self.os_version,
            "hardware_chip": self.hardware_chip,
            "total_ram_bytes": self.total_ram_bytes,
            "restart_policy": self.restart_policy,
            "background_notes": self.background_notes,
        }
