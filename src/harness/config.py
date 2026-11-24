"""User configuration and localhost server settings resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CONFIG_PATH = Path.home() / ".harness" / "config.json"
LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def validate_local_base_url(base_url: str) -> str:
    """Validate that *base_url* targets localhost and return a normalized URL."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty localhost URL")
    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url must use http:// or https://")
    host = (parsed.hostname or "").lower()
    if host not in LOCALHOST_HOSTS:
        raise ValueError(
            "base_url must point to localhost (127.0.0.1, localhost, or ::1)"
        )
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include query parameters or fragments")
    return cleaned


def load_user_config(path: Path | None = None) -> dict[str, str]:
    """Load saved user settings from *path*, returning an empty dict if absent/invalid."""
    resolved_path = DEFAULT_CONFIG_PATH if path is None else path
    if not resolved_path.is_file():
        return {}
    try:
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    raw_url = raw.get("base_url")
    if isinstance(raw_url, str) and raw_url.strip():
        try:
            result["base_url"] = validate_local_base_url(raw_url)
        except ValueError:
            pass
    raw_model = raw.get("model")
    if isinstance(raw_model, str):
        result["model"] = raw_model.strip()
    return result


def save_user_config(
    *,
    base_url: str | None = None,
    model: str | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Merge and atomically persist user server settings to *path*."""
    resolved_path = DEFAULT_CONFIG_PATH if path is None else path
    current = load_user_config(resolved_path)
    if base_url is not None:
        current["base_url"] = validate_local_base_url(base_url)
    if model is not None:
        current["model"] = model.strip()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(resolved_path)
    return current


def resolve_server_settings(
    *,
    cli_base_url: str | None = None,
    cli_model: str | None = None,
    config_path: Path | None = None,
) -> tuple[str, str]:
    """Resolve (base_url, model) with precedence: CLI > env > config.json > defaults."""
    saved = load_user_config(config_path)
    env_url = os.environ.get("HARNESS_BASE_URL", "").strip()
    env_model = os.environ.get("HARNESS_MODEL", "").strip()

    raw_url = (
        cli_base_url
        if cli_base_url is not None
        else (env_url or saved.get("base_url") or DEFAULT_BASE_URL)
    )
    resolved_url = validate_local_base_url(raw_url)

    if cli_model is not None:
        resolved_model = cli_model.strip()
    elif env_model:
        resolved_model = env_model
    else:
        resolved_model = saved.get("model", "")

    return resolved_url, resolved_model
