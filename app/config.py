"""Minimal local environment loading shared by the runtime components."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env() -> None:
    """Load simple KEY=VALUE settings from the repository .env without overriding."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip("\"'")
