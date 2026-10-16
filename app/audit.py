"""Append-only audit entries with hashes instead of raw request payloads."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.schemas import AuditLogEntry


def hash_payload(payload: Any) -> str:
    """Hash canonical JSON so equivalent mappings produce the same digest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AuditLog:
    """Append records in memory and optionally to a newline-delimited JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = asyncio.Lock()
        self._entries: list[AuditLogEntry] = []

    async def append(self, entry: AuditLogEntry) -> None:
        """Append one entry; existing entries are never edited or removed."""
        stored = AuditLogEntry.model_validate(entry.model_dump())
        async with self._lock:
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(stored.model_dump_json() + "\n")
            self._entries.append(stored)

    async def entries(self) -> list[AuditLogEntry]:
        """Return detached copies so readers cannot mutate the log."""
        async with self._lock:
            return [AuditLogEntry.model_validate(item.model_dump()) for item in self._entries]
