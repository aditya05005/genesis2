"""In-memory tool schemas with versions and rollback references."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSchema:
    """A tool's current agent-facing request schema."""

    name: str
    url: str
    schema: dict[str, Any]
    version: int
    idempotent: bool = False


class SchemaRegistry:
    """Protect schema reads and successful updates with one async lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tools: dict[str, ToolSchema] = {}
        self._history: dict[str, dict[str, ToolSchema]] = {}

    async def register(
        self, name: str, url: str, schema: dict[str, Any], *, idempotent: bool = False
    ) -> ToolSchema:
        """Register an initial schema, or return the existing tool unchanged."""
        async with self._lock:
            if name in self._tools:
                return self._copy(self._tools[name])
            tool = ToolSchema(name, url, deepcopy(schema), 1, idempotent)
            self._tools[name] = tool
            self._history[name] = {}
            return self._copy(tool)

    async def get(self, name: str) -> ToolSchema:
        """Return a snapshot so callers cannot mutate the registry by reference."""
        async with self._lock:
            try:
                return self._copy(self._tools[name])
            except KeyError as exc:
                raise KeyError(f"tool {name!r} is not registered") from exc

    async def publish_after_success(
        self, name: str, schema: dict[str, Any], *, expected_version: int
    ) -> tuple[ToolSchema, str]:
        """Advance a schema only if the successful replay used the current version."""
        async with self._lock:
            current = self._tools[name]
            if current.version != expected_version:
                raise RuntimeError("schema changed during the tool call")
            rollback_ref = f"{name}:v{current.version}"
            self._history[name][rollback_ref] = self._copy(current)
            updated = ToolSchema(
                current.name, current.url, deepcopy(schema), current.version + 1,
                current.idempotent,
            )
            self._tools[name] = updated
            return self._copy(updated), rollback_ref

    async def rollback(self, name: str, rollback_ref: str) -> ToolSchema:
        """Restore a schema snapshot and advance version to keep versions monotonic."""
        async with self._lock:
            snapshot = self._history[name].get(rollback_ref)
            if snapshot is None:
                raise KeyError(f"unknown rollback reference {rollback_ref!r}")
            current = self._tools[name]
            restored = ToolSchema(
                current.name, current.url, deepcopy(snapshot.schema), current.version + 1,
                current.idempotent,
            )
            self._tools[name] = restored
            return self._copy(restored)

    @staticmethod
    def _copy(tool: ToolSchema) -> ToolSchema:
        return ToolSchema(tool.name, tool.url, deepcopy(tool.schema), tool.version, tool.idempotent)


async def register_mock_tool(registry: SchemaRegistry, base_url: str) -> ToolSchema:
    """Register the mock's initial string-ID contract for the demo executor."""
    return await registry.register(
        "lookup_user",
        f"{base_url.rstrip('/')}/tool",
        {
            "type": "object",
            "required": ["user_id"],
            "properties": {"user_id": {"type": "string"}},
        },
    )
