"""Versioned tool schemas backed by memory or optional Redis storage."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import jsonpatch

from app.config import load_local_env
from app.schemas import MappingOp


@dataclass(frozen=True)
class ToolSchema:
    """A tool's current agent-facing request and response contracts."""

    name: str
    url: str
    schema: dict[str, Any]
    version: int
    idempotent: bool = False
    response_schema: dict[str, Any] | None = None


class SchemaStore(Protocol):
    async def get(self, name: str) -> dict[str, Any] | None: ...
    async def set(self, name: str, value: dict[str, Any], ttl: int) -> None: ...


class MemorySchemaStore:
    """Thread-safe process-local storage with expiration timestamps."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(name)
            if item is None:
                return None
            expires, value = item
            if expires <= time.monotonic():
                self._items.pop(name, None)
                return None
            return deepcopy(value)

    async def set(self, name: str, value: dict[str, Any], ttl: int) -> None:
        with self._lock:
            self._items[name] = (time.monotonic() + ttl, deepcopy(value))


class RedisSchemaStore:
    """Redis JSON storage; import is deferred unless REDIS_URL is configured."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._client = Redis.from_url(url, decode_responses=True)

    async def get(self, name: str) -> dict[str, Any] | None:
        value = await self._client.get(f"healgate:schema:{name}")
        return json.loads(value) if value is not None else None

    async def set(self, name: str, value: dict[str, Any], ttl: int) -> None:
        await self._client.set(f"healgate:schema:{name}", json.dumps(value), ex=ttl)


class SchemaRegistry:
    """Serialize updates, persist RFC 6902 patches, and coalesce diagnosis work."""

    def __init__(self, *, ttl_seconds: int | None = None, store: SchemaStore | None = None) -> None:
        load_local_env()
        self._lock = asyncio.Lock()
        self.ttl_seconds = ttl_seconds or int(os.environ.get("REGISTRY_TTL_SECONDS", "3600"))
        self._memory = MemorySchemaStore()
        redis_url = os.environ.get("REDIS_URL")
        try:
            self._store: SchemaStore = store or (RedisSchemaStore(redis_url) if redis_url else self._memory)
        except Exception:
            self._store = self._memory
        self._singleflight_lock = asyncio.Lock()
        self._inflight: dict[tuple[str, str], asyncio.Future[Any]] = {}

    async def _read(self, name: str) -> dict[str, Any] | None:
        try:
            return await self._store.get(name)
        except Exception:
            if self._store is self._memory:
                raise
            self._store = self._memory
            return await self._memory.get(name)

    async def _write(self, name: str, value: dict[str, Any]) -> None:
        try:
            await self._store.set(name, value, self.ttl_seconds)
        except Exception:
            if self._store is self._memory:
                raise
            self._store = self._memory
            await self._memory.set(name, value, self.ttl_seconds)

    async def register(
        self, name: str, url: str, schema: dict[str, Any], *,
        idempotent: bool = False, response_schema: dict[str, Any] | None = None,
    ) -> ToolSchema:
        async with self._lock:
            current = await self._read(name)
            if current is None:
                current = {
                    "tool": asdict(ToolSchema(name, url, deepcopy(schema), 1, idempotent,
                                               deepcopy(response_schema))),
                    "history": {},
                }
                await self._write(name, current)
            return self._tool(current["tool"])

    async def get(self, name: str) -> ToolSchema:
        async with self._lock:
            record = await self._read(name)
            if record is None:
                raise KeyError(f"tool {name!r} is not registered or its TTL expired")
            return self._tool(record["tool"])

    async def publish_patch(
        self, name: str, patch: list[dict[str, Any]], *, expected_version: int,
    ) -> tuple[ToolSchema, str]:
        """Stage a JSON Patch and save its rollback data before a replay attempt."""
        async with self._lock:
            record = await self._read(name)
            if record is None:
                raise KeyError(f"tool {name!r} is not registered or its TTL expired")
            current = self._tool(record["tool"])
            if current.version != expected_version:
                raise RuntimeError("schema changed during the tool call")
            updated_schema = jsonpatch.JsonPatch(patch).apply(current.schema, in_place=False)
            rollback_ref = f"{name}:v{current.version}"
            record["history"][rollback_ref] = {
                "tool": asdict(current), "patch": deepcopy(patch),
            }
            updated = ToolSchema(
                current.name, current.url, updated_schema, current.version + 1,
                current.idempotent, deepcopy(current.response_schema),
            )
            record["tool"] = asdict(updated)
            await self._write(name, record)
            return self._tool(record["tool"]), rollback_ref

    async def publish_after_success(
        self, name: str, patch: list[dict[str, Any]], *, expected_version: int,
    ) -> tuple[ToolSchema, str]:
        """Backward-compatible alias for callers using the earlier method name."""
        return await self.publish_patch(name, patch, expected_version=expected_version)

    async def rollback(
        self, name: str, rollback_ref: str, *, expected_version: int | None = None,
    ) -> ToolSchema:
        async with self._lock:
            record = await self._read(name)
            if record is None or rollback_ref not in record["history"]:
                raise KeyError(f"unknown rollback reference {rollback_ref!r}")
            current = self._tool(record["tool"])
            if expected_version is not None and current.version != expected_version:
                raise RuntimeError("schema changed before rollback; refusing to overwrite newer state")
            snapshot = record["history"][rollback_ref]["tool"]
            restored = ToolSchema(
                current.name, current.url, deepcopy(snapshot["schema"]), current.version + 1,
                current.idempotent, deepcopy(snapshot.get("response_schema")),
            )
            record["tool"] = asdict(restored)
            await self._write(name, record)
            return self._tool(record["tool"])

    async def history(self, name: str) -> dict[str, Any]:
        async with self._lock:
            record = await self._read(name)
            return deepcopy(record["history"]) if record else {}

    async def singleflight(self, tool: str, signature: str, factory: Any) -> Any:
        """Run one async diagnosis for concurrent failures with the same signature."""
        key = (tool, signature)
        async with self._singleflight_lock:
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
        assert future is not None
        if not owner:
            return await asyncio.shield(future)
        try:
            value = await factory()
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            # Consume the exception if there were no waiters.
            future.exception()
            raise
        finally:
            async with self._singleflight_lock:
                self._inflight.pop(key, None)

    @staticmethod
    def _tool(value: dict[str, Any]) -> ToolSchema:
        return ToolSchema(**deepcopy(value))


def schema_patch_for_ops(schema: dict[str, Any], ops: list[MappingOp]) -> list[dict[str, Any]]:
    """Build an RFC 6902 document describing the agent-facing schema change."""
    updated = deepcopy(schema)
    for op in ops:
        if op.op in ("move", "rename") and op.source and op.target:
            value_schema = _schema_remove(updated, op.source) or {}
            _schema_set(updated, op.target, value_schema)
        elif op.op == "cast" and op.target and op.cast_to:
            value_schema = _schema_get(updated, op.target) or {}
            value_schema["type"] = {
                "int": "integer", "str": "string", "float": "number", "bool": "boolean",
            }[op.cast_to]
            _schema_set(updated, op.target, value_schema)
        elif op.op == "wrap" and op.source and op.target:
            value_schema = _schema_get(updated, op.source) or {}
            _schema_set(updated, op.target, {
                "type": "object", "required": ["value"],
                "properties": {"value": value_schema},
            })
        elif op.op == "add_header" and op.header:
            required = updated.setdefault("required_headers", [])
            if op.header not in required:
                required.append(op.header)
    return jsonpatch.make_patch(schema, updated).patch


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _schema_get(schema: dict[str, Any], pointer: str) -> dict[str, Any] | None:
    current = schema
    try:
        for part in _pointer_parts(pointer):
            current = current.get("properties", {}).get(part)
            if not isinstance(current, dict):
                return None
        return current
    except ValueError:
        return None


def _schema_remove(schema: dict[str, Any], pointer: str) -> dict[str, Any] | None:
    parts = _pointer_parts(pointer)
    parent = schema
    for part in parts[:-1]:
        parent = parent.get("properties", {}).get(part)
        if not isinstance(parent, dict):
            return None
    removed = parent.get("properties", {}).pop(parts[-1], None)
    if "required" in parent:
        parent["required"] = [field for field in parent["required"] if field != parts[-1]]
    return removed


def _schema_set(schema: dict[str, Any], pointer: str, value: dict[str, Any]) -> None:
    parts = _pointer_parts(pointer)
    parent = schema
    for part in parts[:-1]:
        properties = parent.setdefault("properties", {})
        if part not in properties:
            properties[part] = {"type": "object", "properties": {}, "required": []}
        parent.setdefault("required", [])
        if part not in parent["required"]:
            parent["required"].append(part)
        parent = properties[part]
    parent.setdefault("properties", {})[parts[-1]] = value
    parent.setdefault("required", [])
    if parts[-1] not in parent["required"]:
        parent["required"].append(parts[-1])


async def register_mock_tool(registry: SchemaRegistry, base_url: str) -> ToolSchema:
    """Register the initial request and response contracts for the demo tool."""
    return await registry.register(
        "lookup_user", f"{base_url.rstrip('/')}/tool",
        {"type": "object", "required": ["user_id"],
         "properties": {"user_id": {"type": "string"}}},
        response_schema={
            "type": "object", "required": ["ok", "record_id", "user_id"],
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}, "record_id": {"type": "integer"},
                           "user_id": {"type": "string"}},
        },
    )
