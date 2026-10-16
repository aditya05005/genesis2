"""Registry patch history, expiration, and diagnosis single-flight coverage."""

import asyncio
from typing import Any

import jsonpatch

from app.registry import SchemaRegistry


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_registry_applies_json_patch_and_keeps_it_with_rollback_history() -> None:
    async def scenario() -> None:
        registry = SchemaRegistry()
        await registry.register(
            "lookup", "http://upstream", {
                "type": "object", "required": ["user_id"],
                "properties": {"user_id": {"type": "string"}},
            }
        )
        patch = [
            {"op": "remove", "path": "/properties/user_id"},
            {"op": "add", "path": "/properties/account_id", "value": {"type": "string"}},
            {"op": "replace", "path": "/required/0", "value": "account_id"},
        ]
        before = await registry.get("lookup")
        updated, rollback_ref = await registry.publish_after_success(
            "lookup", patch, expected_version=before.version
        )
        history = await registry.history("lookup")

        assert jsonpatch.JsonPatch(patch).apply(before.schema) == updated.schema
        assert updated.version == 2
        assert history[rollback_ref]["patch"] == patch
        restored = await registry.rollback("lookup", rollback_ref)
        assert restored.version == 3
        assert restored.schema == before.schema

    _run(scenario())


def test_registry_singleflight_coalesces_diagnosis_for_same_signature() -> None:
    async def scenario() -> None:
        registry = SchemaRegistry()
        calls = 0

        async def diagnose() -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.02)
            return "diagnosis"

        results = await asyncio.gather(*[
            registry.singleflight("lookup", "same-error", diagnose)
            for _ in range(8)
        ])
        assert calls == 1
        assert results == ["diagnosis"] * 8

    _run(scenario())
