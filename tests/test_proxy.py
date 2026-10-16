"""HTTP surface checks for the thin proxy layer."""

import asyncio
from typing import Any

import httpx

from app.audit import AuditLog
from app.executor import HealingExecutor
from app.proxy import app, configure_executor
from app.registry import SchemaRegistry, register_mock_tool


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_proxy_returns_upstream_body_and_current_schema() -> None:
    async def scenario() -> None:
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        registry = SchemaRegistry()
        await register_mock_tool(registry, "http://upstream")
        configure_executor(HealingExecutor(
            registry, transport=httpx.MockTransport(respond), audit_log=AuditLog(in_memory=True)
        ))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            result = await client.post(
                "/tools/lookup_user/call", json={"payload": {"user_id": "1"}}
            )
            schema = await client.get("/tools/lookup_user/schema")

        assert result.status_code == 200
        assert result.json() == {"ok": True}
        assert result.headers["x-healgate-outcome"] == "passed"
        assert schema.headers["x-healgate-schema-version"] == "1"
        assert schema.json()["required"] == ["user_id"]

    _run(scenario())


def test_proxy_halts_with_escalation_and_upstream_status() -> None:
    async def scenario() -> None:
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "denied"})

        registry = SchemaRegistry()
        await register_mock_tool(registry, "http://upstream")
        configure_executor(HealingExecutor(
            registry, transport=httpx.MockTransport(respond), audit_log=AuditLog(in_memory=True)
        ))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            result = await client.post(
                "/tools/lookup_user/call", json={"payload": {"user_id": "1"}}
            )

        assert result.status_code == 403
        assert result.headers["x-healgate-outcome"] == "halted"
        assert "Review the API rejection" in result.json()["suggested_action"]

    _run(scenario())
