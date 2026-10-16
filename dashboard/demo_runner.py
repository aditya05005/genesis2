"""Run real baseline and HealGate calls against the in-process mock API."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.audit import AuditLog
from app.diff_engine import GeminiMappingDiagnoser
from app.executor import HealingExecutor
from app.registry import SchemaRegistry, register_mock_tool
from mock_service.server import app as mock_app

BASE_URL = "http://healgate-mock"


async def _admin(client: httpx.AsyncClient, mutation: str) -> None:
    response = await client.put(
        "/admin/mutation", json={"mutation": mutation, "error_format": "fastapi"}
    )
    response.raise_for_status()
    response = await client.delete("/admin/records")
    response.raise_for_status()


async def _state(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/admin/state")
    response.raise_for_status()
    return response.json()


async def _run_baseline(
    mutation: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=5.0) as client:
        await _admin(client, mutation)
        response = await client.post("/tool", json=payload, headers=headers)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return {
            "mode": "baseline", "mutation": mutation,
            "request": payload, "headers": headers, "status_code": response.status_code,
            "body": body, "accepted_calls": (await _state(client)).get("records", []),
        }


async def _run_healgate(
    mutation: str, payload: dict[str, Any], headers: dict[str, str], use_gemini: bool
) -> dict[str, Any]:
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=5.0) as client:
        await _admin(client, mutation)
        registry = SchemaRegistry()
        await register_mock_tool(registry, BASE_URL)
        audit_log = AuditLog()
        diagnoser = GeminiMappingDiagnoser.from_env() if use_gemini else None
        result = await HealingExecutor(
            registry, transport=transport,
            header_allowlist={"X-Api-Version": "2026-01-01"},
            audit_log=audit_log, llm_diagnoser=diagnoser,
        ).execute("lookup_user", payload, headers=headers)
        tool = await registry.get("lookup_user")
        audit_entries = await audit_log.entries()
        audit = audit_entries[-1] if audit_entries else None
        return {
            "mode": "healgate", "mutation": mutation,
            "status_code": result.status_code, "body": result.body,
            "attempts": result.attempts, "healed": result.healed,
            "original_payload": result.original_payload or payload,
            "original_headers": headers,
            "initial_status_code": result.initial_status_code,
            "initial_error": result.initial_error,
            "diagnosis": result.diagnosis.model_dump() if result.diagnosis else None,
            "gate": result.gate.model_dump() if result.gate else None,
            "transformed_payload": result.transformed_payload,
            "transformed_headers": result.transformed_headers,
            "response_contract_ok": result.response_contract_ok,
            "response_contract_issues": result.response_contract_issues,
            "tokens_used": audit.tokens_used if audit else 0,
            "audit": audit.model_dump(mode="json") if audit else None,
            "schema_version": tool.version, "rollback_ref": result.rollback_ref,
            "escalation": result.escalation.model_dump() if result.escalation else None,
            "accepted_calls": (await _state(client)).get("records", []),
        }


def run_baseline(
    mutation: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run one unmodified request against the real mock endpoint."""
    return asyncio.run(_run_baseline(mutation, payload, headers or {}))


def run_healgate(
    mutation: str, payload: dict[str, Any], headers: dict[str, str] | None = None,
    *, use_gemini: bool = False,
) -> dict[str, Any]:
    """Run the configured HealingExecutor against the real mock endpoint."""
    return asyncio.run(_run_healgate(mutation, payload, headers or {}, use_gemini))


def apply_scenario(mutation: str) -> dict[str, Any]:
    """Change the in-process mock contract without requiring a second server."""
    async def _apply() -> dict[str, Any]:
        transport = httpx.ASGITransport(app=mock_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=5.0) as client:
            response = await client.put(
                "/admin/mutation", json={"mutation": mutation, "error_format": "fastapi"}
            )
            response.raise_for_status()
            return await _state(client)
    return asyncio.run(_apply())
