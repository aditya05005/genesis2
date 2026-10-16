"""Run type-change and field-rename healing paths against the mock service."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.executor import HealingExecutor
from app.registry import SchemaRegistry, register_mock_tool
from mock_service.server import app as mock_app


async def run_demo() -> dict[str, Any]:
    """Exercise two validation repairs and check each mock record's intent."""
    transport = httpx.ASGITransport(app=mock_app)
    cases = []
    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-service", timeout=httpx.Timeout(5.0)
    ) as admin:
        for mutation, payload, expected_received, header_allowlist in (
            ("type_change", {"user_id": "123"}, {"user_id": 123}, {}),
            ("rename", {"user_id": "123"}, {"account_id": "123"}, {}),
            (
                "required_header",
                {"user_id": "123"},
                {"user_id": "123"},
                {"X-Api-Version": "2026-01-01"},
            ),
        ):
            configured = await admin.put(
                "/admin/mutation",
                json={"mutation": mutation, "error_format": "fastapi"},
            )
            configured.raise_for_status()
            await admin.delete("/admin/records")

            registry = SchemaRegistry()
            await register_mock_tool(registry, "http://mock-service")
            result = await HealingExecutor(
                registry, transport=transport, header_allowlist=header_allowlist
            ).execute("lookup_user", payload
            )
            state_response = await admin.get("/admin/state")
            state_response.raise_for_status()
            state = state_response.json()

            if result.status_code != 200 or not result.healed or result.attempts != 2:
                raise RuntimeError(f"{mutation} demo did not heal successfully: {result!r}")
            if len(state["records"]) != 1:
                raise RuntimeError(f"{mutation} did not record exactly one accepted replay")
            record = state["records"][0]
            if record["received"] != expected_received:
                raise RuntimeError(f"{mutation} replay sent an unexpected payload: {record['received']!r}")
            if record["ground_truth"] != {"user_id": "123"}:
                raise RuntimeError(f"{mutation} recorded unexpected ground-truth intent")
            expected_headers = {"X-Api-Version": "2026-01-01"} if mutation == "required_header" else {}
            if record["received_headers"] != expected_headers:
                raise RuntimeError(f"{mutation} replay sent unexpected headers: {record['received_headers']!r}")

            current_schema = await registry.get("lookup_user")
            cases.append({
                "mutation": mutation,
                "status_code": result.status_code,
                "attempts": result.attempts,
                "healed": result.healed,
                "schema_version": current_schema.version,
                "schema": current_schema.schema,
                "rollback_ref": result.rollback_ref,
                "received": record["received"],
                "ground_truth": record["ground_truth"],
            })

    return {"cases": cases}


def main() -> None:
    """Print a compact result; raise a nonzero exit on a failed demo assertion."""
    print(json.dumps(asyncio.run(run_demo()), indent=2))


if __name__ == "__main__":
    main()
