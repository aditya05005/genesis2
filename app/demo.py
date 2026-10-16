"""Run the type-change healing path against the mock service in-process."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.executor import HealingExecutor
from app.registry import SchemaRegistry, register_mock_tool
from mock_service.server import app as mock_app


async def run_demo() -> dict[str, Any]:
    """Exercise toggle, first rejection, safe cast, replay, and recorded intent."""
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-service", timeout=httpx.Timeout(5.0)
    ) as admin:
        configured = await admin.put(
            "/admin/mutation",
            json={"mutation": "type_change", "error_format": "fastapi"},
        )
        configured.raise_for_status()
        await admin.delete("/admin/records")

        registry = SchemaRegistry()
        await register_mock_tool(registry, "http://mock-service")
        executor = HealingExecutor(registry, transport=transport)
        result = await executor.execute("lookup_user", {"user_id": "123"})

        state_response = await admin.get("/admin/state")
        state_response.raise_for_status()
        state = state_response.json()

    if result.status_code != 200 or not result.healed or result.attempts != 2:
        raise RuntimeError(f"type-change demo did not heal successfully: {result!r}")
    if len(state["records"]) != 1:
        raise RuntimeError("mock service did not record exactly one accepted replay")
    record = state["records"][0]
    if record["received"].get("user_id") != 123:
        raise RuntimeError("replayed call did not contain an integer user_id")
    if record["ground_truth"] != {"user_id": "123"}:
        raise RuntimeError("mock service recorded unexpected ground-truth intent")

    return {
        "mutation": state["mutation"],
        "status_code": result.status_code,
        "attempts": result.attempts,
        "healed": result.healed,
        "schema_version": result.schema_version,
        "rollback_ref": result.rollback_ref,
        "received": record["received"],
        "ground_truth": record["ground_truth"],
    }


def main() -> None:
    """Print a compact result; raise a nonzero exit on a failed demo assertion."""
    print(json.dumps(asyncio.run(run_demo()), indent=2))


if __name__ == "__main__":
    main()
