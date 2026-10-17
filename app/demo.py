"""Run the supported healing mutations against the mock service."""

from __future__ import annotations

import asyncio
import argparse
import json
from typing import Any

import httpx

from app.audit import AuditLog
from app.diff_engine import GeminiMappingDiagnoser
from app.executor import HealingExecutor
from app.registry import SchemaRegistry, register_mock_tool
from app.proxy import app as proxy_app, configure_executor
from mock_service.server import app as mock_app


async def _proxy_call(
    registry: SchemaRegistry, upstream_transport: httpx.AsyncBaseTransport,
    audit_log: AuditLog, payload: dict[str, Any], headers: dict[str, str], *,
    llm_diagnoser: GeminiMappingDiagnoser | None = None,
) -> tuple[httpx.Response, Any]:
    """Invoke a configured tool through the same HTTP proxy agents use."""
    executor = HealingExecutor(
        registry, transport=upstream_transport, header_allowlist=headers,
        audit_log=audit_log, llm_diagnoser=llm_diagnoser,
    )
    configure_executor(executor)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://healgate-proxy"
    ) as proxy_client:
        response = await proxy_client.post(
            "/tools/lookup_user/call", json={"payload": payload, "headers": {}}
        )
    audit_entries = await audit_log.entries()
    return response, audit_entries[-1] if audit_entries else None


async def run_custom_demo(
    mutation: str, error_format: str, *,
    llm_diagnoser: GeminiMappingDiagnoser | None = None,
) -> dict[str, Any]:
    """Run one user-selected mock contract/error-format combination."""
    transport = httpx.ASGITransport(app=mock_app)
    audit_log = AuditLog()
    payload = {"user_id": "1.7" if mutation == "lossy_integer" else "123"}
    header_allowlist = (
        {"X-Api-Version": "2026-01-01"}
        if mutation in {"required_header", "compound"} else {}
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-service") as admin:
        response = await admin.put(
            "/admin/mutation", json={"mutation": mutation, "error_format": error_format}
        )
        response.raise_for_status()
        await admin.delete("/admin/records")

        registry = SchemaRegistry()
        await register_mock_tool(registry, "http://mock-service")
        result, audit_entry = await _proxy_call(
            registry, transport, audit_log, payload, header_allowlist,
            llm_diagnoser=llm_diagnoser if mutation == "profile_nesting" else None,
        )
        state_response = await admin.get("/admin/state")
        state_response.raise_for_status()
        state = state_response.json()

    audit_entries = await audit_log.entries()
    contract_ok = (
        False if audit_entry and audit_entry.outcome == "response_contract_mismatch"
        else True if result.is_success and audit_entry and audit_entry.outcome in {"healed", "succeeded"}
        else None
    )
    return {
        "cases": [{
            "mutation": mutation, "error_format": error_format,
            "status_code": result.status_code,
            "attempts": 2 if audit_entry and audit_entry.outcome in {"healed", "failed_rolled_back", "replay_failed"} else 1,
            "healed": result.headers.get("x-healgate-outcome") == "healed",
            "response_contract_ok": contract_ok,
            "response_contract_issues": [],
            "schema_version": audit_entry.schema_version_after if audit_entry else None,
            "rollback_ref": audit_entry.rollback_ref if audit_entry else None,
            "escalation": result.json() if result.headers.get("x-healgate-outcome") == "halted" else None,
            "received": state["records"][-1]["received"] if state["records"] else None,
        }],
        "audit": [entry.model_dump(mode="json") for entry in audit_entries],
    }


async def run_demo(
    *, llm_diagnoser: GeminiMappingDiagnoser | None = None,
) -> dict[str, Any]:
    """Exercise deterministic repairs and, when supplied, one LLM-only repair."""
    transport = httpx.ASGITransport(app=mock_app)
    cases = []
    audit_log = AuditLog()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-service", timeout=httpx.Timeout(5.0)
    ) as admin:
        scenarios = [
            ("type_change", {"user_id": "123"}, {"user_id": 123}, {}),
            ("rename", {"user_id": "123"}, {"account_id": "123"}, {}),
            ("nesting", {"user_id": "123"}, {"target": {"id": 123}}, {}),
            (
                "required_header",
                {"user_id": "123"},
                {"user_id": "123"},
                {"X-Api-Version": "2026-01-01"},
            ),
            (
                "compound",
                {"user_id": "123"},
                {"account_id": "123"},
                {"X-Api-Version": "2026-01-01"},
            ),
            ("response_nested", {"user_id": "123"}, {"user_id": "123"}, {}),
        ]
        if llm_diagnoser is not None:
            scenarios.append((
                "profile_nesting", {"user_id": "123"}, {"profile": {"id": "123"}}, {}
            ))

        for mutation, payload, expected_received, header_allowlist in scenarios:
            configured = await admin.put(
                "/admin/mutation",
                json={"mutation": mutation, "error_format": "fastapi"},
            )
            configured.raise_for_status()
            await admin.delete("/admin/records")

            registry = SchemaRegistry()
            await register_mock_tool(registry, "http://mock-service")
            proxy_result, audit_entry = await _proxy_call(
                registry, transport, audit_log, payload, header_allowlist,
                llm_diagnoser=llm_diagnoser if mutation == "profile_nesting" else None,
            )
            state_response = await admin.get("/admin/state")
            state_response.raise_for_status()
            state = state_response.json()

            response_mutation = mutation == "response_nested"
            expected_attempts = 1 if response_mutation else 2
            expected_healed = not response_mutation
            healed = proxy_result.headers.get("x-healgate-outcome") == "healed"
            attempts = 2 if audit_entry and audit_entry.outcome in {
                "healed", "failed_rolled_back", "replay_failed"
            } else 1
            response_contract_ok = not (
                audit_entry and audit_entry.outcome == "response_contract_mismatch"
            )
            if (
                proxy_result.status_code != 200
                or healed != expected_healed
                or attempts != expected_attempts
            ):
                raise RuntimeError(f"{mutation} demo did not heal successfully: {proxy_result!r}")
            expected_response_ok = not response_mutation
            if response_contract_ok is not expected_response_ok:
                raise RuntimeError(
                    f"{mutation} had unexpected response-contract status: "
                    f"{response_contract_ok!r}"
                )
            if len(state["records"]) != 1:
                raise RuntimeError(f"{mutation} did not record exactly one accepted replay")
            record = state["records"][0]
            if record["received"] != expected_received:
                raise RuntimeError(f"{mutation} replay sent an unexpected payload: {record['received']!r}")
            if record["ground_truth"] != {"user_id": "123"}:
                raise RuntimeError(f"{mutation} recorded unexpected ground-truth intent")
            expected_headers = (
                {"X-Api-Version": "2026-01-01"}
                if mutation in ("required_header", "compound") else {}
            )
            if record["received_headers"] != expected_headers:
                raise RuntimeError(f"{mutation} replay sent unexpected headers: {record['received_headers']!r}")

            current_schema = await registry.get("lookup_user")
            if mutation == "nesting" and current_schema.schema.get("required") != ["target"]:
                raise RuntimeError("nested schema did not mark target as required")
            cases.append({
                "mutation": mutation,
                "status_code": proxy_result.status_code,
                "attempts": attempts,
                "healed": healed,
                "response_contract_ok": response_contract_ok,
                "response_contract_issues": [],
                "schema_version": current_schema.version,
                "schema": current_schema.schema,
                "rollback_ref": audit_entry.rollback_ref if audit_entry else None,
                "received": record["received"],
                "ground_truth": record["ground_truth"],
            })

    audit_entries = await audit_log.entries()
    return {
        "cases": cases,
        "audit": [entry.model_dump(mode="json") for entry in audit_entries],
    }


def main() -> None:
    """Print demo results, using Gemini automatically for Tier 2 diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    diagnoser = GeminiMappingDiagnoser.from_env()
    print(json.dumps(asyncio.run(run_demo(llm_diagnoser=diagnoser)), indent=2))


if __name__ == "__main__":
    main()
