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
from mock_service.server import app as mock_app


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
            result = await HealingExecutor(
                registry, transport=transport, header_allowlist=header_allowlist,
                audit_log=audit_log,
                llm_diagnoser=llm_diagnoser if mutation == "profile_nesting" else None,
            ).execute("lookup_user", payload
            )
            state_response = await admin.get("/admin/state")
            state_response.raise_for_status()
            state = state_response.json()

            response_mutation = mutation == "response_nested"
            expected_attempts = 1 if response_mutation else 2
            expected_healed = not response_mutation
            if (
                result.status_code != 200
                or result.healed != expected_healed
                or result.attempts != expected_attempts
            ):
                raise RuntimeError(f"{mutation} demo did not heal successfully: {result!r}")
            expected_response_ok = not response_mutation
            if result.response_contract_ok is not expected_response_ok:
                raise RuntimeError(
                    f"{mutation} had unexpected response-contract status: "
                    f"{result.response_contract_ok!r}"
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
                "status_code": result.status_code,
                "attempts": result.attempts,
                "healed": result.healed,
                "response_contract_ok": result.response_contract_ok,
                "response_contract_issues": result.response_contract_issues,
                "schema_version": current_schema.version,
                "schema": current_schema.schema,
                "rollback_ref": result.rollback_ref,
                "received": record["received"],
                "ground_truth": record["ground_truth"],
            })

    audit_entries = await audit_log.entries()
    return {
        "cases": cases,
        "audit": [entry.model_dump(mode="json") for entry in audit_entries],
    }


def main() -> None:
    """Print demo results; --gemini adds a real, opt-in model call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemini", action="store_true",
        help="run the profile-nesting case with the configured Gemini model",
    )
    args = parser.parse_args()
    diagnoser = GeminiMappingDiagnoser.from_env() if args.gemini else None
    print(json.dumps(asyncio.run(run_demo(llm_diagnoser=diagnoser)), indent=2))


if __name__ == "__main__":
    main()
