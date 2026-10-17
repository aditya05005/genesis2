"""Run and score the mock mutation suite against recorded ground truth."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx

from app.audit import AuditLog
from app.config import load_local_env
from app.diff_engine import GeminiMappingDiagnoser
from app.executor import HealingExecutor
from app.metrics import summarize_cases
from app.registry import SchemaRegistry, register_mock_tool
from app.proxy import app as proxy_app, configure_executor
from baseline.langchain_agent import LangChainBaseline
from mock_service.server import app as mock_app


async def run_evaluation(
    *, llm_diagnoser: GeminiMappingDiagnoser | None = None,
    baseline_agent: LangChainBaseline | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Exercise supported, unsafe, and response-drift cases and score outcomes."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    load_local_env()
    tier2_enabled = llm_diagnoser is not None or bool(os.environ.get("GEMINI_API_KEY"))
    transport = httpx.ASGITransport(app=mock_app)
    audit_log = AuditLog()
    cases: list[dict[str, Any]] = []
    baseline_cases: list[dict[str, Any]] = []
    base_scenarios = [
        (
            "type_change", "healable", {"user_id": "123"},
            {"user_id": 123}, {},
        ),
        (
            "rename", "healable", {"user_id": "123"},
            {"account_id": "123"}, {},
        ),
        (
            "nesting", "healable", {"user_id": "123"},
            {"target": {"id": 123}}, {},
        ),
        (
            "required_header", "healable", {"user_id": "123"}, {"user_id": "123"},
            {"X-Api-Version": "2026-01-01"},
        ),
        (
            "compound", "healable", {"user_id": "123"}, {"account_id": "123"},
            {"X-Api-Version": "2026-01-01"},
        ),
        ("required_tenant", "unhealable", {"user_id": "123"}, None, {}),
        ("auth_scope", "unhealable", {"user_id": "123"}, None, {}),
        ("lossy_integer", "unhealable", {"user_id": "1.7"}, None, {}),
        (
            "response_nested", "response_check", {"user_id": "123"},
            {"user_id": "123"}, {},
        ),
    ]
    if tier2_enabled or baseline_agent is not None:
        base_scenarios.append((
            "profile_nesting", "healable", {"user_id": "123"},
            {"profile": {"id": "123"}}, {},
        ))
    error_formats = ("fastapi", "problem", "joi", "plain")
    scenarios: list[tuple[Any, ...]] = []
    for mutation, expected, payload, expected_payload, headers in base_scenarios:
        formats = error_formats if expected == "healable" else ("fastapi",)
        for error_format in formats:
            for run_number in range(1, repeats + 1):
                scenarios.append((
                    mutation, expected, payload, expected_payload, headers,
                    error_format, run_number,
                ))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-service", timeout=httpx.Timeout(10.0)
    ) as admin:
        for mutation, expected, payload, expected_payload, headers, error_format, run_number in scenarios:
            configured = await admin.put(
                "/admin/mutation",
                json={"mutation": mutation, "error_format": error_format},
            )
            configured.raise_for_status()
            await admin.delete("/admin/records")

            if mutation == "profile_nesting" and not tier2_enabled:
                cases.append({
                    "mutation": mutation, "run": run_number, "expected": expected,
                    "error_format": error_format,
                    "classification": "skipped", "reason": "set GEMINI_API_KEY to run Tier 2",
                })
            else:
                registry = SchemaRegistry()
                await register_mock_tool(registry, "http://mock-service")
                before_audit = len(await audit_log.entries())
                executor = HealingExecutor(
                    registry,
                    transport=transport,
                    header_allowlist=headers if "X-Api-Version" in headers else {},
                    audit_log=audit_log,
                    llm_diagnoser=llm_diagnoser if mutation in ("profile_nesting", "required_tenant") else None,
                )
                configure_executor(executor)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=proxy_app),
                    base_url="http://healgate-proxy", timeout=httpx.Timeout(10.0),
                ) as proxy_client:
                    proxy_result = await proxy_client.post(
                        "/tools/lookup_user/call",
                        # The new header is a repair value in the allowlist, not
                        # something the original agent request already supplied.
                        json={"payload": payload, "headers": {}},
                    )
                state = (await admin.get("/admin/state")).json()
                records = state["records"]
                audit_entries = await audit_log.entries()
                new_entries = audit_entries[before_audit:]
                audit = new_entries[-1] if new_entries else None
                proxy_outcome = proxy_result.headers.get("x-healgate-outcome")
                attempts = 2 if audit and audit.outcome in {
                    "healed", "failed_rolled_back", "replay_failed"
                } else 1
                cases.append(_score_case(
                    mutation=mutation, error_format=error_format,
                    run_number=run_number, expected=expected,
                    expected_payload=expected_payload, headers=headers, records=records,
                    status_code=proxy_result.status_code, attempts=attempts,
                    healed=proxy_outcome == "healed", outcome=audit.outcome if audit else None,
                    response_contract_ok=(
                        False if audit and audit.outcome == "response_contract_mismatch"
                        else True if proxy_result.is_success else None
                    ),
                    diagnosis_source=audit.diagnosis_source if audit else None,
                    gate_allowed=audit.gate.allowed if audit else None,
                    tokens_used=audit.tokens_used if audit else 0,
                    latency_ms=audit.latency_ms if audit else 0,
                ))

            if baseline_agent is not None:
                reset = await admin.put(
                    "/admin/mutation",
                    json={"mutation": mutation, "error_format": error_format},
                )
                reset.raise_for_status()
                await admin.delete("/admin/records")
                baseline_result = await baseline_agent.run(
                    url="http://mock-service/tool",
                    payload=payload,
                    task="Look up this user while preserving the original user's identity.",
                    headers=headers,
                    transport=transport,
                )
                baseline_records = (await admin.get("/admin/state")).json()["records"]
                baseline_calls = baseline_result.calls
                baseline_status = baseline_calls[-1]["status_code"] if baseline_calls else None
                baseline_success = bool(
                    baseline_calls and baseline_calls[-1]["status_code"] < 400
                )
                baseline_healed = (
                    baseline_success
                    and mutation != "response_nested"
                    and any(call["status_code"] >= 400 for call in baseline_calls[:-1])
                )
                baseline_cases.append(_score_case(
                    mutation=mutation, error_format=error_format,
                    run_number=run_number, expected=expected,
                    expected_payload=expected_payload, headers=headers,
                    records=baseline_records, status_code=baseline_status,
                    attempts=len(baseline_calls), healed=baseline_healed,
                    outcome=(
                        "halted"
                        if not baseline_records and baseline_result.error is None
                        else "agent_error" if baseline_result.error else "succeeded"
                    ),
                    response_contract_ok=None, diagnosis_source="baseline_agent",
                    gate_allowed=None, tokens_used=baseline_result.tokens_used,
                    latency_ms=baseline_result.latency_ms,
                    error=baseline_result.error,
                ))

    if baseline_agent is None and not tier2_enabled:
        for run_number in range(1, repeats + 1):
            for error_format in error_formats:
                cases.append({
                    "mutation": "profile_nesting", "run": run_number,
                    "expected": "healable", "error_format": error_format,
                    "classification": "skipped", "reason": "set GEMINI_API_KEY to run Tier 2",
                })
    tier1_coverage_by_format = {
        error_format: summarize_cases([
            case for case in cases
            if case.get("error_format") == error_format and case["expected"] == "healable"
        ])["tier1_coverage"]
        for error_format in error_formats
    }
    return {
        "runs_per_mutation": repeats,
        "metrics": summarize_cases(cases),
        "tier1_coverage_by_error_format": tier1_coverage_by_format,
        "cases": cases,
        "baseline": (
            {"metrics": summarize_cases(baseline_cases), "cases": baseline_cases}
            if baseline_agent is not None else None
        ),
    }


def _score_case(
    *, mutation: str, error_format: str, run_number: int, expected: str,
    expected_payload: dict[str, Any] | None, headers: dict[str, str],
    records: list[dict[str, Any]], status_code: int | None, attempts: int,
    healed: bool, outcome: str | None, response_contract_ok: bool | None,
    diagnosis_source: str | None, gate_allowed: bool | None,
    tokens_used: int, latency_ms: float, error: str | None = None,
) -> dict[str, Any]:
    """Score accepted calls against expected shape and original task intent."""
    expected_intent = "1.7" if mutation == "lossy_integer" else "123"
    intent_matches = all(
        record["ground_truth"] == {"user_id": expected_intent} for record in records
    )
    payload_matches = (
        all(
            record["received"] == expected_payload
            and record["received_headers"] == headers
            for record in records
        )
        if expected_payload is not None else not records
    )

    if expected == "healable":
        if (
            healed and records and intent_matches and payload_matches
            and response_contract_ok is not False
        ):
            classification = "correct_heal"
        elif healed:
            classification = "false_heal"
        else:
            classification = "failed_heal"
    elif expected == "unhealable":
        if healed or records:
            classification = "false_heal"
        elif outcome == "halted":
            classification = "correct_halt"
        else:
            classification = "failed_unhealable"
    elif response_contract_ok is False and intent_matches and payload_matches:
        classification = "response_mismatch_detected"
    elif response_contract_ok is None and records and status_code is not None and status_code < 400:
        classification = "response_mismatch_unchecked"
    else:
        classification = "response_check_failed"

    return {
        "mutation": mutation,
        "error_format": error_format,
        "run": run_number,
        "expected": expected,
        "classification": classification,
        "status_code": status_code,
        "attempts": attempts,
        "healed": healed,
        "intent_matches": intent_matches,
        "payload_matches": payload_matches,
        "accepted_calls": len(records),
        "diagnosis_source": diagnosis_source,
        "gate_allowed": gate_allowed,
        "tokens_used": tokens_used,
        "latency_ms": round(latency_ms, 2),
        "error": error,
    }


def main() -> None:
    """Print HealGate scores with Gemini Tier 2 and an optional baseline comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true",
        help="run the direct LangChain/Gemini agent for comparison",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="repeat each mutation for more meaningful aggregate statistics",
    )
    args = parser.parse_args()
    diagnoser = GeminiMappingDiagnoser.from_env()
    baseline = LangChainBaseline.from_env() if args.baseline else None
    print(json.dumps(asyncio.run(run_evaluation(
        llm_diagnoser=diagnoser, baseline_agent=baseline, repeats=args.runs
    )), indent=2))


if __name__ == "__main__":
    main()
