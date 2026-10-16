"""Run and score the mock mutation suite against recorded ground truth."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx

from app.audit import AuditLog
from app.diff_engine import GeminiMappingDiagnoser
from app.executor import HealingExecutor
from app.metrics import summarize_cases
from app.registry import SchemaRegistry, register_mock_tool
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
    transport = httpx.ASGITransport(app=mock_app)
    audit_log = AuditLog()
    cases: list[dict[str, Any]] = []
    baseline_cases: list[dict[str, Any]] = []
    scenarios = [
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
    if llm_diagnoser is not None or baseline_agent is not None:
        scenarios.append((
            "profile_nesting", "healable", {"user_id": "123"},
            {"profile": {"id": "123"}}, {},
        ))
    scenarios = [
        (*scenario, run_number)
        for scenario in scenarios
        for run_number in range(1, repeats + 1)
    ]

    async with httpx.AsyncClient(
        transport=transport, base_url="http://mock-service", timeout=httpx.Timeout(10.0)
    ) as admin:
        for mutation, expected, payload, expected_payload, headers, run_number in scenarios:
            configured = await admin.put(
                "/admin/mutation",
                json={"mutation": mutation, "error_format": "fastapi"},
            )
            configured.raise_for_status()
            await admin.delete("/admin/records")

            if mutation == "profile_nesting" and llm_diagnoser is None:
                cases.append({
                    "mutation": mutation, "run": run_number, "expected": expected,
                    "classification": "skipped", "reason": "requires --gemini",
                })
            else:
                registry = SchemaRegistry()
                await register_mock_tool(registry, "http://mock-service")
                before_audit = len(await audit_log.entries())
                result = await HealingExecutor(
                    registry,
                    transport=transport,
                    header_allowlist=headers if "X-Api-Version" in headers else {},
                    audit_log=audit_log,
                    llm_diagnoser=llm_diagnoser if mutation in ("profile_nesting", "required_tenant") else None,
                ).execute("lookup_user", payload)
                state = (await admin.get("/admin/state")).json()
                records = state["records"]
                audit_entries = await audit_log.entries()
                new_entries = audit_entries[before_audit:]
                audit = new_entries[-1] if new_entries else None
                cases.append(_score_case(
                    mutation=mutation, run_number=run_number, expected=expected,
                    expected_payload=expected_payload, headers=headers, records=records,
                    status_code=result.status_code, attempts=result.attempts,
                    healed=result.healed, outcome=audit.outcome if audit else None,
                    response_contract_ok=result.response_contract_ok,
                    diagnosis_source=audit.diagnosis_source if audit else None,
                    gate_allowed=audit.gate.allowed if audit else None,
                    tokens_used=audit.tokens_used if audit else 0,
                    latency_ms=audit.latency_ms if audit else 0,
                ))

            if baseline_agent is not None:
                reset = await admin.put(
                    "/admin/mutation",
                    json={"mutation": mutation, "error_format": "fastapi"},
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
                    mutation=mutation, run_number=run_number, expected=expected,
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

    if baseline_agent is None and llm_diagnoser is None:
        for run_number in range(1, repeats + 1):
            cases.append({
                "mutation": "profile_nesting", "run": run_number,
                "expected": "healable", "classification": "skipped",
                "reason": "requires --gemini",
            })
    return {
        "runs_per_mutation": repeats,
        "metrics": summarize_cases(cases),
        "cases": cases,
        "baseline": (
            {"metrics": summarize_cases(baseline_cases), "cases": baseline_cases}
            if baseline_agent is not None else None
        ),
    }


def _score_case(
    *, mutation: str, run_number: int, expected: str,
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
    """Print HealGate scores and an optional LangChain baseline comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", action="store_true", help="include the Gemini-only mutation")
    parser.add_argument(
        "--baseline", action="store_true",
        help="run the direct LangChain/Gemini agent for comparison",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="repeat each mutation for more meaningful aggregate statistics",
    )
    args = parser.parse_args()
    diagnoser = GeminiMappingDiagnoser.from_env() if args.gemini else None
    baseline = LangChainBaseline.from_env() if args.baseline else None
    print(json.dumps(asyncio.run(run_evaluation(
        llm_diagnoser=diagnoser, baseline_agent=baseline, repeats=args.runs
    )), indent=2))


if __name__ == "__main__":
    main()
