"""One-replay executor with deterministic type, rename, header, and nesting rules."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.adapter import MappingError, apply_mapping
from app.audit import AuditLog, hash_payload
from app.diff_engine import diagnose_error
from app.gate import evaluate_mapping
from app.registry import SchemaRegistry, ToolSchema
from app.response_check import check_response
from app.sanitizer import sanitize_error
from app.schemas import AuditLogEntry, Escalation, GateVerdict, MappingOp
from app.triage import Action, action_for_status


@dataclass
class ExecutionResult:
    """Result summary suitable for a calling agent or a small demo."""

    status_code: int
    body: Any
    attempts: int
    healed: bool = False
    schema_version: int | None = None
    rollback_ref: str | None = None
    escalation: Escalation | None = None
    response_contract_ok: bool | None = None
    response_contract_issues: list[str] | None = None


class HealingExecutor:
    """Call a registered tool and attempt at most one safe validation repair."""

    def __init__(
        self,
        registry: SchemaRegistry,
        *,
        timeout_seconds: float = 5.0,
        header_allowlist: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.registry = registry
        self.timeout = httpx.Timeout(timeout_seconds)
        self.header_allowlist = header_allowlist or {}
        self.transport = transport
        self.audit_log = audit_log or AuditLog()

    async def execute(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Call, diagnose supported validation errors, gate, and replay once."""
        tool = await self.registry.get(tool_name)
        request_headers = dict(headers or {})
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            first = await client.post(tool.url, json=payload, headers=request_headers)
            if first.is_success:
                result = _success_result(first, tool, attempts=1, schema_version=tool.version)
                outcome = "response_contract_mismatch" if result.response_contract_ok is False else "succeeded"
                return await self._finish(
                    result, tool, payload, started, outcome=outcome,
                    evidence="; ".join(result.response_contract_issues or []),
                )

            action = action_for_status(first.status_code)
            if action not in (Action.HEAL, Action.HEAL_RULES_ONLY):
                result = ExecutionResult(
                    first.status_code, _body(first), 1,
                    escalation=_escalation(first.status_code, "status policy does not allow healing"),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted",
                    evidence=sanitize_error(_body(first)),
                )
            if action == Action.HEAL_RULES_ONLY:
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "no content-type repair rule is implemented yet"),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted", evidence=result.body,
                )

            diagnosis = diagnose_error(
                _body(first), payload, header_allowlist=self.header_allowlist
            )
            if diagnosis is None or not diagnosis.ops:
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "no supported deterministic repair was found"),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted", evidence=result.body,
                )

            verdict = evaluate_mapping(
                diagnosis.ops, payload,
                implicated_paths={op.target for op in diagnosis.ops if op.target},
                header_allowlist=self.header_allowlist,
            )
            if not verdict.allowed:
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "; ".join(verdict.reasons)),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted",
                    diagnosis=diagnosis, gate=verdict, evidence=diagnosis.evidence,
                )

            try:
                changed_payload, changed_headers = apply_mapping(
                    payload, request_headers, diagnosis.ops,
                    header_allowlist=self.header_allowlist,
                )
                _dry_run(changed_payload, changed_headers, diagnosis.ops, self.header_allowlist)
            except (MappingError, ValueError, TypeError) as exc:
                failed_dry_run = GateVerdict(allowed=False, reasons=[str(exc)])
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, f"dry run refused the repair: {exc}"),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted",
                    diagnosis=diagnosis, gate=failed_dry_run, evidence=diagnosis.evidence,
                )

            # Writes are replayed only after this mock explicitly rejects validation.
            replay = await client.post(tool.url, json=changed_payload, headers=changed_headers)
            if replay.is_success:
                new_schema = _schema_after_ops(tool.schema, diagnosis.ops)
                updated, rollback_ref = await self.registry.publish_after_success(
                    tool_name, new_schema, expected_version=tool.version
                )
                result = _success_result(
                    replay, tool, attempts=2, schema_version=updated.version,
                    healed=True, rollback_ref=rollback_ref,
                )
                outcome = "response_contract_mismatch" if result.response_contract_ok is False else "healed"
                return await self._finish(
                    result, tool, payload, started, outcome=outcome,
                    diagnosis=diagnosis, gate=verdict,
                    evidence="; ".join(result.response_contract_issues or []) or diagnosis.evidence,
                )

            # Failed replay never publishes the proposed schema.
            elapsed_ms = (time.perf_counter() - started) * 1000
            result = ExecutionResult(
                replay.status_code, sanitize_error(_body(replay)), 2,
                escalation=_escalation(
                    replay.status_code,
                    f"replay failed after {elapsed_ms:.1f} ms; the schema was not updated",
                ),
                schema_version=tool.version,
            )
            return await self._finish(
                result, tool, payload, started, outcome="replay_failed",
                diagnosis=diagnosis, gate=verdict, evidence=result.body,
            )

    async def _finish(
        self,
        result: ExecutionResult,
        tool: ToolSchema,
        payload: dict[str, Any],
        started: float,
        *,
        outcome: str,
        diagnosis: Diagnosis | None = None,
        gate: GateVerdict | None = None,
        evidence: str = "",
    ) -> ExecutionResult:
        """Append a payload-hashed audit entry before returning the outcome."""
        await self.audit_log.append(
            AuditLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                tool_name=tool.name,
                schema_version_before=tool.version,
                schema_version_after=result.schema_version,
                status_code=result.status_code,
                diagnosis_source=diagnosis.source if diagnosis else None,
                ops=diagnosis.ops if diagnosis else [],
                evidence=evidence,
                gate=gate if gate is not None else GateVerdict(allowed=True, evaluated=False),
                outcome=outcome,
                payload_hash_before=hash_payload(payload),
                rollback_ref=result.rollback_ref,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return result


def _dry_run(
    payload: dict[str, Any],
    headers: dict[str, str],
    ops: list[MappingOp],
    header_allowlist: dict[str, str],
) -> None:
    """Check the simple inferred target fields before a network replay."""
    for op in ops:
        if op.op == "cast" and op.target == "/user_id":
            value = payload.get("user_id")
            if op.cast_to == "int" and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError("/user_id is not an integer after the proposed cast")
        elif op.op == "rename" and op.source == "/user_id" and op.target == "/account_id":
            if "account_id" not in payload or "user_id" in payload:
                raise ValueError("rename did not produce only the expected /account_id field")
        elif op.op == "cast" and op.target == "/target/id" and op.cast_to == "int":
            target = payload.get("target")
            if not isinstance(target, dict) or isinstance(target.get("id"), bool) or not isinstance(target.get("id"), int):
                raise ValueError("/target/id is not an integer after the proposed cast")
        elif op.op == "add_header":
            if not op.header or headers.get(op.header) != header_allowlist.get(op.header):
                raise ValueError(f"header {op.header!r} has no configured value")


def _schema_after_ops(schema: dict[str, Any], ops: list[MappingOp]) -> dict[str, Any]:
    """Return a copied agent-facing schema reflecting successful simple ops."""
    updated = json.loads(json.dumps(schema))
    for op in ops:
        if op.op == "cast" and op.target == "/user_id" and op.cast_to:
            json_types = {"int": "integer", "str": "string", "float": "number", "bool": "boolean"}
            updated.setdefault("properties", {}).setdefault("user_id", {})["type"] = json_types[op.cast_to]
        elif op.op == "rename" and op.source == "/user_id" and op.target == "/account_id":
            properties = updated.setdefault("properties", {})
            properties["account_id"] = properties.pop("user_id", {"type": "string"})
            required = updated.setdefault("required", [])
            updated["required"] = ["account_id" if field == "user_id" else field for field in required]
        elif op.op == "move" and op.source == "/user_id" and op.target == "/target/id":
            properties = updated.setdefault("properties", {})
            value_schema = properties.pop("user_id", {"type": "string"})
            target_schema = properties.setdefault("target", {"type": "object", "properties": {}, "required": []})
            target_schema.setdefault("properties", {})["id"] = value_schema
            if "id" not in target_schema.setdefault("required", []):
                target_schema["required"].append("id")
            required = updated.setdefault("required", [])
            updated["required"] = [field for field in required if field != "user_id"]
            if "target" not in updated["required"]:
                updated["required"].append("target")
        elif op.op == "cast" and op.target == "/target/id" and op.cast_to:
            json_types = {"int": "integer", "str": "string", "float": "number", "bool": "boolean"}
            updated.setdefault("properties", {}).setdefault(
                "target", {"type": "object", "properties": {}, "required": []}
            ).setdefault("properties", {}).setdefault("id", {})["type"] = json_types[op.cast_to]
        elif op.op == "add_header" and op.header:
            required = updated.setdefault("required_headers", [])
            if op.header not in required:
                required.append(op.header)
    return updated


def _body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _success_result(
    response: httpx.Response,
    tool: ToolSchema,
    *,
    attempts: int,
    schema_version: int,
    healed: bool = False,
    rollback_ref: str | None = None,
) -> ExecutionResult:
    """Report response shape issues without retrying a successful request."""
    body = _body(response)
    issues = (
        check_response(body, tool.response_schema)
        if tool.response_schema is not None
        else None
    )
    return ExecutionResult(
        status_code=response.status_code,
        body=body,
        attempts=attempts,
        healed=healed,
        schema_version=schema_version,
        rollback_ref=rollback_ref,
        response_contract_ok=None if issues is None else not issues,
        response_contract_issues=issues,
    )


def _escalation(status_code: int, reason: str) -> Escalation:
    return Escalation(
        reason=reason,
        suggested_action=f"Review the API rejection (HTTP {status_code}) and update the call or tool contract.",
    )
