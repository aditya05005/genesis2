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
from app.diff_engine import (
    GeminiMappingDiagnoser,
    diagnose_error,
    extract_implicated_paths,
)
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
        llm_diagnoser: GeminiMappingDiagnoser | None = None,
    ) -> None:
        self.registry = registry
        self.timeout = httpx.Timeout(timeout_seconds)
        self.header_allowlist = header_allowlist or {}
        self.transport = transport
        self.audit_log = audit_log or AuditLog()
        self.llm_diagnoser = llm_diagnoser

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

            error_body = _body(first)
            diagnosis = diagnose_error(error_body, payload, header_allowlist=self.header_allowlist)
            tokens_used = 0
            llm_failure = False
            if diagnosis is None and self.llm_diagnoser is not None:
                try:
                    diagnosis, tokens_used = await self.llm_diagnoser.diagnose(
                        error_body, payload, header_allowlist=self.header_allowlist
                    )
                except Exception:
                    # Invalid JSON, SDK errors, and unavailable credentials all fail closed.
                    llm_failure = True
            if diagnosis is None or not diagnosis.ops:
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(
                        first.status_code,
                        "Gemini diagnosis failed; call halted" if llm_failure
                        else "no supported safe repair was found",
                    ),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted", evidence=result.body,
                    tokens_used=tokens_used,
                )

            verdict = evaluate_mapping(
                diagnosis.ops, payload,
                implicated_paths=extract_implicated_paths(error_body),
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
                    tokens_used=tokens_used,
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
                    tokens_used=tokens_used,
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
                    tokens_used=tokens_used,
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
                tokens_used=tokens_used,
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
        tokens_used: int = 0,
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
                tokens_used=tokens_used,
            )
        )
        return result


def _dry_run(
    payload: dict[str, Any],
    headers: dict[str, str],
    ops: list[MappingOp],
    header_allowlist: dict[str, str],
) -> None:
    """Check every transformed target and cast before a network replay."""
    for op in ops:
        if op.op == "add_header":
            if not op.header or headers.get(op.header) != header_allowlist.get(op.header):
                raise ValueError(f"header {op.header!r} has no configured value")
            continue
        found, value = _read_pointer(payload, op.target or "")
        if not found:
            raise ValueError(f"target {op.target!r} was not created")
        if op.op == "cast" and not _matches_cast(value, op.cast_to):
            raise ValueError(f"target {op.target!r} does not match cast type {op.cast_to!r}")
        if op.op in ("move", "rename") and op.source != op.target:
            source_found, _ = _read_pointer(payload, op.source or "")
            if source_found:
                raise ValueError(f"source {op.source!r} was not removed")


def _schema_after_ops(schema: dict[str, Any], ops: list[MappingOp]) -> dict[str, Any]:
    """Apply successful operation shapes to a copied agent-facing schema."""
    updated = json.loads(json.dumps(schema))
    for op in ops:
        if op.op in ("move", "rename") and op.source and op.target:
            value_schema = _schema_remove(updated, op.source) or {}
            _schema_set(updated, op.target, value_schema)
        elif op.op == "cast" and op.target and op.cast_to:
            value_schema = _schema_get(updated, op.target) or {}
            json_types = {"int": "integer", "str": "string", "float": "number", "bool": "boolean"}
            value_schema["type"] = json_types[op.cast_to]
            _schema_set(updated, op.target, value_schema)
        elif op.op == "wrap" and op.source and op.target:
            value_schema = _schema_get(updated, op.source) or {}
            _schema_set(updated, op.target, {
                "type": "object", "required": ["value"],
                "properties": {"value": value_schema},
            })
        elif op.op == "add_header" and op.header:
            required = updated.setdefault("required_headers", [])
            if op.header not in required:
                required.append(op.header)
    return updated


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _read_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
    current = document
    try:
        for part in _pointer_parts(pointer):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return False, None
    except ValueError:
        return False, None
    return True, current


def _matches_cast(value: Any, target: str | None) -> bool:
    return {
        "int": lambda: isinstance(value, int) and not isinstance(value, bool),
        "float": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "str": lambda: isinstance(value, str),
        "bool": lambda: isinstance(value, bool),
    }.get(target, lambda: False)()


def _schema_get(schema: dict[str, Any], pointer: str) -> dict[str, Any] | None:
    current = schema
    for part in _pointer_parts(pointer):
        current = current.get("properties", {}).get(part)
        if not isinstance(current, dict):
            return None
    return current


def _schema_remove(schema: dict[str, Any], pointer: str) -> dict[str, Any] | None:
    parts = _pointer_parts(pointer)
    if not parts:
        return None
    parent = _schema_parent(schema, parts[:-1], create=False)
    if parent is None:
        return None
    removed = parent.get("properties", {}).pop(parts[-1], None)
    if "required" in parent:
        parent["required"] = [field for field in parent["required"] if field != parts[-1]]
    return removed


def _schema_set(schema: dict[str, Any], pointer: str, value_schema: dict[str, Any]) -> None:
    parts = _pointer_parts(pointer)
    if not parts:
        raise ValueError("mapping the whole request schema is not supported")
    parent = _schema_parent(schema, parts[:-1], create=True)
    assert parent is not None
    parent.setdefault("properties", {})[parts[-1]] = value_schema
    required = parent.setdefault("required", [])
    if parts[-1] not in required:
        required.append(parts[-1])


def _schema_parent(
    schema: dict[str, Any], parts: list[str], *, create: bool
) -> dict[str, Any] | None:
    current = schema
    for part in parts:
        properties = current.setdefault("properties", {})
        child = properties.get(part)
        if child is None and create:
            child = {"type": "object", "properties": {}, "required": []}
            properties[part] = child
        if not isinstance(child, dict):
            return None
        if create:
            required = current.setdefault("required", [])
            if part not in required:
                required.append(part)
        current = child
    return current


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
