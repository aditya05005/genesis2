"""One-replay executor with deterministic type, rename, header, and nesting rules."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.adapter import MappingError, apply_mapping
from app.audit import AuditLog, hash_payload
from app.config import load_local_env
from app.diff_engine import (
    GeminiMappingDiagnoser,
    diagnose_error,
    extract_implicated_paths,
)
from app.gate import evaluate_mapping
from app.registry import SchemaRegistry, ToolSchema, schema_patch_for_ops
from app.response_check import check_response, diagnose_response
from app.sanitizer import sanitize_error, sanitize_error_details
from app.schemas import AuditLogEntry, Diagnosis, Escalation, GateVerdict, MappingOp
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
        timeout_seconds: float | None = None,
        timeout: httpx.Timeout | None = None,
        header_allowlist: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        audit_log: AuditLog | None = None,
        llm_diagnoser: GeminiMappingDiagnoser | None = None,
    ) -> None:
        load_local_env()
        self.registry = registry
        self.timeout = timeout or httpx.Timeout(
            connect=float(os.environ.get("HEALGATE_TIMEOUT_CONNECT", timeout_seconds or 5.0)),
            read=float(os.environ.get("HEALGATE_TIMEOUT_READ", timeout_seconds or 5.0)),
            write=float(os.environ.get("HEALGATE_TIMEOUT_WRITE", timeout_seconds or 5.0)),
            pool=float(os.environ.get("HEALGATE_TIMEOUT_POOL", timeout_seconds or 5.0)),
        )
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
                body = _body(first)
                result, diagnosis, gate, tokens_used = await self._heal_response(
                    body, tool, payload
                )
                result.status_code = first.status_code
                return await self._finish(
                    result, tool, payload, started,
                    outcome="healed" if result.healed else (
                        "response_contract_mismatch" if result.response_contract_ok is False else "succeeded"
                    ),
                    diagnosis=diagnosis, gate=gate,
                    evidence="; ".join(result.response_contract_issues or []) or (diagnosis.evidence if diagnosis else ""),
                    tokens_used=tokens_used,
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
            sanitized_error = sanitize_error_details(error_body)
            if sanitized_error.injection_flagged:
                result = ExecutionResult(
                    first.status_code, sanitized_error.text, 1,
                    escalation=_escalation(
                        first.status_code,
                        "prompt-injection-like instructions were detected in the API error; diagnosis was skipped",
                        missing_fields=_missing_fields(error_body, payload),
                    ),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted",
                    evidence=sanitized_error.text,
                )
            signature = hashlib.sha256(
                f"{first.status_code}:{sanitized_error.text}".encode("utf-8")
            ).hexdigest()
            diagnosis, tokens_used, llm_failure = await self.registry.singleflight(
                tool_name, signature,
                lambda: self._diagnose_request(error_body, payload),
            )
            if diagnosis is None or not diagnosis.ops:
                result = ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(
                        first.status_code,
                        "Gemini diagnosis failed; call halted" if llm_failure
                        else "no supported safe repair was found",
                        missing_fields=_missing_fields(error_body, payload),
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
                    escalation=_escalation(
                        first.status_code, "; ".join(verdict.reasons),
                        missing_fields=_missing_fields(error_body, payload),
                    ),
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
                    escalation=_escalation(
                        first.status_code, f"dry run refused the repair: {exc}",
                        missing_fields=_missing_fields(error_body, payload),
                    ),
                    schema_version=tool.version,
                )
                return await self._finish(
                    result, tool, payload, started, outcome="halted",
                    diagnosis=diagnosis, gate=failed_dry_run, evidence=diagnosis.evidence,
                    tokens_used=tokens_used,
                )

            # Stage the agent-facing contract update so failed replays can roll it back.
            schema_patch = schema_patch_for_ops(tool.schema, diagnosis.ops)
            staged, rollback_ref = await self.registry.publish_patch(
                tool_name, schema_patch, expected_version=tool.version
            )
            replay = await client.post(tool.url, json=changed_payload, headers=changed_headers)
            if replay.is_success:
                result, response_diagnosis, response_gate, response_tokens = await self._heal_response(
                    _body(replay), tool, payload
                )
                result.status_code = replay.status_code
                result.attempts = 2
                result.healed = True
                result.schema_version = staged.version
                result.rollback_ref = rollback_ref
                outcome = "response_contract_mismatch" if result.response_contract_ok is False else "healed"
                return await self._finish(
                    result, tool, payload, started, outcome=outcome,
                    diagnosis=response_diagnosis or diagnosis,
                    gate=(
                        response_gate
                        if response_gate is not None and response_gate.evaluated
                        else verdict
                    ),
                    evidence="; ".join(result.response_contract_issues or [])
                    or (response_diagnosis.evidence if response_diagnosis else diagnosis.evidence),
                    tokens_used=tokens_used + response_tokens,
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            rolled_back = await self.registry.rollback(
                tool_name, rollback_ref, expected_version=staged.version
            )
            failed_validation = 400 <= replay.status_code < 500
            replay_error = _body(replay)
            result = ExecutionResult(
                replay.status_code, sanitize_error(replay_error), 2,
                escalation=_escalation(
                    replay.status_code,
                    f"replay failed after {elapsed_ms:.1f} ms; the schema patch was rolled back",
                    missing_fields=_missing_fields(replay_error, changed_payload),
                ),
                schema_version=rolled_back.version,
                rollback_ref=rollback_ref,
            )
            return await self._finish(
                result, tool, payload, started,
                outcome="failed_rolled_back" if failed_validation else "replay_failed",
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

    async def _diagnose_request(
        self, error_body: Any, payload: dict[str, Any]
    ) -> tuple[Any, int, bool]:
        diagnosis = diagnose_error(error_body, payload, header_allowlist=self.header_allowlist)
        tokens_used = 0
        llm_failure = False
        if diagnosis is None and self.llm_diagnoser is not None:
            try:
                diagnosis, tokens_used = await self.llm_diagnoser.diagnose(
                    error_body, payload, header_allowlist=self.header_allowlist
                )
            except Exception:
                llm_failure = True
        return diagnosis, tokens_used, llm_failure

    async def _heal_response(
        self, body: Any, tool: ToolSchema, request_payload: dict[str, Any]
    ) -> tuple[ExecutionResult, Any, GateVerdict | None, int]:
        if tool.response_schema is None:
            return ExecutionResult(200, body, 1, schema_version=tool.version), None, None, 0
        issues = check_response(body, tool.response_schema)
        if not issues:
            return ExecutionResult(
                200, body, 1, schema_version=tool.version,
                response_contract_ok=True, response_contract_issues=[],
            ), None, GateVerdict(allowed=True, evaluated=False), 0
        if not isinstance(body, dict):
            return ExecutionResult(
                200, body, 1, schema_version=tool.version,
                response_contract_ok=False, response_contract_issues=issues,
            ), None, GateVerdict(allowed=False, reasons=["response root is not an object"]), 0

        diagnosis, implicated = diagnose_response(body, tool.response_schema)
        tokens_used = 0
        if diagnosis is None and self.llm_diagnoser is not None and implicated:
            error = {"detail": [
                {"loc": ["body", *path.strip("/").split("/")], "msg": "required"}
                for path in sorted(implicated) if path
            ]}
            try:
                diagnosis, tokens_used = await self.llm_diagnoser.diagnose(
                    error, body, header_allowlist={},
                )
            except Exception:
                diagnosis = None

        if diagnosis is None or not diagnosis.ops or any(
            operation.op not in {"rename", "move", "cast"} for operation in diagnosis.ops
        ):
            return ExecutionResult(
                200, body, 1, schema_version=tool.version,
                response_contract_ok=False, response_contract_issues=issues,
            ), diagnosis, None, tokens_used

        verdict = evaluate_mapping(diagnosis.ops, body, implicated_paths=implicated)
        if not verdict.allowed:
            return ExecutionResult(
                200, body, 1, schema_version=tool.version,
                response_contract_ok=False, response_contract_issues=issues,
            ), diagnosis, verdict, tokens_used
        try:
            mapped, _ = apply_mapping(body, {}, diagnosis.ops)
            for operation in diagnosis.ops:
                if operation.op in {"move", "rename"} and operation.source:
                    _prune_empty_ancestors(mapped, operation.source)
            remaining = check_response(mapped, tool.response_schema)
        except (MappingError, ValueError, TypeError) as exc:
            remaining = [str(exc)]
            mapped = body
        if remaining:
            return ExecutionResult(
                200, body, 1, schema_version=tool.version,
                response_contract_ok=False, response_contract_issues=remaining,
            ), diagnosis, verdict, tokens_used
        return ExecutionResult(
            200, mapped, 1, healed=True, schema_version=tool.version,
            response_contract_ok=True, response_contract_issues=[],
        ), diagnosis, verdict, tokens_used


def _prune_empty_ancestors(document: dict[str, Any], pointer: str) -> None:
    """Remove only object wrappers emptied by this move; preserve other empties."""
    parts = _pointer_parts(pointer)
    for length in range(len(parts) - 1, 0, -1):
        parent: Any = document
        valid = True
        for part in parts[: length - 1]:
            if not isinstance(parent, dict) or part not in parent:
                valid = False
                break
            parent = parent[part]
        if valid and isinstance(parent, dict):
            key = parts[length - 1]
            child = parent.get(key)
            if isinstance(child, dict) and not child:
                del parent[key]


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


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _matches_cast(value: Any, target: str | None) -> bool:
    return {
        "int": lambda: isinstance(value, int) and not isinstance(value, bool),
        "float": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "str": lambda: isinstance(value, str),
        "bool": lambda: isinstance(value, bool),
    }.get(target, lambda: False)()


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


def _missing_fields(error_body: Any, payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for path in extract_implicated_paths(error_body):
        if path.rsplit("/", 1)[-1].lower().startswith("x-"):
            continue
        if not _read_pointer(payload, path)[0]:
            field = path.strip("/").replace("/", ".")
            if field and field not in missing:
                missing.append(field)
    return missing


def _escalation(
    status_code: int, reason: str, *, missing_fields: list[str] | None = None,
) -> Escalation:
    return Escalation(
        reason=reason,
        missing_fields=missing_fields or [],
        suggested_action=f"Review the API rejection (HTTP {status_code}) and update the call or tool contract.",
    )
