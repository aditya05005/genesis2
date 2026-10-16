"""Minimal one-replay executor with deterministic type and rename rules."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.adapter import MappingError, apply_mapping
from app.gate import evaluate_mapping
from app.registry import SchemaRegistry, ToolSchema
from app.sanitizer import sanitize_error
from app.schemas import Diagnosis, Escalation, MappingOp
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


class HealingExecutor:
    """Call a registered tool and attempt at most one safe validation repair."""

    def __init__(
        self,
        registry: SchemaRegistry,
        *,
        timeout_seconds: float = 5.0,
        header_allowlist: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.registry = registry
        self.timeout = httpx.Timeout(timeout_seconds)
        self.header_allowlist = header_allowlist or {}
        self.transport = transport

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
                return ExecutionResult(first.status_code, _body(first), 1, schema_version=tool.version)

            action = action_for_status(first.status_code)
            if action not in (Action.HEAL, Action.HEAL_RULES_ONLY):
                return ExecutionResult(
                    first.status_code, _body(first), 1,
                    escalation=_escalation(first.status_code, "status policy does not allow healing"),
                    schema_version=tool.version,
                )
            if action == Action.HEAL_RULES_ONLY:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "no content-type repair rule is implemented yet"),
                    schema_version=tool.version,
                )

            diagnosis = _diagnose(first, payload, self.header_allowlist)
            if diagnosis is None or not diagnosis.ops:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "no supported deterministic repair was found"),
                    schema_version=tool.version,
                )

            verdict = evaluate_mapping(
                diagnosis.ops, payload,
                implicated_paths={op.target for op in diagnosis.ops if op.target},
                header_allowlist=self.header_allowlist,
            )
            if not verdict.allowed:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "; ".join(verdict.reasons)),
                    schema_version=tool.version,
                )

            try:
                changed_payload, changed_headers = apply_mapping(
                    payload, request_headers, diagnosis.ops,
                    header_allowlist=self.header_allowlist,
                )
                _dry_run(changed_payload, changed_headers, diagnosis.ops, self.header_allowlist)
            except (MappingError, ValueError, TypeError) as exc:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, f"dry run refused the repair: {exc}"),
                    schema_version=tool.version,
                )

            # Writes are replayed only after this mock explicitly rejects validation.
            replay = await client.post(tool.url, json=changed_payload, headers=changed_headers)
            if replay.is_success:
                new_schema = _schema_after_ops(tool.schema, diagnosis.ops)
                updated, rollback_ref = await self.registry.publish_after_success(
                    tool_name, new_schema, expected_version=tool.version
                )
                return ExecutionResult(
                    replay.status_code, _body(replay), 2, healed=True,
                    schema_version=updated.version, rollback_ref=rollback_ref,
                )

            # Failed replay never publishes the proposed schema.
            elapsed_ms = (time.perf_counter() - started) * 1000
            return ExecutionResult(
                replay.status_code, sanitize_error(_body(replay)), 2,
                escalation=_escalation(
                    replay.status_code,
                    f"replay failed after {elapsed_ms:.1f} ms; the schema was not updated",
                ),
                schema_version=tool.version,
            )


def _diagnose(
    response: httpx.Response,
    payload: dict[str, Any],
    header_allowlist: dict[str, str],
) -> Diagnosis | None:
    """Recognize supported field errors or a missing allowlisted header."""
    try:
        detail = response.json()
    except ValueError:
        return None
    text = sanitize_error(detail).lower()
    if "user_id" in text and any(word in text for word in ("integer", "int")):
        return Diagnosis(
            is_schema_drift=True,
            source="rule",
            ops=[MappingOp(op="cast", source="/user_id", target="/user_id", cast_to="int")],
            evidence=text,
        )
    if (
        "account_id" in text
        and "required" in text
        and "user_id" in payload
    ):
        return Diagnosis(
            is_schema_drift=True,
            source="rule",
            ops=[MappingOp(op="rename", source="/user_id", target="/account_id")],
            evidence=text,
        )
    if "header" in text and "required" in text:
        name = next(
            (candidate for candidate in header_allowlist if candidate.lower() in text),
            None,
        )
        # The mock's structured error names this header even when it is not configured.
        if name is None and "x-api-version" in text:
            name = "X-Api-Version"
        if name is not None:
            return Diagnosis(
                is_schema_drift=True,
                source="rule",
                ops=[MappingOp(op="add_header", header=name)],
                evidence=text,
            )
    return None


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


def _escalation(status_code: int, reason: str) -> Escalation:
    return Escalation(
        reason=reason,
        suggested_action=f"Review the API rejection (HTTP {status_code}) and update the call or tool contract.",
    )
