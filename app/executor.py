"""Minimal one-replay executor with a deterministic type-change rule."""

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
        """Call, diagnose only the known integer mismatch, gate, and replay once."""
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

            diagnosis = _type_change_diagnosis(first)
            if diagnosis is None or not diagnosis.ops:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, "no supported deterministic repair was found"),
                    schema_version=tool.version,
                )

            verdict = evaluate_mapping(
                diagnosis.ops, payload, implicated_paths={"/user_id"},
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
                _dry_run(changed_payload, diagnosis.ops)
            except (MappingError, ValueError, TypeError) as exc:
                return ExecutionResult(
                    first.status_code, sanitize_error(_body(first)), 1,
                    escalation=_escalation(first.status_code, f"dry run refused the repair: {exc}"),
                    schema_version=tool.version,
                )

            # Writes are replayed only after this mock explicitly rejects validation.
            replay = await client.post(tool.url, json=changed_payload, headers=changed_headers)
            if replay.is_success:
                new_schema = _schema_with_cast(tool.schema, diagnosis.ops)
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


def _type_change_diagnosis(response: httpx.Response) -> Diagnosis | None:
    """Recognize only an explicit user_id integer-validation rejection."""
    try:
        detail = response.json()
    except ValueError:
        return None
    text = sanitize_error(detail).lower()
    if "user_id" not in text or not any(word in text for word in ("integer", "int")):
        return None
    return Diagnosis(
        is_schema_drift=True,
        source="rule",
        ops=[MappingOp(op="cast", source="/user_id", target="/user_id", cast_to="int")],
        evidence=text,
    )


def _dry_run(payload: dict[str, Any], ops: list[MappingOp]) -> None:
    """Check the inferred target field types before a network replay."""
    for op in ops:
        if op.op != "cast" or op.target != "/user_id":
            continue
        value = payload.get("user_id")
        if op.cast_to == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("/user_id is not an integer after the proposed cast")


def _schema_with_cast(schema: dict[str, Any], ops: list[MappingOp]) -> dict[str, Any]:
    """Return a copied agent-facing schema reflecting successful casts."""
    updated = json.loads(json.dumps(schema))
    for op in ops:
        if op.op == "cast" and op.target == "/user_id" and op.cast_to:
            updated.setdefault("properties", {}).setdefault("user_id", {})["type"] = op.cast_to
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
