"""Shared contracts for diagnoses, safety decisions, and audit records."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MappingOp(BaseModel):
    """Describe a data move; values must come from the original request."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["rename", "move", "cast", "wrap", "add_header"]
    source: str | None = None
    target: str | None = None
    cast_to: Literal["str", "int", "float", "bool"] | None = None
    header: str | None = None


class Diagnosis(BaseModel):
    """A rule or model's proposed response to a sanitized API error."""

    model_config = ConfigDict(extra="forbid")

    is_schema_drift: bool
    source: Literal["rule", "llm"]
    ops: list[MappingOp] = Field(default_factory=list)
    evidence: str
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GuardrailCheck(BaseModel):
    """One named pass/fail result from the deterministic safety matrix."""

    name: str
    passed: bool
    reason: str


class GateVerdict(BaseModel):
    """Deterministic decision about whether proposed operations are safe."""

    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    evaluated: bool = True
    checks: list[GuardrailCheck] | None = None


class Escalation(BaseModel):
    """Explain why an agent or user must resolve a call manually."""

    reason: str
    missing_fields: list[str] = Field(default_factory=list)
    suggested_action: str


class AuditLogEntry(BaseModel):
    """Record an outcome without retaining the original payload."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    tool_name: str
    schema_version_before: int
    schema_version_after: int | None
    status_code: int
    diagnosis_source: Literal["rule", "llm"] | None
    ops: list[MappingOp]
    evidence: str
    gate: GateVerdict
    outcome: Literal[
        "succeeded", "healed", "halted", "replay_failed", "failed_rolled_back",
        "response_contract_mismatch",
    ]
    payload_hash_before: str
    rollback_ref: str | None
    latency_ms: float
    tokens_used: int = 0
