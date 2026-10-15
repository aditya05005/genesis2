from typing import Literal, Optional
from pydantic import BaseModel, Field

class MappingOp(BaseModel):
    """One structural change. Carries no literal values: data comes only from the original call."""
    op: Literal["rename", "move", "cast", "wrap", "add_header"]
    source: Optional[str] = None   # JSON pointer into the ORIGINAL payload
    target: Optional[str] = None   # JSON pointer in the NEW payload shape
    cast_to: Optional[Literal["str", "int", "float", "bool"]] = None
    header: Optional[str] = None   # header NAME only; values come from HEADER_ALLOWLIST in config

class Diagnosis(BaseModel):
    is_schema_drift: bool
    source: Literal["rule", "llm"]
    ops: list[MappingOp] = []
    evidence: str                  # sanitized error excerpt that justified the ops
    llm_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)  # logged only, never gates

class GateVerdict(BaseModel):
    allowed: bool
    reasons: list[str] = []        # e.g. "/target/id has no source value"

class Escalation(BaseModel):
    reason: str
    missing_fields: list[str] = []
    suggested_action: str          # what the agent or user should do next

class AuditLogEntry(BaseModel):
    timestamp: str
    tool_name: str
    schema_version_before: int
    schema_version_after: Optional[int]
    status_code: int
    diagnosis_source: Literal["rule", "llm"]
    ops: list[MappingOp]
    evidence: str
    gate: GateVerdict
    outcome: Literal["healed", "halted", "failed_rolled_back"]
    payload_hash_before: str       # hash, not the raw payload (may contain PII)
    rollback_ref: Optional[str]
    latency_ms: float
    tokens_used: int = 0