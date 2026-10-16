"""Generate realistic sample data for offline dashboard development."""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.schemas import AuditLogEntry, GateVerdict, MappingOp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_NAMES = ["lookup_user", "create_order", "get_invoice"]

HEALABLE_MUTATIONS = [
    "field_rename",
    "type_change",
    "nesting",
    "required_header",
    "compound_rename_header",
]

UNHEALABLE_MUTATIONS = [
    "new_required_field",
    "new_auth_scope",
    "lossy_cast",
]

ERROR_FORMATS = ["fastapi", "problem", "joi", "plain"]

_OPS_POOL: list[list[dict[str, Any]]] = [
    [{"op": "cast", "source": "/user_id", "target": "/user_id", "cast_to": "int"}],
    [{"op": "rename", "source": "/user_id", "target": "/account_id"}],
    [
        {"op": "move", "source": "/user_id", "target": "/target/id"},
        {"op": "cast", "source": "/target/id", "target": "/target/id", "cast_to": "int"},
    ],
    [{"op": "add_header", "header": "X-Api-Version"}],
    [
        {"op": "rename", "source": "/user_id", "target": "/account_id"},
        {"op": "add_header", "header": "X-Api-Version"},
    ],
]

_EVIDENCE_POOL = [
    "body.user_id: must be an integer",
    "body.account_id: field required",
    "body.target.id: field required",
    "X-Api-Version: required header is missing or invalid",
    "body.tenant_id: field required",
    "Required scope: accounts:write",
    "body.user_id: must be a whole number",
]


def _random_hash() -> str:
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Audit entries
# ---------------------------------------------------------------------------


def generate_audit_entries(count: int = 60) -> list[AuditLogEntry]:
    """Return *count* varied audit log entries with plausible timestamps."""
    rng = random.Random(42)
    now = datetime.now(timezone.utc)
    entries: list[AuditLogEntry] = []

    for i in range(count):
        ts = now - timedelta(seconds=(count - i) * rng.randint(2, 8))
        is_healable = rng.random() < 0.65
        mutation = rng.choice(HEALABLE_MUTATIONS if is_healable else UNHEALABLE_MUTATIONS)

        if is_healable:
            outcome_roll = rng.random()
            if outcome_roll < 0.75:
                outcome = "healed"
            elif outcome_roll < 0.90:
                outcome = "halted"
            else:
                outcome = "failed_rolled_back"
        else:
            outcome = rng.choice(["halted", "failed_rolled_back"])

        source: str = rng.choice(["rule", "llm"])
        status = rng.choice([400, 422]) if outcome != "healed" else 422
        ops_raw = rng.choice(_OPS_POOL)
        ops = [MappingOp(**op) for op in ops_raw]
        gate_allowed = outcome == "healed"
        gate_reasons = [] if gate_allowed else [rng.choice(_EVIDENCE_POOL)]
        latency = round(rng.uniform(15.0, 3800.0), 1)
        tokens = rng.randint(0, 520) if source == "llm" else 0
        ver_before = rng.randint(1, 5)
        ver_after = ver_before + 1 if outcome == "healed" else None
        rollback = f"{rng.choice(TOOL_NAMES)}:v{ver_before}" if outcome == "healed" else None

        entries.append(
            AuditLogEntry(
                timestamp=ts.isoformat(),
                tool_name=rng.choice(TOOL_NAMES),
                schema_version_before=ver_before,
                schema_version_after=ver_after,
                status_code=status,
                diagnosis_source=source,
                ops=ops,
                evidence=rng.choice(_EVIDENCE_POOL),
                gate=GateVerdict(allowed=gate_allowed, reasons=gate_reasons),
                outcome=outcome,
                payload_hash_before=_random_hash(),
                rollback_ref=rollback,
                latency_ms=latency,
                tokens_used=tokens,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------


def generate_benchmark_runs(runs_per_cell: int = 5) -> pd.DataFrame:
    """Return a DataFrame of benchmark run records for every combination."""
    rng = random.Random(99)
    rows: list[dict[str, Any]] = []
    run_id = 0

    all_mutations = HEALABLE_MUTATIONS + UNHEALABLE_MUTATIONS

    for system in ("healgate", "baseline"):
        for mutation in all_mutations:
            for fmt in ERROR_FORMATS:
                for seed in range(runs_per_cell):
                    run_id += 1
                    is_healable = mutation in HEALABLE_MUTATIONS

                    if system == "healgate":
                        if is_healable:
                            roll = rng.random()
                            if roll < 0.82:
                                outcome = "healed"
                            elif roll < 0.92:
                                outcome = "false_heal"
                            else:
                                outcome = "failed"
                            tier = 1 if rng.random() < 0.6 else 2
                        else:
                            outcome = "correctly_halted" if rng.random() < 0.90 else "failed"
                            tier = None
                    else:
                        if is_healable:
                            outcome = "healed" if rng.random() < 0.35 else "failed"
                            tier = None
                        else:
                            outcome = "correctly_halted" if rng.random() < 0.30 else "failed"
                            tier = None

                    mttr = round(rng.uniform(120, 4200) if system == "healgate" else rng.uniform(800, 28000), 1)
                    tokens = rng.randint(0, 350) if system == "healgate" else rng.randint(200, 2400)

                    rows.append(
                        {
                            "run_id": run_id,
                            "system": system,
                            "mutation": mutation,
                            "error_format": fmt,
                            "seed": seed,
                            "outcome": outcome,
                            "mttr_ms": mttr,
                            "tokens": tokens,
                            "tier": tier,
                        }
                    )

    return pd.DataFrame(rows)
