"""Single data-access layer for the HealGate dashboard.

This is the ONLY file that reads files, calls HTTP, or generates fake data.
Everything else imports from here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.schemas import AuditLogEntry

# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

SOURCE: str = os.environ.get("HEALGATE_SOURCE", "fake").lower()

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Audit entries
# ---------------------------------------------------------------------------


def load_audit_entries() -> list[AuditLogEntry]:
    """Return audit log entries from the configured source.

    * *fake*: generate ~60 realistic entries via ``fake_data``.
    * *live*: read ``audit_log.jsonl`` at the repo root (one JSON per line).
      Returns ``[]`` when the file does not exist.
    """
    if SOURCE == "fake":
        from dashboard.fake_data import generate_audit_entries

        return generate_audit_entries()

    log_path = _REPO_ROOT / "audit_log.jsonl"
    if not log_path.exists():
        return []

    entries: list[AuditLogEntry] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(AuditLogEntry.model_validate_json(line))
        except Exception:
            continue  # skip malformed lines
    return entries


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------


def load_benchmark_runs() -> pd.DataFrame:
    """Return benchmark runs as a DataFrame.

    * *fake*: generate sample runs via ``fake_data``.
    * *live*: read ``harness/results.json`` (a JSON list of run dicts).
      Returns an empty DataFrame when the file does not exist.

    Columns: run_id, system, mutation, error_format, seed, outcome,
    mttr_ms, tokens, tier.
    """
    if SOURCE == "fake":
        from dashboard.fake_data import generate_benchmark_runs

        return generate_benchmark_runs()

    results_path = _REPO_ROOT / "harness" / "results.json"
    if not results_path.exists():
        return pd.DataFrame(
            columns=[
                "run_id", "system", "mutation", "error_format",
                "seed", "outcome", "mttr_ms", "tokens", "tier",
            ]
        )

    try:
        data: list[dict[str, Any]] = json.loads(
            results_path.read_text(encoding="utf-8")
        )
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(
            columns=[
                "run_id", "system", "mutation", "error_format",
                "seed", "outcome", "mttr_ms", "tokens", "tier",
            ]
        )


# ---------------------------------------------------------------------------
# Mutation control
# ---------------------------------------------------------------------------

MUTATION_NAMES = [
    "none",
    "field_rename",
    "type_change",
    "nesting",
    "required_header",
    "compound_rename_header",
    "new_required_field",
    "new_auth_scope",
    "lossy_cast",
]


def set_mutation(name: str) -> bool:
    """Apply *name* as the active mutation on the mock service.

    Connect to the actual local mock service. The dashboard data source setting
    controls audit/benchmark records only; it must not turn a live control into
    a fake success.
    """
    try:
        resp = httpx.put(
            "http://localhost:8000/admin/mutation",
            json={"mutation": name, "error_format": "fastapi"},
            timeout=5.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_mock_state() -> dict[str, Any]:
    """Read active mutation and accepted-call records from the mock service."""
    response = httpx.get("http://localhost:8000/admin/state", timeout=3.0)
    response.raise_for_status()
    return response.json()


def clear_mock_records() -> int | None:
    """Clear accepted-call records from the mock service; return count if known."""
    try:
        response = httpx.delete("http://localhost:8000/admin/records", timeout=3.0)
        response.raise_for_status()
        return int(response.json().get("deleted", 0))
    except Exception:
        return None
