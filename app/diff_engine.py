"""Deterministic Tier 1 rules that turn structured API errors into MappingOps."""

from collections.abc import Mapping
from typing import Any

from app.sanitizer import sanitize_error
from app.schemas import Diagnosis, MappingOp


def diagnose_error(
    error_body: Any,
    payload: Mapping[str, Any],
    *,
    header_allowlist: Mapping[str, str] | None = None,
) -> Diagnosis | None:
    """Recognize supported validation changes from extracted error fields.

    The returned operations point only to source data in the original payload,
    or to a header whose value must come from the configured allowlist.
    """
    evidence = sanitize_error(error_body)
    text = evidence.lower()
    allowed_headers = header_allowlist or {}
    ops: list[MappingOp] = []

    if "user_id" in text and any(word in text for word in ("integer", "int")):
        ops.append(MappingOp(op="cast", source="/user_id", target="/user_id", cast_to="int"))

    if (
        "target.id" in text
        and "required" in text
        and "integer" in text
        and "user_id" in payload
    ):
        ops.extend(
            [
                MappingOp(op="move", source="/user_id", target="/target/id"),
                MappingOp(op="cast", source="/user_id", target="/target/id", cast_to="int"),
            ]
        )

    if "account_id" in text and "required" in text and "user_id" in payload:
        ops.append(MappingOp(op="rename", source="/user_id", target="/account_id"))

    if "header" in text and "required" in text:
        name = next(
            (candidate for candidate in allowed_headers if candidate.lower() in text),
            None,
        )
        # This mock-specific rule can identify the header so the gate can refuse it.
        if name is None and "x-api-version" in text:
            name = "X-Api-Version"
        if name is not None:
            ops.append(MappingOp(op="add_header", header=name))

    if not ops:
        return None
    return Diagnosis(is_schema_drift=True, source="rule", ops=ops, evidence=evidence)
