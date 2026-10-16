"""Deterministic checks for proposed payload and allowlisted-header mappings."""

from collections.abc import Mapping, Set
from typing import Any

from app.schemas import GateVerdict, MappingOp


def evaluate_mapping(
    ops: list[MappingOp],
    payload: Mapping[str, Any],
    *,
    implicated_paths: Set[str],
    header_allowlist: Mapping[str, str] | None = None,
) -> GateVerdict:
    """Check provenance, scope, and cast safety before an adapter applies ops.

    ``implicated_paths`` must come from structured error analysis. This check
    does not replace validation of the transformed request against the target
    schema, which must happen before replay.
    """
    allowed_headers = header_allowlist or {}
    reasons: list[str] = []
    for index, op in enumerate(ops):
        prefix = f"operation {index}"
        if op.op == "add_header":
            if not op.header:
                reasons.append(f"{prefix} has no header name")
            elif op.header not in allowed_headers:
                reasons.append(f"header {op.header!r} is not allowlisted")
            if op.source is not None or op.target is not None or op.cast_to is not None:
                reasons.append(f"{prefix} has fields unrelated to add_header")
            continue

        if not op.source or not op.target:
            reasons.append(f"{prefix} requires source and target paths")
            continue
        value_found, source_value = _read_pointer(payload, op.source)
        if not value_found:
            reasons.append(f"{op.target} has no source value at {op.source}")
            continue
        if op.target not in implicated_paths:
            reasons.append(f"{op.target} was not implicated by the error")
        if op.op == "cast":
            if op.cast_to is None:
                reasons.append(f"{prefix} has no cast target")
            elif not _lossless_cast(source_value, op.cast_to):
                reasons.append(f"cast from {op.source} to {op.cast_to} is not lossless")
        elif op.cast_to is not None:
            reasons.append(f"{prefix} sets cast_to for a non-cast operation")
    return GateVerdict(allowed=not reasons, reasons=reasons)


def _read_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
    """Read a JSON Pointer and distinguish a missing value from JSON null."""
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _lossless_cast(value: Any, target: str) -> bool:
    """Allow only round-trippable scalar conversions; reject ambiguous forms."""
    if target == "str":
        return isinstance(value, (str, int, float, bool))
    if target == "int":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            try:
                return str(int(value)) == value.strip()
            except ValueError:
                return False
        if isinstance(value, float):
            return value.is_integer()
        return False
    if target == "float":
        if isinstance(value, bool):
            return False
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        return converted == value or (isinstance(value, str) and str(converted) == value.strip())
    if target == "bool":
        return isinstance(value, bool)
    return False
