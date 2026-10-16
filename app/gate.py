"""Named deterministic guardrails for proposed payload and header mappings."""

from collections.abc import Mapping, Set
from typing import Any

from app.schemas import GateVerdict, GuardrailCheck, MappingOp


_SENSITIVE_PARTS = ("auth", "token", "scope", "credential", "secret", "password")


def evaluate_mapping(
    ops: list[MappingOp],
    payload: Mapping[str, Any],
    *,
    implicated_paths: Set[str],
    header_allowlist: Mapping[str, str] | None = None,
) -> GateVerdict:
    """Evaluate every named safety check and return the complete guardrail matrix."""
    allowed_headers = header_allowlist or {}
    failures: list[str] = []

    auth_reason = "No authentication, token, or scope repair is proposed."
    sensitive = [path for path in implicated_paths if _sensitive(path)]
    sensitive.extend(
        path for op in ops for path in (op.source, op.target, op.header)
        if path and _sensitive(path)
    )
    auth_passed = not sensitive
    if not auth_passed:
        auth_reason = "Authentication tokens, credentials, and scopes cannot be repaired automatically."
        failures.append(auth_reason)

    lossy_ops = [
        op for op in ops
        if op.op == "cast" and (
            op.cast_to is None
            or not (found := _read_pointer(payload, op.source or ""))[0]
            or not _lossless_cast(found[1], op.cast_to)
        )
    ]
    cast_passed = not lossy_ops
    cast_reason = (
        "Every cast is lossless."
        if cast_passed else "One or more casts are missing a target type or would lose/change data."
    )
    if not cast_passed:
        failures.append(cast_reason)

    provenance_issues: list[str] = []
    for index, op in enumerate(ops):
        if op.op == "add_header":
            if not op.header or op.header not in allowed_headers:
                provenance_issues.append(f"operation {index}: header is not allowlisted")
            if op.source is not None or op.target is not None or op.cast_to is not None:
                provenance_issues.append(f"operation {index}: header operation has payload fields")
            continue
        found, _ = _read_pointer(payload, op.source or "")
        if not op.source or not op.target or not found:
            provenance_issues.append(f"operation {index}: target has no source value in the original payload")
    provenance_passed = not provenance_issues
    provenance_reason = "; ".join(provenance_issues) if provenance_issues else (
        "Every target value comes from the original payload or an allowlisted header."
    )
    if not provenance_passed:
        failures.append(provenance_reason)

    scope_issues: list[str] = []
    for index, op in enumerate(ops):
        if op.op == "add_header":
            if op.header and not any(
                path.rsplit("/", 1)[-1].lower() == op.header.lower()
                for path in implicated_paths
            ):
                scope_issues.append(f"operation {index}: header was not implicated")
        elif op.target and op.target not in implicated_paths:
            scope_issues.append(f"operation {index}: target {op.target} was not implicated")
    scope_passed = not scope_issues
    scope_reason = "; ".join(scope_issues) if scope_issues else (
        "Every operation is limited to a field implicated by the error."
    )
    if not scope_passed:
        failures.append(scope_reason)

    ambiguity_issues: list[str] = []
    for index, op in enumerate(ops):
        if op.op not in {"move", "rename"} or not op.target or not op.source:
            continue
        target_found, _ = _read_pointer(payload, op.target)
        if target_found:
            continue
        source_found, source_value = _read_pointer(payload, op.source)
        if not source_found:
            continue
        source_kind = _value_kind(source_value)
        candidates = [
            path for path, value in _value_paths(payload)
            if path not in implicated_paths and _value_kind(value) == source_kind
        ]
        if len(candidates) > 1:
            ambiguity_issues.append(
                f"operation {index}: {op.target} has {len(candidates)} compatible source candidates"
            )
    ambiguity_passed = not ambiguity_issues
    ambiguity_reason = "; ".join(ambiguity_issues) if ambiguity_issues else (
        "No target has multiple compatible source candidates."
    )
    if not ambiguity_passed:
        failures.append(ambiguity_reason)

    checks = [
        GuardrailCheck(name="auth", passed=auth_passed, reason=auth_reason),
        GuardrailCheck(name="lossy_cast", passed=cast_passed, reason=cast_reason),
        GuardrailCheck(name="provenance", passed=provenance_passed, reason=provenance_reason),
        GuardrailCheck(name="scope", passed=scope_passed, reason=scope_reason),
        GuardrailCheck(name="ambiguity", passed=ambiguity_passed, reason=ambiguity_reason),
    ]
    return GateVerdict(allowed=not failures, reasons=failures, checks=checks)


def _sensitive(path: str) -> bool:
    lowered = path.lower()
    return any(part in lowered for part in _SENSITIVE_PARTS)


def _read_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
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
        return isinstance(value, float) and value.is_integer()
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


def _value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "other"


def _value_paths(payload: Any, pointer: str = "") -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            values.extend(_value_paths(child, f"{pointer}/{token}"))
    elif isinstance(payload, list):
        for index, child in enumerate(payload):
            values.extend(_value_paths(child, f"{pointer}/{index}"))
    else:
        values.append((pointer, payload))
    return values
