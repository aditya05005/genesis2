"""Small deterministic response-shape checker for the mock tool contract."""

from collections.abc import Mapping
from typing import Any

from app.sanitizer import sanitize_error
from app.schemas import Diagnosis, MappingOp


def check_response(value: Any, schema: Mapping[str, Any], pointer: str = "") -> list[str]:
    """Report required, type, and explicitly forbidden-extra-field violations.

    This checks structure only. It cannot detect a response that is structurally
    valid but has silently changed meaning.
    """
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        return [f"{pointer or '/'} should have type {expected}"]
    if expected == "object":
        properties = schema.get("properties", {})
        issues = []
        for field in schema.get("required", []):
            if field not in value:
                issues.append(f"{pointer}/{field} is required")
        for field, child_schema in properties.items():
            if field in value:
                issues.extend(check_response(value[field], child_schema, f"{pointer}/{field}"))
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            for field in value.keys() - allowed:
                issues.append(f"{pointer}/{field} is unexpected")
        return issues
    if expected == "array" and "items" in schema:
        issues = []
        for index, item in enumerate(value):
            issues.extend(check_response(item, schema["items"], f"{pointer}/{index}"))
        return issues
    return []


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def diagnose_response(value: Any, schema: Mapping[str, Any]) -> tuple[Diagnosis | None, set[str]]:
    """Propose only structural response repairs supported by matching field names."""
    missing_or_invalid: dict[str, Mapping[str, Any]] = {}

    def inspect(actual: Any, expected: Mapping[str, Any], pointer: str = "") -> None:
        kind = expected.get("type")
        if kind == "object" and isinstance(actual, Mapping):
            properties = expected.get("properties", {})
            for field in expected.get("required", []):
                child_path = f"{pointer}/{field}"
                if field not in actual:
                    missing_or_invalid[child_path] = properties.get(field, {})
            for field, child_schema in properties.items():
                if field in actual:
                    inspect(actual[field], child_schema, f"{pointer}/{field}")
        elif kind == "array" and isinstance(actual, list) and "items" in expected:
            for index, item in enumerate(actual):
                inspect(item, expected["items"], f"{pointer}/{index}")
        elif kind and not _matches_type(actual, kind):
            missing_or_invalid[pointer or ""] = expected

    inspect(value, schema)
    implicated = set(missing_or_invalid)
    sources = _value_paths(value)
    ops: list[MappingOp] = []
    for target, target_schema in missing_or_invalid.items():
        if not target:
            continue
        leaf = target.rsplit("/", 1)[-1]
        candidates = [path for path in sources if path.rsplit("/", 1)[-1] == leaf]
        if len(candidates) != 1:
            continue
        source = candidates[0]
        if source != target:
            ops.append(MappingOp(op="move", source=source, target=target))
        cast_to = {
            "integer": "int", "number": "float", "string": "str", "boolean": "bool",
        }.get(target_schema.get("type"))
        if cast_to:
            source_value = _read_value(value, source)
            if not _matches_type(source_value, target_schema["type"]):
                ops.append(MappingOp(op="cast", source=source, target=target, cast_to=cast_to))
    if not ops:
        return None, implicated
    return Diagnosis(
        is_schema_drift=True,
        source="rule",
        ops=ops,
        evidence=sanitize_error(check_response(value, schema)),
    ), implicated


def _value_paths(value: Any) -> list[str]:
    paths: list[str] = []

    def visit(item: Any, pointer: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{pointer}/{index}")
        else:
            paths.append(pointer)

    visit(value, "")
    return paths


def _read_value(value: Any, pointer: str) -> Any:
    current = value
    for part in pointer.strip("/").split("/") if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current
