"""Small deterministic response-shape checker for the mock tool contract."""

from collections.abc import Mapping
from typing import Any


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
