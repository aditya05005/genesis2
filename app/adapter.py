"""Apply declarative MappingOps without inventing payload values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from app.schemas import MappingOp


class MappingError(ValueError):
    """A mapping cannot be applied to the supplied call."""


def apply_mapping(
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    ops: list[MappingOp],
    *,
    header_allowlist: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply ops using source values from the unmodified original payload."""
    original = deepcopy(dict(payload))
    output = deepcopy(original)
    output_headers = dict(headers)
    allowed_headers = header_allowlist or {}

    for op in ops:
        if op.op == "add_header":
            if not op.header or op.header not in allowed_headers:
                raise MappingError(f"header {op.header!r} is not allowlisted")
            output_headers[op.header] = allowed_headers[op.header]
            continue
        if not op.source or not op.target:
            raise MappingError("payload mappings require source and target pointers")
        found, value = _read_pointer(original, op.source)
        if not found:
            raise MappingError(f"source {op.source!r} does not exist")
        if op.op == "cast":
            value = _cast(value, op.cast_to)
        elif op.op == "wrap":
            value = {"value": deepcopy(value)}
        _write_pointer(output, op.target, value)
        if op.op in ("rename", "move") and op.source != op.target:
            _remove_pointer(output, op.source)

    return output, output_headers


def _parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise MappingError(f"invalid JSON Pointer {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _read_pointer(value: Any, pointer: str) -> tuple[bool, Any]:
    current = value
    for part in _parts(pointer):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _write_pointer(document: Any, pointer: str, value: Any) -> None:
    parts = _parts(pointer)
    if not parts:
        raise MappingError("replacing the whole request is not supported")
    parent = document
    for part in parts[:-1]:
        if isinstance(parent, dict):
            if part not in parent:
                parent[part] = {}
            parent = parent[part]
        elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
            parent = parent[int(part)]
        else:
            raise MappingError(f"parent for {pointer!r} does not exist")
    key = parts[-1]
    if isinstance(parent, dict):
        parent[key] = deepcopy(value)
    elif isinstance(parent, list) and key.isdigit() and int(key) < len(parent):
        parent[int(key)] = deepcopy(value)
    else:
        raise MappingError(f"target {pointer!r} does not have a writable parent")


def _remove_pointer(document: Any, pointer: str) -> None:
    parent, key = _parent(document, pointer)
    if isinstance(parent, dict) and key in parent:
        del parent[key]
    elif isinstance(parent, list) and key.isdigit() and int(key) < len(parent):
        del parent[int(key)]
    else:
        raise MappingError(f"source {pointer!r} does not exist")


def _cast(value: Any, target: str | None) -> Any:
    if target == "int":
        if isinstance(value, bool):
            raise MappingError("boolean values cannot be cast to int")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                converted = int(value)
            except ValueError as exc:
                raise MappingError("string is not an integer") from exc
            if str(converted) != value.strip():
                raise MappingError("integer cast would change the value")
            return converted
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif target == "float" and not isinstance(value, bool):
        converted = float(value)
        if converted == value:
            return converted
    elif target == "str" and isinstance(value, (str, int, float, bool)):
        return str(value)
    elif target == "bool" and isinstance(value, bool):
        return value
    raise MappingError(f"value cannot be safely cast to {target!r}")
