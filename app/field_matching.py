"""Conservative field-name matching shared by diagnosis and guardrails."""

from collections.abc import Mapping, Set
import re
from typing import Any, Callable


def matching_sources(
    payload: Mapping[str, Any], target: str, implicated: Set[str],
    compatible: Callable[[Any], bool],
) -> list[tuple[str, Any]]:
    """Prefer unique, clearly related names; otherwise require a unique compatible value."""
    candidates = [
        (path, value) for path, value in value_paths(payload)
        if path and path not in implicated and compatible(value)
    ]
    target_leaf = _leaf(target)
    related = [(path, value) for path, value in candidates if _related(_leaf(path), target_leaf)]
    return related if related else candidates


def value_paths(payload: Any, pointer: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.extend(value_paths(child, f"{pointer}/{token}"))
    elif isinstance(payload, (list, tuple)):
        for index, child in enumerate(payload):
            result.extend(value_paths(child, f"{pointer}/{index}"))
    else:
        result.append((pointer, payload))
    return result


def _leaf(pointer: str) -> str:
    raw = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def _related(source: str, target: str) -> bool:
    if not source or not target:
        return False
    if source == target:
        return True
    # Permit common decorated names (user_email -> email) and small typos (emaill -> email).
    if source.endswith(target) or target.endswith(source):
        return True
    if len(source) < 3 or len(target) < 3:
        return False
    if abs(len(source) - len(target)) > 1:
        return False
    previous = list(range(len(target) + 1))
    for i, left in enumerate(source, 1):
        current = [i]
        for j, right in enumerate(target, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] <= 1
