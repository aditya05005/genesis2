"""Extract short validation details from untrusted API error bodies."""

from collections.abc import Mapping
import json
from typing import Any


DEFAULT_MAX_CHARS = 2_000
_DETAIL_KEYS = (
    "detail", "errors", "message", "title", "type", "path", "loc", "msg", "field",
)


def sanitize_error(body: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return a bounded JSON excerpt containing recognized error fields only.

    This is data minimization, not an injection defense. Callers must still treat
    the result as untrusted text and validate any model output.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    if isinstance(body, Mapping):
        excerpt = _pick_fields(body)
    elif isinstance(body, (list, tuple)):
        excerpt = [_pick_fields(item) if isinstance(item, Mapping) else _safe_scalar(item) for item in body]
    elif isinstance(body, str):
        excerpt = body.strip()
    else:
        excerpt = _safe_scalar(body)

    try:
        rendered = json.dumps(excerpt, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = json.dumps(str(excerpt), ensure_ascii=True)
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 1] + "…"
    return rendered


def _pick_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep known error keys, recursively, and discard unrelated body fields."""
    result: dict[str, Any] = {}
    for key in _DETAIL_KEYS:
        if key in value:
            item = value[key]
            if isinstance(item, Mapping):
                result[key] = _pick_fields(item)
            elif isinstance(item, (list, tuple)):
                result[key] = [_pick_fields(x) if isinstance(x, Mapping) else _safe_scalar(x) for x in item]
            else:
                result[key] = _safe_scalar(item)
    return result


def _safe_scalar(value: Any) -> Any:
    """Keep JSON-like scalar data; stringify unusual objects without repr()."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
