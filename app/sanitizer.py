"""Minimize and classify untrusted API errors before diagnosis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from typing import Any


DEFAULT_MAX_CHARS = 2_000
_DETAIL_KEYS = (
    "detail", "errors", "message", "title", "type", "path", "loc", "msg", "field",
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_INJECTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|rules?|prompts?)\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\b",
    r"\b(?:call|invoke|run|use)\s+(?:the\s+)?(?:tool|function|browser|shell|terminal)\b",
    r"\b(?:open|visit|fetch|navigate\s+to|request)\s+(?:the\s+)?(?:url\b|https?://)",
    r"https?://[^\s\]\[<>]+",
))


@dataclass(frozen=True)
class SanitizedError:
    """Bounded sanitized text plus its prompt-injection screening result."""

    text: str
    injection_flagged: bool
    excerpt: Any


def sanitize_error_details(body: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> SanitizedError:
    """Filter recognized error fields, strip controls, truncate, and flag directives."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    if isinstance(body, Mapping):
        excerpt = _pick_fields(body)
    elif isinstance(body, (list, tuple)):
        excerpt = [
            _pick_fields(item) if isinstance(item, Mapping) else _safe_scalar(item)
            for item in body
        ]
    elif isinstance(body, str):
        excerpt = _clean_text(body.strip())
    else:
        excerpt = _safe_scalar(body)

    try:
        rendered = json.dumps(excerpt, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = json.dumps(_clean_text(str(excerpt)), ensure_ascii=True)
    flagged = any(pattern.search(rendered) for pattern in _INJECTION_PATTERNS)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] if max_chars <= 3 else rendered[:max_chars - 3] + "..."
    return SanitizedError(text=rendered, injection_flagged=flagged, excerpt=excerpt)


def sanitize_error(body: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Compatibility helper returning only sanitized text."""
    return sanitize_error_details(body, max_chars=max_chars).text


def _pick_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep known error keys recursively and discard unrelated response fields."""
    result: dict[str, Any] = {}
    for key in _DETAIL_KEYS:
        if key in value:
            item = value[key]
            if isinstance(item, Mapping):
                result[key] = _pick_fields(item)
            elif isinstance(item, (list, tuple)):
                result[key] = [
                    _pick_fields(child) if isinstance(child, Mapping) else _safe_scalar(child)
                    for child in item
                ]
            else:
                result[key] = _safe_scalar(item)
    return result


def _clean_text(value: str) -> str:
    """Replace control characters with spaces so words cannot be joined."""
    return _CONTROL_CHARS.sub(" ", value)


def _safe_scalar(value: Any) -> Any:
    """Keep JSON-like scalar data while removing control characters from strings."""
    if isinstance(value, str):
        return _clean_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clean_text(str(value))
