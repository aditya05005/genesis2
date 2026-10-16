"""Request a user-supplied HTTPS API and optionally propose one gated repair."""

from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.adapter import MappingError, apply_mapping
from app.diff_engine import GeminiMappingDiagnoser, diagnose_error, extract_implicated_paths
from app.executor import _dry_run
from app.gate import evaluate_mapping
from app.sanitizer import sanitize_error
from app.triage import Action, action_for_status


SENSITIVE_HEADER_NAMES = {"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"}


def _validate_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Enter a complete HTTPS endpoint URL.")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local and internal hostnames are not allowed in External API mode.")
    try:
        address = ipaddress.ip_address(host)
        if not address.is_global:
            raise ValueError("Private and reserved IP addresses are not allowed.")
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise
    if parsed.username or parsed.password:
        raise ValueError("Put credentials in Headers, not in the URL.")
    if parsed.port not in (None, 443):
        raise ValueError("External API mode accepts HTTPS on port 443 only.")
    return url.strip()


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: ("[redacted]" if key.lower() in SENSITIVE_HEADER_NAMES else value)
        for key, value in headers.items()
    }


def _request_kwargs(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"params": payload} if method == "GET" else {"json": payload}


async def _send(client: httpx.AsyncClient, method: str, url: str, payload: dict[str, Any], headers: dict[str, str]):
    return await client.request(method, url, headers=headers, **_request_kwargs(method, payload))


def _decode(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _execute(
    method: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    use_gemini: bool,
    heal_enabled: bool,
) -> dict[str, Any]:
    url = _validate_url(url)
    method = method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError("Choose GET or POST.")
    if not isinstance(payload, dict):
        raise ValueError("Request data must be a JSON object.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        raise ValueError("Headers must be a JSON object with string values.")

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=False) as client:
        first = await _send(client, method, url, payload, headers)
        first_body = _decode(first)
        safe_headers = _safe_headers(headers)
        result: dict[str, Any] = {
            "method": method,
            "status_code": first.status_code,
            "body": first_body,
            "attempts": 1,
            "healed": False,
            "original_payload": payload,
            "original_headers": safe_headers,
            "initial_status_code": first.status_code,
            "initial_error": None if first.is_success else sanitize_error(first_body),
            "diagnosis": None,
            "gate": None,
            "transformed_payload": None,
            "transformed_headers": None,
            "escalation": None,
            "tokens_used": 0,
        }
        if first.is_success:
            result["outcome"] = "SUCCEEDED"
            return result

        if not heal_enabled:
            result["outcome"] = "BASELINE · NO REPAIR"
            return result

        action = action_for_status(first.status_code)
        if action not in (Action.HEAL, Action.HEAL_RULES_ONLY):
            result["outcome"] = "SAFE STOP"
            result["escalation"] = f"HTTP {first.status_code} is not eligible for automatic repair."
            return result

        diagnosis = diagnose_error(first_body, payload, header_allowlist={})
        tokens_used = 0
        if diagnosis is None and use_gemini:
            try:
                diagnosis, tokens_used = await GeminiMappingDiagnoser.from_env().diagnose(
                    first_body, payload, header_allowlist={}
                )
            except Exception:
                result["outcome"] = "SAFE STOP"
                result["escalation"] = "Gemini diagnosis is unavailable; the request was not replayed. Try again later or review the proposed change manually."
                return result
        result["tokens_used"] = tokens_used
        result["diagnosis"] = diagnosis.model_dump() if diagnosis else None
        if diagnosis is None or not diagnosis.ops:
            result["outcome"] = "SAFE STOP"
            result["escalation"] = "No supported repair was found from the API error."
            return result

        verdict = evaluate_mapping(
            diagnosis.ops, payload,
            implicated_paths=extract_implicated_paths(first_body),
            header_allowlist={},
        )
        result["gate"] = verdict.model_dump()
        if not verdict.allowed:
            result["outcome"] = "SAFE STOP"
            result["escalation"] = "Safety gate rejected the proposed request change."
            return result

        try:
            changed_payload, changed_headers = apply_mapping(
                payload, headers, diagnosis.ops, header_allowlist={}
            )
            _dry_run(changed_payload, changed_headers, diagnosis.ops, {})
        except (MappingError, ValueError, TypeError) as exc:
            result["outcome"] = "SAFE STOP"
            result["gate"] = {"allowed": False, "reasons": [str(exc)], "evaluated": True}
            result["escalation"] = "Dry-run validation rejected the proposed change."
            return result

        result["transformed_payload"] = changed_payload
        result["transformed_headers"] = _safe_headers(changed_headers)
        result["outcome"] = "REPAIR READY · REVIEW REQUIRED"
        result["escalation"] = "Review the proposed request. It has not been sent yet."
        return result


def run_external(
    method: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    use_gemini: bool = False,
    heal_enabled: bool = True,
) -> dict[str, Any]:
    """Send the original request once and prepare, but never send, a repair."""
    return asyncio.run(_execute(
        method, url, payload, headers,
        use_gemini=use_gemini,
        heal_enabled=heal_enabled,
    ))


async def _send_corrected_async(
    method: str, url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    """Send one user-approved corrected request, with no further retry."""
    url = _validate_url(url)
    method = method.upper()
    if method not in {"GET", "POST"}:
        raise ValueError("Choose GET or POST.")
    if not isinstance(payload, dict):
        raise ValueError("Corrected request data must be a JSON object.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        raise ValueError("Headers must be a JSON object with string values.")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=False) as client:
        response = await _send(client, method, url, payload, headers)
    body = _decode(response)
    return {
        "method": method,
        "status_code": response.status_code,
        "body": body,
        "attempts": 1,
        "outcome": "SENT · SUCCESS" if response.is_success else "SENT · API ERROR",
        "healed": response.is_success,
    }


def send_corrected(
    method: str, url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    """Send the reviewed correction once after an explicit user action."""
    return asyncio.run(_send_corrected_async(method, url, payload, headers))
