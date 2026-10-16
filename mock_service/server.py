"""A single endpoint whose request contract can change while it is running."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict


Mutation = Literal[
    "legacy",
    "rename",
    "type_change",
    "nesting",
    "profile_nesting",
    "required_header",
    "compound",
    "required_tenant",
    "auth_scope",
    "lossy_integer",
    "response_nested",
]
ErrorFormat = Literal["fastapi", "problem", "joi", "plain"]


class MutationRequest(BaseModel):
    """Settings accepted by the local admin endpoint."""

    model_config = ConfigDict(extra="forbid")

    mutation: Mutation
    error_format: ErrorFormat = "fastapi"


app = FastAPI(title="HealGate mock service")
_lock = asyncio.Lock()
_mutation: Mutation = "legacy"
_error_format: ErrorFormat = "fastapi"
_records: list[dict[str, Any]] = []


@app.exception_handler(HTTPException)
async def format_http_error(request: Request, exc: HTTPException) -> Any:
    """Use actual problem+json and plain-text bodies when those modes are selected."""
    media_type = (exc.headers or {}).get("content-type")
    if media_type == "application/problem+json":
        return JSONResponse(exc.detail, status_code=exc.status_code, media_type=media_type)
    if media_type == "text/plain":
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)
    return JSONResponse(
        {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
    )


def _validation_error(status: int, field: str, message: str) -> HTTPException:
    """Format an error like one of the common API validation responses."""
    return _validation_errors(status, [(field, message)])


def _validation_errors(status: int, errors: list[tuple[str, str]]) -> HTTPException:
    """Return all current request violations in the selected error format."""
    issues = [
        {"loc": ["body", field], "msg": message, "type": "value_error"}
        for field, message in errors
    ]
    if _error_format == "problem":
        detail: Any = {
            "type": "about:blank", "title": "Validation failed", "status": status,
            "detail": "; ".join(message for _, message in errors),
            "errors": [{"field": field, "message": message} for field, message in errors],
        }
        return HTTPException(status_code=status, detail=detail, headers={"content-type": "application/problem+json"})
    if _error_format == "joi":
        details = {"errors": [{"path": [field], "message": message} for field, message in errors]}
        return HTTPException(status_code=status, detail=details)
    if _error_format == "plain":
        detail = "; ".join(f"{field}: {message}" for field, message in errors)
        return HTTPException(status_code=status, detail=detail, headers={"content-type": "text/plain"})
    return HTTPException(status_code=status, detail=issues)


@app.post("/tool")
async def invoke_tool(
    payload: dict[str, Any],
    x_api_version: str | None = Header(default=None, alias="X-Api-Version"),
    x_scope: str | None = Header(default=None, alias="X-Scope"),
) -> dict[str, Any]:
    """Validate one call under the active contract and retain its semantic intent."""
    async with _lock:
        mutation = _mutation

    if mutation == "auth_scope" and x_scope != "accounts:write":
        raise HTTPException(status_code=403, detail="Required scope: accounts:write")
    if mutation == "compound":
        errors = []
        if x_api_version != "2026-01-01":
            errors.append(("X-Api-Version", "required header is missing or invalid"))
        if "account_id" not in payload:
            errors.append(("account_id", "field required; must be a string"))
        if errors:
            raise _validation_errors(422, errors)
    if mutation in ("required_header", "compound") and x_api_version != "2026-01-01":
        raise _validation_error(422, "X-Api-Version", "required header is missing or invalid")
    if mutation == "required_tenant" and "tenant_id" not in payload:
        raise _validation_error(422, "tenant_id", "field required")

    value: Any
    if mutation in ("rename", "compound"):
        if "account_id" not in payload:
            raise _validation_error(422, "account_id", "field required; must be a string")
        value = payload["account_id"]
    elif mutation == "nesting":
        target = payload.get("target")
        if not isinstance(target, dict) or "id" not in target:
            raise _validation_error(422, "target.id", "field required; must be an integer")
        value = target["id"]
    elif mutation == "profile_nesting":
        profile = payload.get("profile")
        if not isinstance(profile, dict) or "id" not in profile:
            raise _validation_error(422, "profile.id", "field required")
        value = profile["id"]
    else:
        if "user_id" not in payload:
            raise _validation_error(422, "user_id", "field required")
        value = payload["user_id"]

    if mutation in ("type_change", "nesting", "lossy_integer"):
        if isinstance(value, bool) or not isinstance(value, int):
            field = "target.id" if mutation == "nesting" else "user_id"
            raise _validation_error(422, field, "must be an integer")
    elif mutation in (
        "legacy", "rename", "required_header", "compound", "required_tenant",
        "auth_scope", "response_nested", "profile_nesting",
    ):
        if not isinstance(value, str):
            raise _validation_error(422, "user_id", "must be a string")

    if mutation == "lossy_integer" and value != int(value):
        raise _validation_error(422, "user_id", "must be a whole number")

    record = {
        "mutation": mutation,
        "received": payload.copy(),
        "received_headers": {"X-Api-Version": x_api_version} if x_api_version is not None else {},
        "ground_truth": {"user_id": str(value)},
    }
    async with _lock:
        _records.append(record)
        record_id = len(_records)
    if mutation == "response_nested":
        return {"ok": True, "record_id": record_id, "data": {"user": {"id": str(value)}}}
    return {"ok": True, "record_id": record_id, "user_id": str(value)}


@app.put("/admin/mutation")
async def set_mutation(settings: MutationRequest) -> dict[str, str]:
    """Switch the active contract and validation error style without restarting."""
    global _mutation, _error_format
    async with _lock:
        _mutation = settings.mutation
        _error_format = settings.error_format
    return {"mutation": _mutation, "error_format": _error_format}


@app.get("/admin/state")
async def get_state() -> dict[str, Any]:
    """Expose local demo state, including accepted calls and their expected intent."""
    async with _lock:
        return {"mutation": _mutation, "error_format": _error_format, "records": list(_records)}


@app.delete("/admin/records")
async def clear_records() -> dict[str, int]:
    """Clear recorded calls between harness runs."""
    async with _lock:
        count = len(_records)
        _records.clear()
    return {"deleted": count}
