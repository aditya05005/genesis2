"""Thin HTTP proxy exposing registered tools through the healing executor."""

from __future__ import annotations

import os
import hmac
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from app.config import load_local_env

load_local_env()

from app.executor import HealingExecutor
from app.registry import SchemaRegistry


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)


class ToolRegistration(BaseModel):
    """User-supplied endpoint and JSON Schema contracts for a custom tool."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,62}$")
    url: AnyHttpUrl
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    request_schema: dict[str, Any]
    response_schema: dict[str, Any] | None = None
    idempotent: bool = False
    header_allowlist: dict[str, str] = Field(default_factory=dict)


registry = SchemaRegistry()
executor = HealingExecutor(registry)
_header_allowlists: dict[str, dict[str, str]] = {}


def configure_executor(value: HealingExecutor) -> None:
    """Replace the default executor for embedding or application startup."""
    global executor, registry
    executor = value
    registry = value.registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Register the example lookup tool when its upstream URL is configured."""
    upstream = os.environ.get("UPSTREAM_LOOKUP_USER_URL")
    if upstream:
        await registry.register(
            "lookup_user", upstream,
            {"type": "object", "required": ["user_id"],
             "properties": {"user_id": {"type": "string"}}},
            idempotent=True,
            response_schema={
                "type": "object", "required": ["ok", "record_id", "user_id"],
                "properties": {
                    "ok": {"type": "boolean"}, "record_id": {"type": "integer"},
                    "user_id": {"type": "string"},
                },
            },
        )
    yield


app = FastAPI(title="HealGate Proxy", lifespan=lifespan)


@app.post("/tools/register", status_code=201)
async def register_tool(
    registration: ToolRegistration,
    x_healgate_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Register a custom endpoint; admin registration requires a configured token."""
    expected_token = os.environ.get("HEALGATE_ADMIN_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Custom registration is disabled until HEALGATE_ADMIN_TOKEN is configured.",
        )
    if not x_healgate_admin_token or not hmac.compare_digest(
        x_healgate_admin_token, expected_token
    ):
        raise HTTPException(status_code=401, detail="Invalid HealGate admin token.")
    if registration.request_schema.get("type") != "object":
        raise HTTPException(status_code=422, detail="request_schema root type must be object.")
    if registration.response_schema is not None and not isinstance(
        registration.response_schema, dict
    ):
        raise HTTPException(status_code=422, detail="response_schema must be a JSON object.")
    try:
        existing = await registry.get(registration.tool_name)
    except KeyError:
        existing = None
    parsed_url = registration.url
    if parsed_url.username or parsed_url.password:
        raise HTTPException(status_code=422, detail="Do not put credentials in the endpoint URL.")
    if existing is not None:
        matches = (
            existing.url == str(parsed_url)
            and existing.method == registration.method
            and existing.schema == registration.request_schema
            and existing.response_schema == registration.response_schema
            and existing.idempotent == registration.idempotent
        )
        if not matches:
            raise HTTPException(
                status_code=409,
                detail=f"Tool {registration.tool_name!r} is already registered with different settings.",
            )
    else:
        existing = await registry.register(
            registration.tool_name, str(parsed_url), registration.request_schema,
            idempotent=registration.idempotent,
            response_schema=registration.response_schema,
            method=registration.method,
        )
        if not (
            existing.url == str(parsed_url)
            and existing.method == registration.method
            and existing.schema == registration.request_schema
            and existing.response_schema == registration.response_schema
            and existing.idempotent == registration.idempotent
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Tool {registration.tool_name!r} was registered concurrently with different settings.",
            )
    if registration.header_allowlist:
        _header_allowlists[registration.tool_name] = dict(registration.header_allowlist)
    else:
        _header_allowlists.pop(registration.tool_name, None)
    return {
        "tool_name": existing.name,
        "url": existing.url,
        "method": existing.method,
        "schema_version": existing.version,
        "response_schema_registered": existing.response_schema is not None,
    }


def _tool_executor(tool_name: str) -> HealingExecutor:
    """Build an executor with this tool's private, application-side header values."""
    allowed_headers = _header_allowlists.get(tool_name)
    if allowed_headers is None:
        return executor
    return HealingExecutor(
        executor.registry,
        timeout=executor.timeout,
        header_allowlist=allowed_headers,
        transport=executor.transport,
        audit_log=executor.audit_log,
        llm_diagnoser=executor.llm_diagnoser,
    )


@app.post("/tools/{tool_name}/call")
async def call_tool(tool_name: str, call: ToolCall) -> Response:
    try:
        result = await _tool_executor(tool_name).execute(
            tool_name, call.payload, headers=call.headers
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if result.escalation is not None:
        return JSONResponse(
            status_code=result.status_code,
            content=result.escalation.model_dump(mode="json"),
            headers={"X-HealGate-Outcome": "halted"},
        )
    if isinstance(result.body, (dict, list)):
        return JSONResponse(
            status_code=result.status_code,
            content=result.body,
            headers={"X-HealGate-Outcome": "healed" if result.healed else "passed"},
        )
    return PlainTextResponse(
        status_code=result.status_code,
        content="" if result.body is None else str(result.body),
        headers={"X-HealGate-Outcome": "healed" if result.healed else "passed"},
    )


@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str) -> JSONResponse:
    try:
        tool = await registry.get(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(
        content=tool.schema,
        headers={"X-HealGate-Schema-Version": str(tool.version)},
    )
