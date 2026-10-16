"""Thin HTTP proxy exposing registered tools through the healing executor."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app.config import load_local_env

load_local_env()

from app.executor import HealingExecutor
from app.registry import SchemaRegistry


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)


registry = SchemaRegistry()
executor = HealingExecutor(registry)


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


@app.post("/tools/{tool_name}/call")
async def call_tool(tool_name: str, call: ToolCall) -> Response:
    try:
        result = await executor.execute(tool_name, call.payload, headers=call.headers)
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
