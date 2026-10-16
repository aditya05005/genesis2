"""A direct tool-calling baseline without HealGate's repair or safety layers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.diff_engine import GeminiMappingDiagnoser


@dataclass
class BaselineResult:
    """Captured tool attempts and model usage from one baseline agent run."""

    calls: list[dict[str, Any]]
    tokens_used: int
    latency_ms: float
    final_message: str
    error: str | None = None


class LangChainBaseline:
    """Let a LangChain agent inspect tool errors and decide whether to retry."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_model_calls: int = 4,
        max_tool_calls: int = 3,
        chat_model: Any = None,
    ) -> None:
        if not api_key and chat_model is None:
            raise ValueError("a Gemini API key is required")
        if max_model_calls < 1 or max_tool_calls < 1:
            raise ValueError("call limits must be positive")
        self.api_key = api_key
        self.model = model
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.chat_model = chat_model

    @classmethod
    def from_env(cls, **kwargs: Any) -> "LangChainBaseline":
        """Use the same Gemini key and model selected for HealGate's Tier 2."""
        provider = GeminiMappingDiagnoser.from_env()
        return cls(api_key=provider.api_key, model=provider.model, **kwargs)

    async def run(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        task: str,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> BaselineResult:
        """Call the API through a LangChain tool; return errors to the agent."""
        # Imports stay local so core HealGate users do not need LangChain installed.
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
        from langchain.tools import tool

        from langchain_google_genai import ChatGoogleGenerativeAI

        requests: list[dict[str, Any]] = []
        allowed_headers = headers or {}

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), transport=transport
        ) as client:

            @tool
            async def lookup_user(
                request_payload: dict[str, Any], request_headers: dict[str, str]
            ) -> str:
                """Call the user lookup API with a JSON payload and headers."""
                response = await client.post(
                    url, json=request_payload, headers=request_headers
                )
                try:
                    body: Any = response.json()
                except ValueError:
                    body = response.text
                requests.append({
                    "payload": request_payload,
                    "headers": request_headers,
                    "status_code": response.status_code,
                    "body": body,
                })
                return json.dumps({"status_code": response.status_code, "body": body})

            model = self.chat_model or ChatGoogleGenerativeAI(
                model=self.model, api_key=self.api_key, temperature=0
            )
            agent = create_agent(
                model=model,
                tools=[lookup_user],
                system_prompt=(
                    "You are a tool-using agent. If the API rejects a request, read its "
                    "validation error and decide whether a corrected request can preserve "
                    "the user's intent using only the original input and configured header "
                    "values. Do not invent values. If a safe correction is unclear, stop."
                ),
                middleware=[
                    ModelCallLimitMiddleware(
                        run_limit=self.max_model_calls, exit_behavior="end"
                    ),
                    ToolCallLimitMiddleware(
                        tool_name="lookup_user", run_limit=self.max_tool_calls,
                        exit_behavior="end",
                    ),
                ],
            )
            prompt = (
                f"Task: {task}\n"
                f"Original request payload: {json.dumps(payload, sort_keys=True)}\n"
                f"Configured header values that may be used: "
                f"{json.dumps(allowed_headers, sort_keys=True)}\n"
                "Call lookup_user and inspect the result. You may correct and retry within "
                "the configured call limits."
            )
            started = time.perf_counter()
            error = None
            final_message = ""
            tokens_used = 0
            try:
                state = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
                messages = state.get("messages", [])
                if messages:
                    final_message = _message_text(messages[-1])
                for message in messages:
                    usage = getattr(message, "usage_metadata", None) or {}
                    tokens_used += int(usage.get("total_tokens", 0) or 0)
            except Exception as exc:
                # Keep provider errors out of the score report; they may contain request data.
                error = type(exc).__name__
            return BaselineResult(
                calls=requests,
                tokens_used=tokens_used,
                latency_ms=(time.perf_counter() - started) * 1000,
                final_message=final_message,
                error=error,
            )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)
