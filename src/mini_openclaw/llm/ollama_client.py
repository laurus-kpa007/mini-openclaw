"""Ollama /api/chat client implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from mini_openclaw.core.errors import LLMConnectionError, LLMResponseError
from mini_openclaw.llm.base import ChatResponse, ChatStreamChunk, LLMClient, ToolCall

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Async client for Ollama's /api/chat endpoint via httpx."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._supports_tools: bool | None = None

    async def close(self) -> None:
        await self._http.aclose()

    async def check_tool_support(self) -> bool:
        """Probe whether the current model supports native tool calling."""
        if self._supports_tools is not None:
            return self._supports_tools

        test_tool = {
            "type": "function",
            "function": {
                "name": "_probe_tool_support",
                "description": "Test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                },
            },
        }
        try:
            resp = await self._http.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "tools": [test_tool],
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                # If the response has a message field without error, tools are supported
                self._supports_tools = "message" in data
            else:
                self._supports_tools = False
        except Exception:
            logger.debug("Tool support probe failed, assuming no support")
            self._supports_tools = False

        logger.info("Model %s tool calling support: %s", self.model, self._supports_tools)
        return self._supports_tools

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a non-streaming chat request to Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self._http.post("/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e

        if resp.status_code != 200:
            raise LLMResponseError(f"Ollama returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls")

        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(name=func.get("name", ""), arguments=args))

        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return ChatResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=tokens,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Send a streaming chat request to Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMResponseError(
                        f"Ollama returned status {resp.status_code}: {body.decode()}"
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = data.get("message", {})
                    done = data.get("done", False)
                    content = message.get("content", "")

                    tool_calls = None
                    raw_tc = message.get("tool_calls")
                    if raw_tc:
                        tool_calls = []
                        for tc in raw_tc:
                            func = tc.get("function", {})
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {"raw": args}
                            tool_calls.append(
                                ToolCall(name=func.get("name", ""), arguments=args)
                            )

                    yield ChatStreamChunk(
                        content=content,
                        tool_calls=tool_calls,
                        done=done,
                    )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e
