"""LM Studio client implementation using OpenAI-compatible /v1/chat/completions API."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from mini_openclaw.core.errors import LLMConnectionError, LLMResponseError
from mini_openclaw.llm.base import ChatResponse, ChatStreamChunk, LLMClient, ToolCall

logger = logging.getLogger(__name__)


class LMStudioClient(LLMClient):
    """
    Async client for LM Studio's OpenAI-compatible API.

    LM Studio serves models at /v1/chat/completions using the same
    request/response schema as OpenAI's Chat Completions API.
    Default endpoint: http://localhost:1234/v1
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "default",
        temperature: float = 0.7,
        timeout: float = 120.0,
        api_key: str = "lm-studio",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
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
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "tools": [test_tool],
                    "max_tokens": 50,
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    # If tool_calls field exists in response, tools are supported
                    self._supports_tools = "tool_calls" in msg or data.get("choices", [{}])[0].get("finish_reason") == "tool_calls"
                else:
                    self._supports_tools = False
            else:
                # Some models return 400 if tools not supported
                self._supports_tools = False
        except Exception:
            logger.debug("Tool support probe failed, assuming no support")
            self._supports_tools = False

        logger.info("LM Studio model %s tool calling support: %s", self.model, self._supports_tools)
        return self._supports_tools

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a non-streaming chat request to LM Studio."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            # Convert Ollama tool format to OpenAI format if needed
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            resp = await self._http.post("/v1/chat/completions", json=payload)
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. Is LM Studio running?"
            ) from e

        if resp.status_code != 200:
            raise LLMResponseError(f"LM Studio returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        return self._parse_response(data)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Send a streaming chat request to LM Studio."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            async with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMResponseError(
                        f"LM Studio returned status {resp.status_code}: {body.decode()}"
                    )

                # Accumulated tool call state for streaming
                tool_call_buffer: dict[int, dict[str, Any]] = {}

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        # Flush any accumulated tool calls
                        if tool_call_buffer:
                            tool_calls = self._flush_tool_buffer(tool_call_buffer)
                            yield ChatStreamChunk(content="", tool_calls=tool_calls, done=True)
                        else:
                            yield ChatStreamChunk(content="", done=True)
                        return
                    if not line.startswith("data: "):
                        continue

                    json_str = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    # Handle streaming tool calls
                    delta_tool_calls = delta.get("tool_calls")
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            idx = tc.get("index", 0)
                            if idx not in tool_call_buffer:
                                tool_call_buffer[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            func = tc.get("function", {})
                            if "name" in func:
                                tool_call_buffer[idx]["name"] = func["name"]
                            if "arguments" in func:
                                tool_call_buffer[idx]["arguments"] += func["arguments"]

                    if content:
                        yield ChatStreamChunk(
                            content=content,
                            done=finish_reason == "stop",
                        )

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. Is LM Studio running?"
            ) from e

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Ollama-format messages to OpenAI-format messages.
        Main difference: Ollama uses 'tool_calls' with 'function.arguments' as dict,
        OpenAI expects 'function.arguments' as JSON string.
        """
        converted = []
        for msg in messages:
            new_msg: dict[str, Any] = {
                "role": msg["role"],
                "content": msg.get("content", ""),
            }

            # Convert tool_calls if present (assistant messages)
            if "tool_calls" in msg and msg["tool_calls"]:
                new_msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    new_msg["tool_calls"].append({
                        "id": tc.get("id", f"call_{id(tc)}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args_str,
                        },
                    })

            converted.append(new_msg)
        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Ensure tools are in OpenAI format.
        Ollama and OpenAI use the same top-level format, but ensure
        we have the 'type': 'function' wrapper.
        """
        converted = []
        for tool in tools:
            if "type" in tool and "function" in tool:
                # Already in OpenAI format
                converted.append(tool)
            elif "function" in tool:
                converted.append({"type": "function", **tool})
            else:
                # Assume it's a bare function definition
                converted.append({"type": "function", "function": tool})
        return converted

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        """Parse OpenAI-format response into ChatResponse."""
        choices = data.get("choices", [])
        if not choices:
            return ChatResponse(content="", tokens_used=0)

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Parse tool calls
        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args_raw = func.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                elif isinstance(args_raw, dict):
                    args = args_raw
                else:
                    args = {}
                tool_calls.append(ToolCall(
                    name=func.get("name", ""),
                    arguments=args,
                    call_id=tc.get("id", ""),
                ))

        # Parse token usage
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        if not tokens_used:
            tokens_used = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        return ChatResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            tokens_used=tokens_used,
            finish_reason="tool_calls" if tool_calls else finish_reason,
        )

    def _flush_tool_buffer(self, buffer: dict[int, dict[str, Any]]) -> list[ToolCall]:
        """Convert accumulated streaming tool call fragments into ToolCall objects."""
        tool_calls = []
        for idx in sorted(buffer.keys()):
            tc = buffer[idx]
            args_str = tc["arguments"]
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}
            tool_calls.append(ToolCall(
                name=tc["name"],
                arguments=args,
                call_id=tc.get("id", ""),
            ))
        return tool_calls
