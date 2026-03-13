"""Auto-detecting adapter: native tool calling or ReAct fallback."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from mini_openclaw.llm.base import ChatResponse, ChatStreamChunk, LLMClient, ToolCall
from mini_openclaw.llm.react_fallback import ReActFallbackClient
from mini_openclaw.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class ToolCallingAdapter(LLMClient):
    """
    Transparent adapter that auto-detects whether to use native tool calling
    or ReAct prompt-based fallback.

    On first call, probes the model for tool support and caches the result.
    Agent instances use this class - they never interact with OllamaClient,
    LMStudioClient, or ReActFallbackClient directly.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: list[ToolDefinition] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tools = tools or []
        self._delegate: LLMClient | None = None
        self._resolved = False

    async def _resolve_delegate(self) -> LLMClient:
        """Determine and cache the appropriate delegate."""
        if self._delegate is not None:
            return self._delegate

        supports_native = await self._llm_client.check_tool_support()
        if supports_native:
            logger.info("Using native tool calling")
            self._delegate = self._llm_client
        else:
            logger.info("Model does not support native tools, using ReAct fallback")
            self._delegate = ReActFallbackClient(self._llm_client, self._tools)

        self._resolved = True
        return self._delegate

    async def check_tool_support(self) -> bool:
        delegate = await self._resolve_delegate()
        return not isinstance(delegate, ReActFallbackClient)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        delegate = await self._resolve_delegate()
        return await delegate.chat(messages, tools)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        delegate = await self._resolve_delegate()
        async for chunk in delegate.chat_stream(messages, tools):
            yield chunk
