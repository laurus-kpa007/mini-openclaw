"""Abstract LLM client interface and shared data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] | None = None
    tokens_used: int = 0
    finish_reason: str = "stop"


@dataclass
class ChatStreamChunk:
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    done: bool = False


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a chat request and return the complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Send a chat request and yield streaming chunks."""
        ...

    @abstractmethod
    async def check_tool_support(self) -> bool:
        """Check if the current model supports native tool calling."""
        ...
