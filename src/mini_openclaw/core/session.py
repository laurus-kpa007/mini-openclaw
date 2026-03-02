"""Session isolation and context management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCallInfo:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: list[ToolCallInfo] | None = None
    tool_call_id: str | None = None  # For tool response messages
    agent_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_ollama_format(self) -> dict[str, Any]:
        """Convert to Ollama /api/chat message format."""
        msg: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass
class SandboxConfig:
    root_path: str | None = None
    allowed_hosts: list[str] | None = None
    allowed_shell_commands: list[str] | None = None
    blocked_shell_commands: list[str] = field(default_factory=list)


class Session:
    """Isolated execution context for an agent tree."""

    def __init__(
        self,
        session_id: str | None = None,
        model: str = "llama3.1:8b",
        max_turns: int = 50,
        token_budget: int = 8192,
        sandbox: SandboxConfig | None = None,
        user_id: str | None = None,
    ) -> None:
        self.session_id = session_id or f"session:{uuid.uuid4().hex[:12]}"
        self.model = model
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.sandbox = sandbox or SandboxConfig()
        self.user_id = user_id
        self.created_at = datetime.now(timezone.utc)
        self.conversation_history: list[Message] = []
        self.agent_ids: list[str] = []
        self.metadata: dict[str, Any] = {}

    def add_message(self, message: Message) -> None:
        self.conversation_history.append(message)

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """Return conversation history, optionally limited to last N messages."""
        if max_messages is None:
            return list(self.conversation_history)
        return list(self.conversation_history[-max_messages:])

    def clear_history(self) -> None:
        self.conversation_history.clear()
