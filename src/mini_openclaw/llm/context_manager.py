"""Conversation history and token budget management."""

from __future__ import annotations

from typing import Any

from mini_openclaw.core.session import Message, MessageRole


class ContextManager:
    """Manages conversation history to stay within token budget."""

    def __init__(
        self,
        max_tokens: int = 8192,
        max_tool_output_chars: int = 4000,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_tool_output_chars = max_tool_output_chars

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count (rough: ~4 chars per token)."""
        return len(text) // 4 + 1

    def truncate_tool_output(self, output: str) -> str:
        """Truncate tool output to max_tool_output_chars."""
        if len(output) <= self.max_tool_output_chars:
            return output
        half = self.max_tool_output_chars // 2
        return (
            output[:half]
            + f"\n\n... [truncated {len(output) - self.max_tool_output_chars} chars] ...\n\n"
            + output[-half:]
        )

    def prepare_messages(
        self,
        system_prompt: str,
        history: list[Message],
    ) -> list[dict[str, Any]]:
        """Build the messages list for the LLM call, fitting within token budget."""
        messages: list[dict[str, Any]] = []

        # System prompt always included
        sys_msg = {"role": "system", "content": system_prompt}
        messages.append(sys_msg)
        used_tokens = self.estimate_tokens(system_prompt)

        # Reserve some tokens for the LLM response
        available = self.max_tokens - used_tokens - 1024

        # Walk history in reverse, collecting messages that fit
        selected: list[dict[str, Any]] = []
        for msg in reversed(history):
            content = msg.content
            if msg.role == MessageRole.TOOL:
                content = self.truncate_tool_output(content)

            msg_tokens = self.estimate_tokens(content)
            if msg_tokens > available:
                break
            available -= msg_tokens
            selected.append(msg.to_ollama_format() if msg.role != MessageRole.TOOL else {
                "role": "tool",
                "content": content,
            })

        # Reverse back to chronological order
        selected.reverse()
        messages.extend(selected)

        return messages
