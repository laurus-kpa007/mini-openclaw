"""ReAct prompt-based tool calling fallback for models without native support."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from mini_openclaw.llm.base import ChatResponse, ChatStreamChunk, LLMClient, ToolCall
from mini_openclaw.llm.ollama_client import OllamaClient
from mini_openclaw.tools.base import ToolDefinition

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """You have access to the following tools:

{tool_descriptions}

To use a tool, you MUST respond with EXACTLY this format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with the arguments]

After receiving the tool result, continue reasoning.

When you have a final answer and no more tools to call, respond with:

Thought: [Your reasoning]
Final Answer: [Your complete response to the user]

IMPORTANT: You must ALWAYS start with "Thought:" and use either "Action:"+"Action Input:" or "Final Answer:".
"""

# Patterns to extract ReAct components
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", re.DOTALL)
ACTION_PATTERN = re.compile(r"Action:\s*(\S+)")
ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(.+?)(?=\nThought:|\nAction:|\nFinal Answer:|\Z)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _format_tool_descriptions(tools: list[ToolDefinition]) -> str:
    """Format tool definitions into human-readable descriptions for the prompt."""
    parts = []
    for tool in tools:
        params_desc = []
        for p in tool.parameters:
            req = "(required)" if p.required else "(optional)"
            params_desc.append(f"    - {p.name} ({p.type}, {req}): {p.description}")
        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"
        parts.append(f"- {tool.name}: {tool.description}\n  Parameters:\n{params_str}")
    return "\n\n".join(parts)


class ReActFallbackClient(LLMClient):
    """
    Wraps an OllamaClient to provide tool calling via ReAct prompt engineering.
    Parses the LLM's text output to extract Thought/Action/Action Input/Final Answer.
    """

    def __init__(self, ollama_client: OllamaClient, tools: list[ToolDefinition]) -> None:
        self._ollama = ollama_client
        self._tools = tools
        self._react_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=_format_tool_descriptions(tools)
        )

    async def check_tool_support(self) -> bool:
        return False  # This IS the fallback

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send message with ReAct system prompt, parse response for tool calls."""
        # Replace or prepend system prompt with ReAct version
        react_messages = self._inject_react_prompt(messages)

        # Call Ollama WITHOUT tools parameter (raw text mode)
        response = await self._ollama.chat(react_messages, tools=None)
        return self._parse_response(response)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Stream not fully supported for ReAct - collect and parse."""
        react_messages = self._inject_react_prompt(messages)
        full_content = ""
        async for chunk in self._ollama.chat_stream(react_messages, tools=None):
            full_content += chunk.content
            yield chunk

    def _inject_react_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepend/replace system message with ReAct prompt."""
        result = []
        has_system = False

        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                result.append({
                    "role": "system",
                    "content": self._react_prompt + "\n\n" + msg.get("content", ""),
                })
            else:
                result.append(msg)

        if not has_system:
            result.insert(0, {"role": "system", "content": self._react_prompt})

        return result

    def _parse_response(self, response: ChatResponse) -> ChatResponse:
        """Parse LLM text for ReAct patterns and extract tool calls."""
        text = response.content.strip()

        # Check for Final Answer
        final_match = FINAL_ANSWER_PATTERN.search(text)
        if final_match and not ACTION_PATTERN.search(text):
            return ChatResponse(
                content=final_match.group(1).strip(),
                tool_calls=None,
                tokens_used=response.tokens_used,
                finish_reason="stop",
            )

        # Check for Action + Action Input
        action_match = ACTION_PATTERN.search(text)
        if action_match:
            tool_name = action_match.group(1).strip()
            input_match = ACTION_INPUT_PATTERN.search(text)

            arguments = {}
            if input_match:
                raw_input = input_match.group(1).strip()
                try:
                    arguments = json.loads(raw_input)
                except json.JSONDecodeError:
                    # Try to extract key-value pairs from non-JSON format
                    arguments = {"raw_input": raw_input}

            # Extract thought for context
            thought = ""
            thought_match = THOUGHT_PATTERN.search(text)
            if thought_match:
                thought = thought_match.group(1).strip()

            return ChatResponse(
                content=thought,
                tool_calls=[ToolCall(name=tool_name, arguments=arguments)],
                tokens_used=response.tokens_used,
                finish_reason="tool_calls",
            )

        # No pattern matched - treat entire response as final answer
        return ChatResponse(
            content=text,
            tool_calls=None,
            tokens_used=response.tokens_used,
            finish_reason="stop",
        )
