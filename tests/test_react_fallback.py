"""Tests for ReAct prompt parsing."""

import pytest

from mini_openclaw.llm.react_fallback import (
    FINAL_ANSWER_PATTERN,
    ACTION_PATTERN,
    ACTION_INPUT_PATTERN,
    _format_tool_descriptions,
    ReActFallbackClient,
)
from mini_openclaw.llm.base import ChatResponse
from mini_openclaw.tools.base import ToolDefinition, ToolParameter


def test_final_answer_pattern():
    text = "Thought: I know the answer\nFinal Answer: Hello, world!"
    match = FINAL_ANSWER_PATTERN.search(text)
    assert match
    assert match.group(1).strip() == "Hello, world!"


def test_action_pattern():
    text = "Thought: I need to search\nAction: web_search\nAction Input: {\"query\": \"test\"}"
    action = ACTION_PATTERN.search(text)
    assert action
    assert action.group(1) == "web_search"

    action_input = ACTION_INPUT_PATTERN.search(text)
    assert action_input
    assert '"query"' in action_input.group(1)


def test_format_tool_descriptions():
    tools = [
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        ),
    ]
    desc = _format_tool_descriptions(tools)
    assert "test_tool" in desc
    assert "query" in desc


def test_parse_response_with_final_answer():
    """Test that parse_response correctly extracts Final Answer."""
    # Create a minimal mock to test parsing
    client = ReActFallbackClient.__new__(ReActFallbackClient)
    client._tools = []

    response = ChatResponse(
        content="Thought: I have the answer\nFinal Answer: The result is 42",
        tokens_used=100,
    )
    parsed = client._parse_response(response)
    assert parsed.content == "The result is 42"
    assert parsed.tool_calls is None


def test_parse_response_with_action():
    """Test that parse_response correctly extracts Action."""
    client = ReActFallbackClient.__new__(ReActFallbackClient)
    client._tools = []

    response = ChatResponse(
        content='Thought: I need to search\nAction: web_search\nAction Input: {"query": "test"}',
        tokens_used=100,
    )
    parsed = client._parse_response(response)
    assert parsed.tool_calls is not None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "web_search"
    assert parsed.tool_calls[0].arguments.get("query") == "test"


def test_parse_response_no_pattern():
    """Test fallback when no pattern matches."""
    client = ReActFallbackClient.__new__(ReActFallbackClient)
    client._tools = []

    response = ChatResponse(content="Just a plain response", tokens_used=50)
    parsed = client._parse_response(response)
    assert parsed.content == "Just a plain response"
    assert parsed.tool_calls is None
