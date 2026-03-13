"""Tests for LM Studio client implementation."""

import json
import pytest
import httpx

from mini_openclaw.llm.lmstudio_client import LMStudioClient
from mini_openclaw.llm.base import ChatResponse, ToolCall


# ====================
# Response parsing tests
# ====================


class TestLMStudioResponseParsing:
    """Test OpenAI-format response parsing without network calls."""

    def setup_method(self):
        self.client = LMStudioClient(
            base_url="http://localhost:1234",
            model="test-model",
        )

    def test_parse_simple_response(self):
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        result = self.client._parse_response(data)
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello! How can I help you?"
        assert result.tool_calls is None
        assert result.tokens_used == 18
        assert result.finish_reason == "stop"

    def test_parse_response_with_tool_calls(self):
        data = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "Python tutorials"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }
        result = self.client._parse_response(data)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "Python tutorials"}
        assert result.tool_calls[0].call_id == "call_abc123"
        assert result.finish_reason == "tool_calls"
        assert result.tokens_used == 35

    def test_parse_multiple_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "file_read",
                                "arguments": '{"path": "/tmp/a.txt"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "file_read",
                                "arguments": '{"path": "/tmp/b.txt"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 50},
        }
        result = self.client._parse_response(data)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "file_read"
        assert result.tool_calls[1].arguments == {"path": "/tmp/b.txt"}

    def test_parse_empty_choices(self):
        data = {"choices": [], "usage": {}}
        result = self.client._parse_response(data)
        assert result.content == ""
        assert result.tokens_used == 0

    def test_parse_null_content(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 5},
        }
        result = self.client._parse_response(data)
        assert result.content == ""

    def test_parse_malformed_tool_arguments(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": "not valid json {{{",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 10},
        }
        result = self.client._parse_response(data)
        assert result.tool_calls is not None
        assert result.tool_calls[0].arguments == {"raw": "not valid json {{{"}

    def test_parse_dict_arguments(self):
        """LM Studio sometimes returns arguments as dict instead of string."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_y",
                        "type": "function",
                        "function": {
                            "name": "shell_exec",
                            "arguments": {"command": "ls -la"},
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 12},
        }
        result = self.client._parse_response(data)
        assert result.tool_calls[0].arguments == {"command": "ls -la"}

    def test_parse_usage_fallback(self):
        """When total_tokens is missing, sum prompt + completion."""
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = self.client._parse_response(data)
        assert result.tokens_used == 15


# ====================
# Message conversion tests
# ====================


class TestMessageConversion:
    def setup_method(self):
        self.client = LMStudioClient()

    def test_convert_simple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        converted = self.client._convert_messages(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[1]["content"] == "Hello"

    def test_convert_messages_with_tool_calls(self):
        """Ollama format tool_calls → OpenAI format (arguments as JSON string)."""
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "test"},
                },
            }],
        }]
        converted = self.client._convert_messages(messages)
        tc = converted[0]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        # Arguments should be JSON string in OpenAI format
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}

    def test_convert_messages_without_tool_calls(self):
        messages = [{"role": "user", "content": "Hi"}]
        converted = self.client._convert_messages(messages)
        assert "tool_calls" not in converted[0]


# ====================
# Tool format conversion tests
# ====================


class TestToolConversion:
    def setup_method(self):
        self.client = LMStudioClient()

    def test_convert_openai_format_passthrough(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "test",
                "description": "A test",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        converted = self.client._convert_tools(tools)
        assert converted == tools

    def test_convert_bare_function(self):
        tools = [{
            "name": "test",
            "description": "A test",
            "parameters": {"type": "object", "properties": {}},
        }]
        converted = self.client._convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "test"


# ====================
# Streaming tool buffer tests
# ====================


class TestStreamingToolBuffer:
    def setup_method(self):
        self.client = LMStudioClient()

    def test_flush_single_tool(self):
        buffer = {
            0: {
                "id": "call_1",
                "name": "web_search",
                "arguments": '{"query": "hello"}',
            }
        }
        tool_calls = self.client._flush_tool_buffer(buffer)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "web_search"
        assert tool_calls[0].arguments == {"query": "hello"}

    def test_flush_multiple_tools(self):
        buffer = {
            0: {"id": "c1", "name": "file_read", "arguments": '{"path": "a.txt"}'},
            1: {"id": "c2", "name": "file_read", "arguments": '{"path": "b.txt"}'},
        }
        tool_calls = self.client._flush_tool_buffer(buffer)
        assert len(tool_calls) == 2
        assert tool_calls[0].arguments == {"path": "a.txt"}
        assert tool_calls[1].arguments == {"path": "b.txt"}

    def test_flush_malformed_json(self):
        buffer = {0: {"id": "c1", "name": "test", "arguments": "broken{{"}}
        tool_calls = self.client._flush_tool_buffer(buffer)
        assert tool_calls[0].arguments == {"raw": "broken{{"}


# ====================
# Client instantiation tests
# ====================


class TestLMStudioClientInit:
    def test_default_config(self):
        client = LMStudioClient()
        assert client.base_url == "http://localhost:1234"
        assert client.model == "default"
        assert client.api_key == "lm-studio"

    def test_custom_config(self):
        client = LMStudioClient(
            base_url="http://192.168.1.100:5000",
            model="mistral-7b",
            temperature=0.3,
            api_key="my-key",
        )
        assert client.base_url == "http://192.168.1.100:5000"
        assert client.model == "mistral-7b"
        assert client.temperature == 0.3
        assert client.api_key == "my-key"

    def test_trailing_slash_stripped(self):
        client = LMStudioClient(base_url="http://localhost:1234/")
        assert client.base_url == "http://localhost:1234"


# ====================
# Config integration tests
# ====================


class TestLMStudioConfig:
    def test_config_defaults_to_ollama(self):
        from mini_openclaw.config import LLMConfig
        config = LLMConfig()
        assert config.provider == "ollama"

    def test_config_lmstudio_provider(self):
        from mini_openclaw.config import LLMConfig
        config = LLMConfig(
            provider="lmstudio",
            base_url="http://localhost:1234",
            model="qwen2.5-7b",
            api_key="test-key",
        )
        assert config.provider == "lmstudio"
        assert config.api_key == "test-key"

    def test_gateway_creates_lmstudio_client(self):
        from mini_openclaw.config import AppConfig
        from mini_openclaw.core.gateway import Gateway

        config = AppConfig(llm={"provider": "lmstudio", "base_url": "http://localhost:1234", "model": "test"})
        gw = Gateway(config)
        client = gw._create_llm_client()

        assert isinstance(client, LMStudioClient)
        assert client.model == "test"

    def test_gateway_creates_ollama_client(self):
        from mini_openclaw.config import AppConfig
        from mini_openclaw.core.gateway import Gateway
        from mini_openclaw.llm.ollama_client import OllamaClient

        config = AppConfig(llm={"provider": "ollama"})
        gw = Gateway(config)
        client = gw._create_llm_client()

        assert isinstance(client, OllamaClient)

    def test_gateway_rejects_unknown_provider(self):
        from mini_openclaw.config import AppConfig
        from mini_openclaw.core.gateway import Gateway

        config = AppConfig(llm={"provider": "unknown_provider"})
        gw = Gateway(config)

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            gw._create_llm_client()


# ====================
# ToolCallingAdapter with LMStudioClient
# ====================


class TestToolCallingAdapterWithLMStudio:
    def test_adapter_accepts_lmstudio_client(self):
        from mini_openclaw.llm.tool_calling import ToolCallingAdapter
        client = LMStudioClient()
        adapter = ToolCallingAdapter(llm_client=client)
        assert adapter._llm_client is client

    def test_adapter_accepts_ollama_client(self):
        from mini_openclaw.llm.tool_calling import ToolCallingAdapter
        from mini_openclaw.llm.ollama_client import OllamaClient
        client = OllamaClient()
        adapter = ToolCallingAdapter(llm_client=client)
        assert adapter._llm_client is client
