"""Tests for the tool registry."""

import pytest

from mini_openclaw.tools.base import ToolContext, ToolDefinition, ToolParameter, ToolResult
from mini_openclaw.tools.registry import ToolRegistry
from mini_openclaw.tools.permissions import resolve_tools, resolve_child_tools


class DummyTool:
    def __init__(self, name: str = "dummy"):
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"A dummy tool named {self._name}",
            parameters=[
                ToolParameter(name="input", type="string", description="test input"),
            ],
        )

    async def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output=f"executed {self._name}")


def test_register_and_list():
    registry = ToolRegistry()
    registry.register(DummyTool("a"))
    registry.register(DummyTool("b"))
    assert set(registry.list_names()) == {"a", "b"}


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(DummyTool("x"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyTool("x"))


def test_unregister():
    registry = ToolRegistry()
    registry.register(DummyTool("y"))
    registry.unregister("y")
    assert "y" not in registry.list_names()


@pytest.mark.asyncio
async def test_execute():
    registry = ToolRegistry()
    registry.register(DummyTool("z"))
    ctx = ToolContext(session_id="s1", agent_id="a1")
    result = await registry.execute("z", {"input": "test"}, ctx)
    assert result.success
    assert "executed z" in result.output


def test_resolve_tools_allowlist():
    defs = [
        ToolDefinition(name="a", description="a"),
        ToolDefinition(name="b", description="b"),
        ToolDefinition(name="c", description="c"),
    ]
    filtered = resolve_tools(defs, allowlist=["a", "c"])
    names = [t.name for t in filtered]
    assert names == ["a", "c"]


def test_resolve_tools_denylist():
    defs = [
        ToolDefinition(name="a", description="a"),
        ToolDefinition(name="b", description="b"),
        ToolDefinition(name="spawn_agent", description="spawn"),
    ]
    filtered = resolve_tools(defs, denylist=["spawn_agent"])
    names = [t.name for t in filtered]
    assert "spawn_agent" not in names
    assert "a" in names


def test_resolve_child_tools_blocks_spawn():
    defs = [
        ToolDefinition(name="file_read", description="read"),
        ToolDefinition(name="spawn_agent", description="spawn"),
    ]
    filtered = resolve_child_tools(defs)
    names = [t.name for t in filtered]
    assert "spawn_agent" not in names
    assert "file_read" in names


def test_builtin_registration():
    registry = ToolRegistry()
    registry.register_builtin_tools()
    names = registry.list_names()
    assert "file_read" in names
    assert "shell_exec" in names
    assert "spawn_agent" in names
    assert "file_write" in names
    assert "web_search" in names
    assert "http_request" in names
