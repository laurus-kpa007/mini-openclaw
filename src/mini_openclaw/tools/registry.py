"""Centralized tool registry owned by the Gateway."""

from __future__ import annotations

import logging
from typing import Any

from mini_openclaw.core.errors import ToolExecutionError, ToolNotFoundError
from mini_openclaw.tools.base import Tool, ToolContext, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized registry for all tools: built-in, plugin, and MCP.
    Agents access tools through the Gateway, never directly.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = tool
        logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            logger.info("Unregistered tool: %s", name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_definitions(self, category: str | None = None) -> list[ToolDefinition]:
        """List all tool definitions, optionally filtered by category."""
        defs = [t.definition for t in self._tools.values()]
        if category:
            defs = [d for d in defs if d.category == category]
        return defs

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")

        try:
            result = await tool.execute(arguments, context)
            return result
        except ToolNotFoundError:
            raise
        except Exception as e:
            logger.exception("Tool '%s' execution failed", tool_name)
            raise ToolExecutionError(f"Tool '{tool_name}' failed: {e}") from e

    def register_builtin_tools(self) -> None:
        """Register all built-in tools."""
        from mini_openclaw.tools.builtin.file_read import FileReadTool
        from mini_openclaw.tools.builtin.file_write import FileWriteTool
        from mini_openclaw.tools.builtin.http_request import HttpRequestTool
        from mini_openclaw.tools.builtin.pip_install import PipInstallTool
        from mini_openclaw.tools.builtin.python_exec import PythonExecTool
        from mini_openclaw.tools.builtin.shell_exec import ShellExecTool
        from mini_openclaw.tools.builtin.spawn_agent import SpawnAgentTool
        from mini_openclaw.tools.builtin.web_search import WebSearchTool

        builtin_classes = [
            FileReadTool,
            FileWriteTool,
            ShellExecTool,
            WebSearchTool,
            HttpRequestTool,
            PipInstallTool,
            PythonExecTool,
            SpawnAgentTool,
        ]
        for tool_cls in builtin_classes:
            tool = tool_cls()
            if tool.definition.name not in self._tools:
                self.register(tool)
