"""Plugin system: @tool decorator and dynamic plugin loading."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

from mini_openclaw.tools.base import Tool, ToolContext, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Global list to collect tools registered via @tool decorator
_PLUGIN_TOOLS: list[Tool] = []


class FunctionTool:
    """Wraps a decorated async function as a Tool-protocol-compliant object."""

    def __init__(self, func: Callable, definition: ToolDefinition) -> None:
        self._func = func
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return await self._func(arguments, context)


def tool(
    name: str,
    description: str,
    parameters: list[dict[str, Any]] | None = None,
    category: str = "plugin",
    requires_approval: bool = False,
):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(
            name="my_tool",
            description="Does something useful",
            parameters=[
                {"name": "input", "type": "string", "description": "The input"},
            ],
        )
        async def my_tool(arguments: dict, context: ToolContext) -> ToolResult:
            value = arguments.get("input", "")
            return ToolResult(success=True, output=f"Processed: {value}")
    """
    def decorator(func: Callable) -> Callable:
        params = []
        if parameters:
            for p in parameters:
                params.append(ToolParameter(**p))

        defn = ToolDefinition(
            name=name,
            description=description,
            parameters=params,
            category=category,
            requires_approval=requires_approval,
        )
        wrapped = FunctionTool(func=func, definition=defn)
        _PLUGIN_TOOLS.append(wrapped)
        return func

    return decorator


class PluginLoader:
    """Scans directories for Python modules with @tool-decorated functions."""

    @staticmethod
    async def load_from_directory(directory: str | Path) -> list[Tool]:
        """
        Scan a directory for Python packages with tools.py files.
        Import each module, collect tools registered via @tool decorator.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.debug("Plugin directory does not exist: %s", directory)
            return []

        discovered: list[Tool] = []
        global _PLUGIN_TOOLS

        for item in directory.iterdir():
            if not item.is_dir():
                continue

            tools_file = item / "tools.py"
            init_file = item / "__init__.py"

            # Try tools.py first, then __init__.py
            module_file = tools_file if tools_file.exists() else (
                init_file if init_file.exists() else None
            )

            if module_file is None:
                continue

            module_name = f"plugins.{item.name}.tools" if module_file == tools_file else f"plugins.{item.name}"

            try:
                # Clear the collection before importing
                before_count = len(_PLUGIN_TOOLS)

                spec = importlib.util.spec_from_file_location(module_name, str(module_file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Collect newly registered tools
                    new_tools = _PLUGIN_TOOLS[before_count:]
                    discovered.extend(new_tools)
                    logger.info(
                        "Loaded %d tools from plugin '%s'",
                        len(new_tools), item.name,
                    )

            except Exception:
                logger.exception("Failed to load plugin from %s", item.name)

        return discovered
