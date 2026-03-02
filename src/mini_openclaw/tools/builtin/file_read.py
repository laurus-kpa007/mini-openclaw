"""Built-in file read tool with sandbox restrictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_openclaw.core.errors import SecurityError
from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class FileReadTool:
    """Read file contents with sandbox path validation."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_read",
            description="Read the contents of a file at the given path.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path (absolute or relative to working directory)",
                ),
                ToolParameter(
                    name="offset",
                    type="integer",
                    description="Line number to start reading from (0-based)",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of lines to read",
                    required=False,
                    default=200,
                ),
            ],
            category="filesystem",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = arguments.get("path", "")
        offset = int(arguments.get("offset", 0))
        limit = int(arguments.get("limit", 200))

        if not file_path:
            return ToolResult(success=False, output="", error="No file path provided")

        # Resolve path
        resolved = Path(context.working_directory) / file_path
        resolved = resolved.resolve()

        # Sandbox check
        if context.sandbox_root:
            sandbox = Path(context.sandbox_root).resolve()
            if not str(resolved).startswith(str(sandbox)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access denied: path '{resolved}' is outside sandbox '{sandbox}'",
                )

        if not resolved.exists():
            return ToolResult(success=False, output="", error=f"File not found: {resolved}")

        if not resolved.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {resolved}")

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[offset : offset + limit]
            numbered = [f"{i + offset + 1:4d} | {line}" for i, line in enumerate(selected)]
            output = f"File: {resolved} ({len(lines)} lines total)\n"
            output += "\n".join(numbered)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Read error: {e}")
