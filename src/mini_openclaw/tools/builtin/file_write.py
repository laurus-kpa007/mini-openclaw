"""Built-in file write tool with sandbox restrictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class FileWriteTool:
    """Write or append to files with sandbox path validation."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write content to a file. Can create new files or overwrite/append to existing ones.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path (absolute or relative to working directory)",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                ),
                ToolParameter(
                    name="mode",
                    type="string",
                    description="Write mode: 'write' (overwrite) or 'append'",
                    required=False,
                    default="write",
                    enum=["write", "append"],
                ),
            ],
            category="filesystem",
            requires_approval=True,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = arguments.get("path", "")
        content = arguments.get("content", "")
        mode = arguments.get("mode", "write")

        if not file_path:
            return ToolResult(success=False, output="", error="No file path provided")

        resolved = Path(context.working_directory) / file_path
        resolved = resolved.resolve()

        # Sandbox check
        if context.sandbox_root:
            sandbox = Path(context.sandbox_root).resolve()
            if not str(resolved).startswith(str(sandbox)):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access denied: path '{resolved}' is outside sandbox",
                )

        try:
            # Ensure parent directory exists
            resolved.parent.mkdir(parents=True, exist_ok=True)

            file_mode = "a" if mode == "append" else "w"
            with open(resolved, file_mode, encoding="utf-8") as f:
                f.write(content)

            action = "appended to" if mode == "append" else "written to"
            return ToolResult(
                success=True,
                output=f"Successfully {action} {resolved} ({len(content)} chars)",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Write error: {e}")
