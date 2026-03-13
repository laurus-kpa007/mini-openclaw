"""Tool for accessing shared memory between agents."""

from __future__ import annotations

from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class SharedMemoryTool:
    """
    Allows agents to store and retrieve shared state.
    Useful for passing findings between agents without going through the parent.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shared_memory",
            description=(
                "Store and retrieve data in a shared memory space accessible by related agents. "
                "Use 'put' to store a value, 'get' to retrieve it, 'list' to see available keys, "
                "or 'search' to find entries by tag. "
                "Example: store research findings for a sibling coding agent to use."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action: 'put', 'get', 'delete', 'list', 'search'",
                    enum=["put", "get", "delete", "list", "search"],
                ),
                ToolParameter(
                    name="key",
                    type="string",
                    description="Memory key (required for put/get/delete)",
                    required=False,
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="Value to store (required for 'put')",
                    required=False,
                ),
                ToolParameter(
                    name="tags",
                    type="array",
                    description="Tags for categorizing the entry (optional for 'put')",
                    required=False,
                ),
                ToolParameter(
                    name="tag",
                    type="string",
                    description="Tag to search for (required for 'search')",
                    required=False,
                ),
                ToolParameter(
                    name="access_level",
                    type="string",
                    description="Who can access: 'public' (all), 'family' (same tree), 'private' (only you). Default: 'family'",
                    required=False,
                    enum=["public", "family", "private"],
                ),
            ],
            category="orchestration",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Handled by Agent._handle_shared_memory()
        return ToolResult(
            success=False,
            output="",
            error="shared_memory must be handled by the Agent, not executed directly",
        )
