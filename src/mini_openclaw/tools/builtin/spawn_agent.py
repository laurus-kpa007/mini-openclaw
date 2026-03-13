"""The spawn_agent tool - enables Dynamic Agent Spawning."""

from __future__ import annotations

from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class SpawnAgentTool:
    """
    The core tool that enables Dynamic Agent Spawning.
    When the LLM decides it needs a specialized sub-agent, it calls this tool.
    Actual execution is handled by Agent._handle_spawn() - this tool definition
    just provides the schema for the LLM.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spawn_agent",
            description=(
                "Spawn a child agent to handle a specific sub-task. "
                "The child agent runs independently with its own tool set and "
                "returns a result when complete. Use this when a task requires "
                "specialized focus or a different set of tools. "
                "Examples: 'research a topic using web_search', "
                "'read and analyze files', 'execute a series of commands'."
            ),
            parameters=[
                ToolParameter(
                    name="task",
                    type="string",
                    description="Clear, detailed description of the task for the child agent",
                ),
                ToolParameter(
                    name="tools",
                    type="array",
                    description=(
                        "List of tool names the child should have access to. "
                        "Available tools: file_read, file_write, shell_exec, web_search, http_request. "
                        "If not specified, child gets all tools except spawn_agent."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="system_prompt",
                    type="string",
                    description="Optional custom system prompt for the child agent",
                    required=False,
                ),
                ToolParameter(
                    name="role",
                    type="string",
                    description=(
                        "Optional role template for the child agent. "
                        "Available roles: researcher, coder, reviewer, sysadmin, analyst, planner. "
                        "Roles provide optimized system prompts and tool sets."
                    ),
                    required=False,
                    enum=["researcher", "coder", "reviewer", "sysadmin", "analyst", "planner"],
                ),
            ],
            category="orchestration",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # This should never be called directly - Agent._handle_spawn() intercepts it
        return ToolResult(
            success=False,
            output="",
            error="spawn_agent must be handled by the Agent, not executed directly",
        )
