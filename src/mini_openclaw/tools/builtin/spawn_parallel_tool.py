"""Tool for spawning multiple agents in parallel with result aggregation."""

from __future__ import annotations

from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class SpawnParallelTool:
    """
    Spawn multiple child agents in parallel and aggregate their results.
    More powerful than spawn_agent for tasks that benefit from parallelism.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spawn_parallel",
            description=(
                "Spawn multiple child agents in parallel to handle sub-tasks concurrently. "
                "Each task gets its own agent. Results are aggregated based on the chosen strategy. "
                "Strategies: 'wait_all' (combine all results), 'first_success' (return first good result), "
                "'majority_vote' (return most common answer), 'priority_chain' (try in order). "
                "Example: spawn 3 researchers to investigate different aspects of a topic simultaneously."
            ),
            parameters=[
                ToolParameter(
                    name="tasks",
                    type="array",
                    description=(
                        "List of task objects, each with 'task' (description string) "
                        "and optional 'tools' (list of tool names). "
                        "Example: [{'task': 'research topic A', 'tools': ['web_search']}, "
                        "{'task': 'research topic B'}]"
                    ),
                ),
                ToolParameter(
                    name="strategy",
                    type="string",
                    description="Aggregation strategy: 'wait_all', 'first_success', 'majority_vote', 'priority_chain'",
                    required=False,
                    enum=["wait_all", "first_success", "majority_vote", "priority_chain"],
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds for the entire operation (optional)",
                    required=False,
                ),
            ],
            category="orchestration",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Handled by Agent._handle_spawn_parallel()
        return ToolResult(
            success=False,
            output="",
            error="spawn_parallel must be handled by the Agent, not executed directly",
        )
