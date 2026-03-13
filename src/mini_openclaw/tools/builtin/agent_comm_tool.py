"""Tool for inter-agent communication - send/receive messages between agents."""

from __future__ import annotations

from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)


class AgentCommTool:
    """
    Allows agents to send messages to and receive messages from other agents.
    This enables coordination between sibling agents working on related tasks.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="agent_comm",
            description=(
                "Send or receive messages to/from other agents. "
                "Use 'send' action to message a specific agent or broadcast to siblings. "
                "Use 'receive' action to check your mailbox for incoming messages. "
                "Use 'send_to_parent' to report progress to your parent agent."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform: 'send', 'receive', 'send_to_parent', 'broadcast'",
                    enum=["send", "receive", "send_to_parent", "broadcast"],
                ),
                ToolParameter(
                    name="receiver_id",
                    type="string",
                    description="Target agent ID (required for 'send' action)",
                    required=False,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Message content (required for send actions)",
                    required=False,
                ),
                ToolParameter(
                    name="priority",
                    type="string",
                    description="Message priority: 'low', 'normal', 'high', 'urgent'",
                    required=False,
                    enum=["low", "normal", "high", "urgent"],
                ),
            ],
            category="orchestration",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Actual execution is handled by the Agent, similar to spawn_agent
        return ToolResult(
            success=False,
            output="",
            error="agent_comm must be handled by the Agent, not executed directly",
        )
