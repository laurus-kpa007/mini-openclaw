"""MCP (Model Context Protocol) client bridge for external tool servers."""

from __future__ import annotations

import logging
from typing import Any

from mini_openclaw.tools.base import Tool, ToolContext, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class MCPToolWrapper:
    """Wraps a single MCP server tool as a local Tool-protocol-compliant object."""

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        parameters: list[ToolParameter],
        session: Any,  # mcp.ClientSession
    ) -> None:
        self._server_name = server_name
        self._tool_name = tool_name
        self._session = session
        self._definition = ToolDefinition(
            name=f"mcp:{server_name}:{tool_name}",
            description=f"[MCP:{server_name}] {description}",
            parameters=parameters,
            category=f"mcp:{server_name}",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            result = await self._session.call_tool(self._tool_name, arguments)
            # MCP tool results have .content which is a list of content blocks
            output_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    output_parts.append(block.text)
                else:
                    output_parts.append(str(block))
            return ToolResult(
                success=not result.isError if hasattr(result, "isError") else True,
                output="\n".join(output_parts),
            )
        except Exception as e:
            logger.exception("MCP tool call failed: %s/%s", self._server_name, self._tool_name)
            return ToolResult(success=False, output="", error=f"MCP error: {e}")


class MCPToolBridge:
    """
    Bridges MCP server tools into the mini-openclaw ToolRegistry.
    Connects via stdio transport, discovers tools, and wraps them.
    """

    def __init__(self) -> None:
        self._exit_stack: Any = None
        self._sessions: dict[str, Any] = {}

    async def connect(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[Tool]:
        """
        Connect to an MCP server via stdio transport.
        Returns list of Tool objects wrapping the server's tools.
        """
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            if self._exit_stack is None:
                self._exit_stack = AsyncExitStack()

            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env,
            )

            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            self._sessions[server_name] = session

            # Discover tools
            tools_result = await session.list_tools()
            wrapped_tools: list[Tool] = []

            for mcp_tool in tools_result.tools:
                # Convert MCP tool schema to our ToolParameter list
                params: list[ToolParameter] = []
                if mcp_tool.inputSchema and "properties" in mcp_tool.inputSchema:
                    required = mcp_tool.inputSchema.get("required", [])
                    for pname, pschema in mcp_tool.inputSchema["properties"].items():
                        params.append(ToolParameter(
                            name=pname,
                            type=pschema.get("type", "string"),
                            description=pschema.get("description", ""),
                            required=pname in required,
                        ))

                wrapper = MCPToolWrapper(
                    server_name=server_name,
                    tool_name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    parameters=params,
                    session=session,
                )
                wrapped_tools.append(wrapper)

            logger.info(
                "Connected to MCP server '%s': %d tools",
                server_name, len(wrapped_tools),
            )
            return wrapped_tools

        except ImportError:
            logger.error(
                "MCP SDK not installed. Install with: pip install 'mini-openclaw[mcp]'"
            )
            return []
        except Exception:
            logger.exception("Failed to connect to MCP server '%s'", server_name)
            return []

    async def disconnect(self, server_name: str) -> None:
        """Disconnect a specific MCP server."""
        if server_name in self._sessions:
            del self._sessions[server_name]
            logger.info("Disconnected MCP server: %s", server_name)

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        self._sessions.clear()
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        logger.info("Disconnected all MCP servers")
