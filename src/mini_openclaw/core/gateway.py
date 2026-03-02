"""Central Gateway orchestrator for mini-openclaw."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from mini_openclaw.config import AppConfig
from mini_openclaw.core.agent import Agent, AgentResult
from mini_openclaw.core.errors import (
    AgentConcurrencyError,
    AgentDepthLimitError,
    AgentError,
)
from mini_openclaw.core.events import Event, EventBus, EventType
from mini_openclaw.core.session import Message, MessageRole, SandboxConfig, Session
from mini_openclaw.llm.ollama_client import OllamaClient
from mini_openclaw.tools.base import ToolDefinition
from mini_openclaw.tools.permissions import resolve_child_tools
from mini_openclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

CHILD_SYSTEM_PROMPT = """You are a specialized sub-agent spawned to handle a specific task.

TASK: {task}

INSTRUCTIONS:
- Focus exclusively on the assigned task.
- Use the available tools to gather information and produce results.
- When the task is complete, provide a clear, concise summary of your findings/actions.
- Do not attempt tasks outside your scope.

Available tools: {tool_names}
"""


class Gateway:
    """
    Central daemon that orchestrates all system components.
    Manages sessions, agents, tools, and event routing.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tool_registry = ToolRegistry()
        self.event_bus = EventBus()
        self.sessions: dict[str, Session] = {}
        self.agents: dict[str, Agent] = {}
        self._semaphore = asyncio.Semaphore(config.gateway.max_concurrent_agents)
        self._ollama: OllamaClient | None = None
        self._mcp_bridge: Any = None

        # HITL approval manager
        from mini_openclaw.core.hitl import HITLManager
        self.hitl = HITLManager()

    async def start(self) -> None:
        """Initialize the Gateway: connect to Ollama, register tools, load plugins."""
        logger.info("Starting Gateway...")

        # Initialize Ollama client
        self._ollama = OllamaClient(
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            timeout=self.config.llm.timeout,
        )

        # Register built-in tools
        self.tool_registry.register_builtin_tools()

        # Load plugins
        from mini_openclaw.tools.plugin_loader import PluginLoader
        for plugin_dir in self.config.tools.plugin_dirs:
            plugins = await PluginLoader.load_from_directory(plugin_dir)
            for plugin_tool in plugins:
                try:
                    self.tool_registry.register(plugin_tool)
                except ValueError:
                    logger.warning("Plugin tool '%s' already registered", plugin_tool.definition.name)

        # Connect MCP servers
        if self.config.tools.mcp_servers:
            from mini_openclaw.tools.mcp_client import MCPToolBridge
            self._mcp_bridge = MCPToolBridge()
            for server_cfg in self.config.tools.mcp_servers:
                mcp_tools = await self._mcp_bridge.connect(
                    server_name=server_cfg["name"],
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                )
                for mcp_tool in mcp_tools:
                    self.tool_registry.register(mcp_tool)

        logger.info(
            "Registered %d tools: %s",
            len(self.tool_registry.list_names()),
            ", ".join(self.tool_registry.list_names()),
        )

        logger.info("Gateway started (model=%s)", self.config.llm.model)

    async def shutdown(self) -> None:
        """Gracefully shut down: terminate agents, close connections."""
        logger.info("Shutting down Gateway...")

        # Cancel pending HITL approvals
        self.hitl.cancel_all()

        # Terminate all agents
        for agent_id in list(self.agents.keys()):
            await self.terminate_agent(agent_id, cascade=True)

        # Disconnect MCP servers
        if self._mcp_bridge:
            await self._mcp_bridge.disconnect_all()

        # Close Ollama client
        if self._ollama:
            await self._ollama.close()

        logger.info("Gateway shut down")

    def create_session(self, user_id: str | None = None) -> Session:
        """Create a new isolated session."""
        sandbox = SandboxConfig(
            root_path=self.config.security.sandbox_root,
            allowed_hosts=self.config.security.allowed_hosts,
            blocked_shell_commands=self.config.security.blocked_shell_commands,
        )
        session = Session(
            model=self.config.llm.model,
            max_turns=self.config.session.max_turns,
            token_budget=self.config.session.default_token_budget,
            sandbox=sandbox,
            user_id=user_id,
        )
        self.sessions[session.session_id] = session

        self.event_bus.emit_nowait(Event(
            type=EventType.SESSION_CREATED,
            source_id=session.session_id,
        ))
        logger.info("Created session: %s", session.session_id)
        return session

    async def spawn_agent(
        self,
        session_id: str,
        system_prompt: str | None = None,
        tool_allowlist: list[str] | None = None,
        tool_denylist: list[str] | None = None,
        parent_agent_id: str | None = None,
        depth: int = 0,
    ) -> Agent:
        """Spawn a new agent within a session."""
        # Enforce depth limit
        if depth > self.config.gateway.max_spawn_depth:
            raise AgentDepthLimitError(
                f"Spawn depth {depth} exceeds limit {self.config.gateway.max_spawn_depth}"
            )

        # Enforce per-parent children limit
        if parent_agent_id:
            parent = self.agents.get(parent_agent_id)
            if parent and len(parent.children) >= self.config.gateway.max_children_per_agent:
                raise AgentConcurrencyError(
                    f"Parent agent has reached max children ({self.config.gateway.max_children_per_agent})"
                )

        # Acquire concurrency slot
        if not self._semaphore._value:
            raise AgentConcurrencyError(
                f"Max concurrent agents reached ({self.config.gateway.max_concurrent_agents})"
            )

        session = self.sessions.get(session_id)
        if not session:
            raise AgentError(f"Session '{session_id}' not found")

        # Resolve tool set
        all_tool_defs = self.tool_registry.list_definitions()
        if parent_agent_id:
            # Child agent: filter from parent's tools
            parent = self.agents.get(parent_agent_id)
            parent_tools = parent.tools if parent else all_tool_defs
            tools = resolve_child_tools(
                parent_tools,
                child_allowlist=tool_allowlist,
                child_denylist=tool_denylist,
                allow_spawning=(depth < self.config.gateway.max_spawn_depth - 1),
            )
        else:
            # Root agent: gets all tools (including spawn_agent)
            tools = all_tool_defs

        agent = Agent(
            session=session,
            gateway=self,
            llm_client=self._ollama,
            system_prompt=system_prompt,
            tools=tools,
            parent_id=parent_agent_id,
            depth=depth,
            max_iterations=self.config.gateway.max_iterations_per_agent,
            token_budget=session.token_budget,
        )

        self.agents[agent.agent_id] = agent
        session.agent_ids.append(agent.agent_id)

        self.event_bus.emit_nowait(Event(
            type=EventType.AGENT_SPAWNED,
            source_id=agent.agent_id,
            data={
                "parent_id": parent_agent_id,
                "depth": depth,
                "tools": [t.name for t in tools],
            },
        ))

        logger.info(
            "Spawned agent %s (depth=%d, tools=%s, parent=%s)",
            agent.agent_id, depth, [t.name for t in tools], parent_agent_id,
        )
        return agent

    async def terminate_agent(self, agent_id: str, cascade: bool = True) -> None:
        """Terminate an agent, optionally cascading to children."""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        if cascade:
            for child_id in list(agent.children):
                await self.terminate_agent(child_id, cascade=True)

        agent.cancel()
        del self.agents[agent_id]
        logger.info("Terminated agent: %s", agent_id)

    async def chat(self, session_id: str, user_message: str) -> AgentResult:
        """
        Main entry point: send a user message and get a response.
        Creates a root agent to handle the request.
        """
        session = self.sessions.get(session_id)
        if not session:
            session = self.create_session()

        # Record user message in session
        session.add_message(Message(
            role=MessageRole.USER,
            content=user_message,
        ))

        # Spawn root agent
        agent = await self.spawn_agent(
            session_id=session.session_id,
            depth=0,
        )

        async with self._semaphore:
            result = await agent.run(user_message)

        # Record assistant response in session
        session.add_message(Message(
            role=MessageRole.ASSISTANT,
            content=result.content,
        ))

        # Emit completion event
        self.event_bus.emit_nowait(Event(
            type=EventType.AGENT_COMPLETED,
            source_id=agent.agent_id,
            data={
                "success": result.success,
                "tool_calls": result.tool_calls_made,
                "children": result.children_spawned,
            },
        ))

        return result

    def get_agent_tree(self, agent_id: str | None = None) -> dict[str, Any]:
        """Get the agent hierarchy as a nested dict for UI display."""
        if agent_id:
            agent = self.agents.get(agent_id)
            if not agent:
                return {}
            return self._agent_to_dict(agent)

        # Return all root agents (depth 0)
        roots = [a for a in self.agents.values() if a.parent_id is None]
        return {"agents": [self._agent_to_dict(a) for a in roots]}

    def _agent_to_dict(self, agent: Agent) -> dict[str, Any]:
        return {
            "id": agent.agent_id,
            "state": agent.state.value,
            "depth": agent.depth,
            "parent_id": agent.parent_id,
            "tools": [t.name for t in agent.tools],
            "children": [
                self._agent_to_dict(self.agents[cid])
                for cid in agent.children
                if cid in self.agents
            ],
        }
