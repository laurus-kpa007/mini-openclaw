"""Central Gateway orchestrator for mini-openclaw."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from mini_openclaw.config import AppConfig
from mini_openclaw.core.agent import Agent, AgentResult
from mini_openclaw.core.agent_comm import AgentCommHub
from mini_openclaw.core.agent_roles import RoleRegistry
from mini_openclaw.core.errors import (
    AgentConcurrencyError,
    AgentDepthLimitError,
    AgentError,
)
from mini_openclaw.core.events import Event, EventBus, EventType
from mini_openclaw.core.health_monitor import HealthMonitor, HealthMonitorConfig
from mini_openclaw.core.session import Message, MessageRole, SandboxConfig, Session
from mini_openclaw.core.shared_memory import SharedMemory
from mini_openclaw.core.tool_discovery import ToolDiscovery
from mini_openclaw.llm.base import LLMClient
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
        self._llm_client: LLMClient | None = None
        self._mcp_bridge: Any = None

        # HITL approval manager
        from mini_openclaw.core.hitl import HITLManager
        self.hitl = HITLManager()

        # Scheduler (lazy-started in start())
        self.scheduler: Any = None

        # --- New subsystems for Dynamic Agent Spawning ---
        # Inter-agent communication hub
        self.comm_hub = AgentCommHub()
        # Per-session shared memory stores
        self._shared_memories: dict[str, SharedMemory] = {}
        # Agent health monitor
        self.health_monitor = HealthMonitor(self, HealthMonitorConfig())
        # Role templates registry
        self.role_registry = RoleRegistry()
        # Tool discovery service
        self.tool_discovery: ToolDiscovery | None = None

    def _create_llm_client(self) -> LLMClient:
        """Create the appropriate LLM client based on the configured provider."""
        provider = self.config.llm.provider.lower()

        if provider == "ollama":
            from mini_openclaw.llm.ollama_client import OllamaClient
            return OllamaClient(
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout,
            )
        elif provider == "lmstudio":
            from mini_openclaw.llm.lmstudio_client import LMStudioClient
            return LMStudioClient(
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout,
                api_key=self.config.llm.api_key,
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported providers: 'ollama', 'lmstudio'"
            )

    async def start(self) -> None:
        """Initialize the Gateway: connect to LLM provider, register tools, load plugins."""
        logger.info("Starting Gateway...")

        # Initialize LLM client based on provider config
        self._llm_client = self._create_llm_client()
        logger.info("Using LLM provider: %s", self.config.llm.provider)

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

        # Initialize tool discovery service
        self.tool_discovery = ToolDiscovery(self.tool_registry)

        # Initialize and start scheduler
        from mini_openclaw.core.scheduler import Scheduler
        from mini_openclaw.tools.builtin.cron_job import set_scheduler
        self.scheduler = Scheduler(self)
        set_scheduler(self.scheduler)
        await self.scheduler.start()

        # Start health monitor
        await self.health_monitor.start()

        logger.info("Gateway started (model=%s)", self.config.llm.model)

    async def shutdown(self) -> None:
        """Gracefully shut down: terminate agents, close connections."""
        logger.info("Shutting down Gateway...")

        # Stop health monitor
        await self.health_monitor.stop()

        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop()

        # Cancel pending HITL approvals
        self.hitl.cancel_all()

        # Terminate all agents
        for agent_id in list(self.agents.keys()):
            await self.terminate_agent(agent_id, cascade=True)

        # Disconnect MCP servers
        if self._mcp_bridge:
            await self._mcp_bridge.disconnect_all()

        # Close Ollama client
        if self._llm_client:
            await self._llm_client.close()

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

        # Create shared memory for this session
        self._shared_memories[session.session_id] = SharedMemory(session.session_id)

        self.event_bus.emit_nowait(Event(
            type=EventType.SESSION_CREATED,
            source_id=session.session_id,
        ))
        logger.info("Created session: %s", session.session_id)
        return session

    def get_shared_memory(self, session_id: str) -> SharedMemory | None:
        """Get the shared memory store for a session."""
        return self._shared_memories.get(session_id)

    async def spawn_agent(
        self,
        session_id: str,
        system_prompt: str | None = None,
        tool_allowlist: list[str] | None = None,
        tool_denylist: list[str] | None = None,
        parent_agent_id: str | None = None,
        depth: int = 0,
        role: str | None = None,
        prior_history: list[Message] | None = None,
    ) -> Agent:
        """Spawn a new agent within a session, optionally using a role template."""
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

        # Apply role template if specified
        role_template = None
        if role:
            role_template = self.role_registry.get(role)
            if role_template:
                if not system_prompt:
                    system_prompt = role_template.system_prompt
                if not tool_allowlist and role_template.tool_allowlist:
                    tool_allowlist = role_template.tool_allowlist
                if not tool_denylist and role_template.tool_denylist:
                    tool_denylist = role_template.tool_denylist

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
            llm_client=self._llm_client,
            system_prompt=system_prompt,
            tools=tools,
            parent_id=parent_agent_id,
            depth=depth,
            max_iterations=self.config.gateway.max_iterations_per_agent,
            token_budget=session.token_budget,
            prior_history=prior_history,
        )

        self.agents[agent.agent_id] = agent
        session.agent_ids.append(agent.agent_id)

        # Register with communication hub and shared memory
        self.comm_hub.register_agent(agent.agent_id, parent_id=parent_agent_id)
        shared_mem = self._shared_memories.get(session_id)
        if shared_mem:
            shared_mem.register_agent(agent.agent_id, parent_id=parent_agent_id)

        # Register with health monitor
        self.health_monitor.register_agent(agent.agent_id)

        self.event_bus.emit_nowait(Event(
            type=EventType.AGENT_SPAWNED,
            source_id=agent.agent_id,
            data={
                "parent_id": parent_agent_id,
                "depth": depth,
                "tools": [t.name for t in tools],
                "role": role,
            },
        ))

        logger.info(
            "Spawned agent %s (depth=%d, tools=%s, parent=%s, role=%s)",
            agent.agent_id, depth, [t.name for t in tools], parent_agent_id, role,
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

        # Unregister from subsystems
        self.comm_hub.unregister_agent(agent_id)
        self.health_monitor.unregister_agent(agent_id)
        if agent.session:
            shared_mem = self._shared_memories.get(agent.session.session_id)
            if shared_mem:
                shared_mem.unregister_agent(agent_id)

        del self.agents[agent_id]
        logger.info("Terminated agent: %s", agent_id)

    def _build_prior_history(self, session: Session) -> list[Message]:
        """
        Build prior conversation history from the session for a new root agent.

        Extracts USER and ASSISTANT messages (the high-level conversation) so
        the LLM can see the full multi-turn context. Tool-level detail from
        previous turns is omitted to save context window tokens — only the
        current turn's tool interactions are tracked in the agent's own _history.
        """
        prior: list[Message] = []
        for msg in session.conversation_history:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                prior.append(msg)
        return prior

    async def chat(self, session_id: str, user_message: str) -> AgentResult:
        """
        Main entry point: send a user message and get a response.
        Creates a root agent to handle the request.
        """
        session = self.sessions.get(session_id)
        if not session:
            session = self.create_session()

        # Build prior history BEFORE adding the new user message,
        # because agent.run() will add the user message itself.
        prior_history = self._build_prior_history(session)

        # Record user message in session
        session.add_message(Message(
            role=MessageRole.USER,
            content=user_message,
        ))

        # Spawn root agent with prior conversation context
        agent = await self.spawn_agent(
            session_id=session.session_id,
            depth=0,
            prior_history=prior_history,
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
        health = self.health_monitor.get_health(agent.agent_id)
        return {
            "id": agent.agent_id,
            "state": agent.state.value,
            "depth": agent.depth,
            "parent_id": agent.parent_id,
            "tools": [t.name for t in agent.tools],
            "health": health.status.value if health else "unknown",
            "children": [
                self._agent_to_dict(self.agents[cid])
                for cid in agent.children
                if cid in self.agents
            ],
        }
