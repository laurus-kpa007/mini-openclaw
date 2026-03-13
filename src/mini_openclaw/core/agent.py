"""Agent class with ReAct loop for tool-augmented LLM execution."""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mini_openclaw.core.errors import (
    AgentDepthLimitError,
    AgentError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
)
from mini_openclaw.core.events import Event, EventType
from mini_openclaw.core.session import Message, MessageRole, Session, ToolCallInfo
from mini_openclaw.llm.base import ChatResponse, LLMClient, ToolCall
from mini_openclaw.llm.context_manager import ContextManager
from mini_openclaw.tools.base import ToolContext, ToolDefinition, ToolResult

if TYPE_CHECKING:
    from mini_openclaw.core.gateway import Gateway

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.
Use the available tools to help the user accomplish their tasks.
When you need to perform an action, use the appropriate tool.
When you have gathered enough information, provide a clear and concise response.

Available tools: {tool_names}
"""


class AgentState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_CHILD = "waiting_for_child"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class AgentResult:
    success: bool
    content: str
    tool_calls_made: int = 0
    tokens_used: int = 0
    children_spawned: list[str] = field(default_factory=list)


# Tool names that require special handling by the Agent
AGENT_HANDLED_TOOLS = {"spawn_agent", "spawn_parallel", "agent_comm", "shared_memory"}


class Agent:
    """
    Single LLM agent executing within a session.
    Runs a ReAct loop: LLM call → tool execution → repeat until done.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        session: Session | None = None,
        gateway: Gateway | None = None,
        llm_client: LLMClient | None = None,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        parent_id: str | None = None,
        depth: int = 0,
        max_iterations: int = 20,
        token_budget: int = 8192,
        prior_history: list[Message] | None = None,
    ) -> None:
        self.agent_id = agent_id or f"agent:{uuid.uuid4().hex[:12]}"
        self.session = session
        self.gateway = gateway
        self.llm_client = llm_client
        self.tools = tools or []
        self.parent_id = parent_id
        self.depth = depth
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        self.state = AgentState.INITIALIZING
        self.children: list[str] = []
        self._task: asyncio.Task | None = None
        self._context_manager = ContextManager(max_tokens=token_budget)

        # Build system prompt with tool names
        tool_names = ", ".join(t.name for t in self.tools) if self.tools else "none"
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if "{tool_names}" in self.system_prompt:
            self.system_prompt = self.system_prompt.format(tool_names=tool_names)

        # Per-agent conversation history.
        # For root agents in multi-turn sessions, prior_history carries
        # earlier USER/ASSISTANT exchanges so the LLM sees the full context.
        self._history: list[Message] = list(prior_history) if prior_history else []

    def _set_state(self, new_state: AgentState) -> None:
        old_state = self.state
        self.state = new_state
        if self.gateway:
            self.gateway.event_bus.emit_nowait(Event(
                type=EventType.AGENT_STATE_CHANGED,
                source_id=self.agent_id,
                data={"old_state": old_state.value, "new_state": new_state.value},
            ))

    async def run(self, user_message: str) -> AgentResult:
        """Execute the agent's ReAct loop."""
        self._set_state(AgentState.RUNNING)
        total_tokens = 0
        total_tool_calls = 0

        # Add user message to history
        self._history.append(Message(
            role=MessageRole.USER,
            content=user_message,
            agent_id=self.agent_id,
        ))

        try:
            for iteration in range(self.max_iterations):
                logger.info(
                    "[%s] Iteration %d/%d",
                    self.agent_id, iteration + 1, self.max_iterations,
                )

                # Send heartbeat to health monitor
                if self.gateway:
                    self.gateway.health_monitor.heartbeat(
                        self.agent_id,
                        metadata={"iteration": iteration + 1},
                    )
                    self.gateway.health_monitor.record_activity(
                        self.agent_id, "iteration"
                    )

                # Check for incoming messages from other agents
                await self._check_mailbox()

                # Prepare messages for LLM
                messages = self._context_manager.prepare_messages(
                    self.system_prompt, self._history
                )

                # Build tool schemas
                tool_schemas = [t.to_ollama_schema() for t in self.tools] if self.tools else None

                # Call LLM
                response = await self.llm_client.chat(messages, tools=tool_schemas)
                total_tokens += response.tokens_used

                # Track tokens in health monitor
                if self.gateway:
                    self.gateway.health_monitor.record_tokens(
                        self.agent_id, response.tokens_used
                    )

                # Emit LLM response event
                if self.gateway:
                    self.gateway.event_bus.emit_nowait(Event(
                        type=EventType.LLM_RESPONSE,
                        source_id=self.agent_id,
                        data={"content": response.content, "has_tool_calls": bool(response.tool_calls)},
                    ))

                # No tool calls → final response
                if not response.tool_calls:
                    self._history.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        agent_id=self.agent_id,
                    ))
                    self._set_state(AgentState.COMPLETED)
                    return AgentResult(
                        success=True,
                        content=response.content,
                        tool_calls_made=total_tool_calls,
                        tokens_used=total_tokens,
                        children_spawned=list(self.children),
                    )

                # Has tool calls → execute each
                # Record assistant message with tool_calls
                self._history.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=[
                        ToolCallInfo(name=tc.name, arguments=tc.arguments)
                        for tc in response.tool_calls
                    ],
                    agent_id=self.agent_id,
                ))

                for tool_call in response.tool_calls:
                    total_tool_calls += 1
                    self._set_state(AgentState.WAITING_FOR_TOOL)

                    result = await self._execute_tool(tool_call)

                    # Add tool result to history
                    self._history.append(Message(
                        role=MessageRole.TOOL,
                        content=result.output if result.success else f"Error: {result.error}",
                        agent_id=self.agent_id,
                    ))

                self._set_state(AgentState.RUNNING)

            # Max iterations reached
            self._set_state(AgentState.COMPLETED)
            return AgentResult(
                success=True,
                content="[Max iterations reached. Here is what I found so far.]\n" + (
                    self._history[-1].content if self._history else ""
                ),
                tool_calls_made=total_tool_calls,
                tokens_used=total_tokens,
                children_spawned=list(self.children),
            )

        except Exception as e:
            logger.exception("[%s] Agent failed", self.agent_id)
            self._set_state(AgentState.FAILED)
            if self.gateway:
                self.gateway.health_monitor.record_failure(self.agent_id)
                self.gateway.event_bus.emit_nowait(Event(
                    type=EventType.AGENT_FAILED,
                    source_id=self.agent_id,
                    data={"error": str(e)},
                ))
            return AgentResult(
                success=False,
                content=f"Agent error: {e}",
                tool_calls_made=total_tool_calls,
                tokens_used=total_tokens,
                children_spawned=list(self.children),
            )

    async def _check_mailbox(self) -> None:
        """Check for incoming messages from other agents and inject them into history."""
        if not self.gateway:
            return
        mailbox = self.gateway.comm_hub.get_mailbox(self.agent_id)
        if not mailbox or mailbox.is_empty:
            return

        messages = await mailbox.get_all()
        for msg in messages:
            # Inject as a system-like message so the LLM is aware
            self._history.append(Message(
                role=MessageRole.TOOL,
                content=f"[Message from agent {msg.sender_id}] ({msg.msg_type}): {msg.content}",
                agent_id=self.agent_id,
            ))
            logger.info(
                "[%s] Received message from %s: %s",
                self.agent_id, msg.sender_id, msg.content[:100],
            )

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with HITL approval and spawn routing."""
        tool_name = tool_call.name
        arguments = tool_call.arguments

        # Check tool is in our allowed set
        allowed_names = {t.name for t in self.tools}
        if tool_name not in allowed_names:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' is not available to this agent",
            )

        # Emit tool_called event
        if self.gateway:
            self.gateway.event_bus.emit_nowait(Event(
                type=EventType.TOOL_CALLED,
                source_id=self.agent_id,
                data={"tool": tool_name, "arguments": arguments},
            ))
            # Record activity in health monitor
            self.gateway.health_monitor.record_activity(self.agent_id, "tool_call")

        # --- HITL: Check if this tool requires approval ---
        tool_def = next((t for t in self.tools if t.name == tool_name), None)
        if tool_def and tool_def.requires_approval and self.gateway and self.gateway.hitl:
            approval = await self._request_approval(tool_name, arguments)
            if not approval.approved:
                reason = approval.reason or "denied by user"
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool '{tool_name}' was not approved: {reason}",
                )
            # User may have modified the arguments
            if approval.modified_arguments:
                arguments = approval.modified_arguments

        # --- Agent-handled tools (special routing) ---
        if tool_name in AGENT_HANDLED_TOOLS:
            if tool_name == "spawn_agent":
                result = await self._handle_spawn(arguments)
            elif tool_name == "spawn_parallel":
                result = await self._handle_spawn_parallel(arguments)
            elif tool_name == "agent_comm":
                result = await self._handle_agent_comm(arguments)
            elif tool_name == "shared_memory":
                result = await self._handle_shared_memory(arguments)
            else:
                result = ToolResult(success=False, output="", error=f"Unhandled agent tool: {tool_name}")
        else:
            # Regular tool execution via gateway
            if self.gateway:
                context = ToolContext(
                    session_id=self.session.session_id if self.session else "",
                    agent_id=self.agent_id,
                    sandbox_root=self.session.sandbox.root_path if self.session else None,
                    allowed_hosts=self.session.sandbox.allowed_hosts if self.session else None,
                    working_directory=".",
                )
                result = await self.gateway.tool_registry.execute(tool_name, arguments, context)
            else:
                result = ToolResult(success=False, output="", error="No gateway available")

        # Emit tool_result event
        if self.gateway:
            self.gateway.event_bus.emit_nowait(Event(
                type=EventType.TOOL_RESULT,
                source_id=self.agent_id,
                data={
                    "tool": tool_name,
                    "success": result.success,
                    "output_preview": result.output[:200] if result.output else "",
                },
            ))

        return result

    async def _request_approval(self, tool_name: str, arguments: dict[str, Any]):
        """Request HITL approval and emit the event for UI to pick up."""
        from mini_openclaw.core.hitl import ApprovalResponse, _describe_tool_call

        session_id = self.session.session_id if self.session else ""

        # Start the approval request (returns a coroutine that awaits user input)
        import asyncio
        # We need to emit the event AND await the future concurrently:
        # 1) Create the future inside hitl manager
        # 2) Emit the event so UI knows to prompt the user
        # 3) Await the future

        description = _describe_tool_call(tool_name, arguments)

        # Create a task for the approval request
        approval_coro = self.gateway.hitl.request_approval(
            agent_id=self.agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            description=description,
        )

        # We need to start the request, then emit the event with the request_id
        # The request_approval creates the future and adds to pending, then awaits
        # So we schedule it as a task and emit event after a tiny delay
        approval_task = asyncio.create_task(approval_coro)

        # Give the coroutine a moment to register the pending request
        await asyncio.sleep(0.01)

        # Now emit the event so UI picks it up
        pending = self.gateway.hitl.get_pending()
        for req in pending:
            if req.agent_id == self.agent_id and req.tool_name == tool_name:
                await self.gateway.event_bus.emit(Event(
                    type=EventType.TOOL_APPROVAL_REQUESTED,
                    source_id=self.agent_id,
                    data={
                        "request_id": req.request_id,
                        "tool": tool_name,
                        "arguments": arguments,
                        "description": req.description,
                    },
                ))
                break

        # Wait for the user response
        return await approval_task

    async def _handle_spawn(self, spawn_params: dict[str, Any]) -> ToolResult:
        """Handle the spawn_agent tool call."""
        if not self.gateway:
            return ToolResult(success=False, output="", error="No gateway available for spawning")

        task = spawn_params.get("task", "")
        requested_tools = spawn_params.get("tools")
        custom_prompt = spawn_params.get("system_prompt")
        role = spawn_params.get("role")

        if not task:
            return ToolResult(success=False, output="", error="No task description provided")

        try:
            self._set_state(AgentState.WAITING_FOR_CHILD)

            # Auto-suggest role if not specified and tool discovery is available
            if not role and self.gateway.tool_discovery:
                suggested = self.gateway.tool_discovery.suggest_role(task)
                if suggested:
                    role = suggested
                    logger.info("[%s] Auto-suggested role '%s' for task: %s", self.agent_id, role, task[:60])

            child = await self.gateway.spawn_agent(
                session_id=self.session.session_id if self.session else "",
                system_prompt=custom_prompt,
                tool_allowlist=requested_tools,
                parent_agent_id=self.agent_id,
                depth=self.depth + 1,
                role=role,
            )
            self.children.append(child.agent_id)

            # Run child agent and await result
            child_result = await child.run(task)

            return ToolResult(
                success=child_result.success,
                output=f"[Child agent {child.agent_id} result]\n{child_result.content}",
            )
        except AgentDepthLimitError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            logger.exception("Failed to spawn child agent")
            return ToolResult(success=False, output="", error=f"Spawn failed: {e}")

    async def _handle_spawn_parallel(self, params: dict[str, Any]) -> ToolResult:
        """Handle the spawn_parallel tool call for multi-child parallel execution."""
        if not self.gateway:
            return ToolResult(success=False, output="", error="No gateway available")

        from mini_openclaw.core.result_aggregator import (
            AggregationStrategy,
            ChildTask,
            ResultAggregator,
        )

        tasks_raw = params.get("tasks", [])
        strategy_name = params.get("strategy", "wait_all")
        timeout = params.get("timeout")

        if not tasks_raw:
            return ToolResult(success=False, output="", error="No tasks provided")

        # Parse tasks
        child_tasks = []
        for t in tasks_raw:
            if isinstance(t, str):
                child_tasks.append(ChildTask(task_description=t))
            elif isinstance(t, dict):
                child_tasks.append(ChildTask(
                    task_description=t.get("task", ""),
                    tool_allowlist=t.get("tools"),
                    system_prompt=t.get("system_prompt"),
                    priority=t.get("priority", 0),
                ))

        # Parse strategy
        try:
            strategy = AggregationStrategy(strategy_name)
        except ValueError:
            strategy = AggregationStrategy.WAIT_ALL

        # Emit parallel spawn start event
        self.gateway.event_bus.emit_nowait(Event(
            type=EventType.PARALLEL_SPAWN_STARTED,
            source_id=self.agent_id,
            data={
                "task_count": len(child_tasks),
                "strategy": strategy.value,
            },
        ))

        self._set_state(AgentState.WAITING_FOR_CHILD)

        aggregator = ResultAggregator(self.gateway, self)
        result = await aggregator.spawn_and_aggregate(
            child_tasks,
            strategy=strategy,
            timeout=float(timeout) if timeout else None,
        )

        # Emit completion event
        self.gateway.event_bus.emit_nowait(Event(
            type=EventType.PARALLEL_SPAWN_COMPLETED,
            source_id=self.agent_id,
            data={
                "success": result.success,
                "strategy": result.strategy_used,
                "children_total": result.children_total,
                "children_succeeded": result.children_succeeded,
            },
        ))

        return ToolResult(
            success=result.success,
            output=result.summary,
        )

    async def _handle_agent_comm(self, params: dict[str, Any]) -> ToolResult:
        """Handle the agent_comm tool for inter-agent messaging."""
        if not self.gateway:
            return ToolResult(success=False, output="", error="No gateway available")

        from mini_openclaw.core.agent_comm import AgentMessage, MessagePriority

        action = params.get("action", "")
        content = params.get("content", "")
        receiver_id = params.get("receiver_id", "")
        priority_str = params.get("priority", "normal")

        priority_map = {
            "low": MessagePriority.LOW,
            "normal": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "urgent": MessagePriority.URGENT,
        }
        priority = priority_map.get(priority_str, MessagePriority.NORMAL)

        if action == "send":
            if not receiver_id:
                return ToolResult(success=False, output="", error="receiver_id required for 'send'")
            if not content:
                return ToolResult(success=False, output="", error="content required for 'send'")
            msg = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                content=content,
                msg_type="info",
                priority=priority,
            )
            delivered = await self.gateway.comm_hub.send(msg)
            if delivered:
                self.gateway.event_bus.emit_nowait(Event(
                    type=EventType.AGENT_MESSAGE_SENT,
                    source_id=self.agent_id,
                    data={"receiver": receiver_id, "content_preview": content[:100]},
                ))
            return ToolResult(
                success=delivered,
                output=f"Message {'delivered' if delivered else 'failed'} to {receiver_id}",
            )

        elif action == "send_to_parent":
            if not content:
                return ToolResult(success=False, output="", error="content required")
            if not self.parent_id:
                return ToolResult(success=False, output="", error="No parent agent")
            delivered = await self.gateway.comm_hub.send_to_parent(
                self.agent_id, content, priority=priority,
            )
            return ToolResult(
                success=delivered,
                output=f"Message {'delivered' if delivered else 'failed'} to parent {self.parent_id}",
            )

        elif action == "broadcast":
            if not content:
                return ToolResult(success=False, output="", error="content required")
            root = self.gateway.comm_hub._find_root(self.agent_id)
            group = f"family:{root}"
            count = await self.gateway.comm_hub.broadcast(
                self.agent_id, group, content, priority=priority,
            )
            self.gateway.event_bus.emit_nowait(Event(
                type=EventType.AGENT_BROADCAST,
                source_id=self.agent_id,
                data={"group": group, "delivered_count": count},
            ))
            return ToolResult(
                success=True,
                output=f"Broadcast delivered to {count} agents in group '{group}'",
            )

        elif action == "receive":
            mailbox = self.gateway.comm_hub.get_mailbox(self.agent_id)
            if not mailbox or mailbox.is_empty:
                return ToolResult(success=True, output="No pending messages")
            messages = await mailbox.get_all()
            parts = []
            for msg in messages:
                parts.append(f"[From {msg.sender_id}] ({msg.msg_type}, {msg.priority.name}): {msg.content}")
            return ToolResult(success=True, output="\n".join(parts))

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")

    async def _handle_shared_memory(self, params: dict[str, Any]) -> ToolResult:
        """Handle the shared_memory tool for inter-agent state sharing."""
        if not self.gateway or not self.session:
            return ToolResult(success=False, output="", error="No gateway/session available")

        from mini_openclaw.core.shared_memory import AccessLevel

        shared_mem = self.gateway.get_shared_memory(self.session.session_id)
        if not shared_mem:
            return ToolResult(success=False, output="", error="No shared memory for this session")

        action = params.get("action", "")
        key = params.get("key", "")
        value = params.get("value", "")

        access_map = {
            "public": AccessLevel.PUBLIC,
            "family": AccessLevel.FAMILY,
            "private": AccessLevel.PRIVATE,
        }
        access_level = access_map.get(params.get("access_level", "family"), AccessLevel.FAMILY)

        if action == "put":
            if not key:
                return ToolResult(success=False, output="", error="key required for 'put'")
            tags = params.get("tags", [])
            try:
                entry = await shared_mem.put(
                    key=key,
                    value=value,
                    owner_id=self.agent_id,
                    access_level=access_level,
                    tags=tags if isinstance(tags, list) else [],
                )
                self.gateway.event_bus.emit_nowait(Event(
                    type=EventType.SHARED_MEMORY_UPDATED,
                    source_id=self.agent_id,
                    data={"key": key, "action": "put", "version": entry.version},
                ))
                return ToolResult(
                    success=True,
                    output=f"Stored '{key}' (version={entry.version}, access={access_level.value})",
                )
            except PermissionError as e:
                return ToolResult(success=False, output="", error=str(e))

        elif action == "get":
            if not key:
                return ToolResult(success=False, output="", error="key required for 'get'")
            result = await shared_mem.get(key, requester_id=self.agent_id)
            if result is None:
                return ToolResult(success=True, output=f"Key '{key}' not found or not accessible")
            return ToolResult(success=True, output=str(result))

        elif action == "delete":
            if not key:
                return ToolResult(success=False, output="", error="key required for 'delete'")
            deleted = await shared_mem.delete(key, requester_id=self.agent_id)
            return ToolResult(
                success=True,
                output=f"Key '{key}' {'deleted' if deleted else 'not found or no permission'}",
            )

        elif action == "list":
            prefix = params.get("key")  # Reuse key param as prefix
            keys = await shared_mem.list_keys(requester_id=self.agent_id, prefix=prefix)
            if not keys:
                return ToolResult(success=True, output="No accessible keys found")
            return ToolResult(success=True, output=f"Keys: {', '.join(keys)}")

        elif action == "search":
            tag = params.get("tag", "")
            if not tag:
                return ToolResult(success=False, output="", error="tag required for 'search'")
            entries = await shared_mem.search_by_tag(tag, requester_id=self.agent_id)
            if not entries:
                return ToolResult(success=True, output=f"No entries with tag '{tag}'")
            parts = [f"  {e.key}: {str(e.value)[:100]}" for e in entries]
            return ToolResult(success=True, output=f"Entries with tag '{tag}':\n" + "\n".join(parts))

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")

    def cancel(self) -> None:
        """Cancel the running agent task."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._set_state(AgentState.TERMINATED)
