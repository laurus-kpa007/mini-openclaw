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

        # Per-agent conversation history (separate from session global history)
        self._history: list[Message] = []

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

                # Prepare messages for LLM
                messages = self._context_manager.prepare_messages(
                    self.system_prompt, self._history
                )

                # Build tool schemas
                tool_schemas = [t.to_ollama_schema() for t in self.tools] if self.tools else None

                # Call LLM
                response = await self.llm_client.chat(messages, tools=tool_schemas)
                total_tokens += response.tokens_used

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

        # Special handling for spawn_agent
        if tool_name == "spawn_agent":
            result = await self._handle_spawn(arguments)
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

        if not task:
            return ToolResult(success=False, output="", error="No task description provided")

        try:
            self._set_state(AgentState.WAITING_FOR_CHILD)

            child = await self.gateway.spawn_agent(
                session_id=self.session.session_id if self.session else "",
                system_prompt=custom_prompt,
                tool_allowlist=requested_tools,
                parent_agent_id=self.agent_id,
                depth=self.depth + 1,
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

    def cancel(self) -> None:
        """Cancel the running agent task."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._set_state(AgentState.TERMINATED)
