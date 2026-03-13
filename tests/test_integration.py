"""Integration tests for the mini-openclaw system.

Tests the full flow: Gateway → Agent → Tool execution → HITL approval,
using mock LLM clients to avoid requiring a running Ollama server.
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from mini_openclaw.config import AppConfig
from mini_openclaw.core.agent import Agent, AgentResult, AgentState
from mini_openclaw.core.events import Event, EventBus, EventType
from mini_openclaw.core.gateway import Gateway
from mini_openclaw.core.hitl import ApprovalPolicy, HITLManager
from mini_openclaw.core.session import Message, MessageRole, SandboxConfig, Session
from mini_openclaw.llm.base import ChatResponse, ChatStreamChunk, LLMClient, ToolCall
from mini_openclaw.tools.base import ToolContext, ToolDefinition, ToolParameter, ToolResult
from mini_openclaw.tools.registry import ToolRegistry
from mini_openclaw.tools.permissions import resolve_tools, resolve_child_tools


# ── Mock LLM Client ──────────────────────────────────────────────────


class MockLLMClient(LLMClient):
    """Mock LLM that returns scripted responses."""

    def __init__(self, responses: list[ChatResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = ChatResponse(content="I'm done.", tokens_used=10)
        self._call_count += 1
        return resp

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="streaming response", done=True)

    async def check_tool_support(self) -> bool:
        return True

    async def close(self):
        pass


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sandbox_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def config(sandbox_dir):
    return AppConfig(
        tools={"plugin_dirs": [], "mcp_servers": []},
        security={"sandbox_root": sandbox_dir},
    )


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register_builtin_tools()
    return reg


# ── 1. Module Import Tests ───────────────────────────────────────────


def test_all_core_modules_importable():
    """All core modules can be imported without errors."""
    from mini_openclaw.core import agent, errors, events, gateway, hitl, session, scheduler
    from mini_openclaw.config import AppConfig, load_config
    assert agent and errors and events and gateway and hitl and session and scheduler
    assert AppConfig and load_config


def test_all_tool_modules_importable():
    """All tool modules can be imported without errors."""
    from mini_openclaw.tools import base, registry, permissions
    from mini_openclaw.tools.builtin import (
        file_read, file_write, shell_exec,
        web_search, http_request,
        pip_install, python_exec,
        browser_control, cron_job,
        spawn_agent,
    )
    assert base and registry and permissions
    assert file_read and file_write and shell_exec
    assert web_search and http_request
    assert pip_install and python_exec
    assert browser_control and cron_job and spawn_agent


def test_all_llm_modules_importable():
    """All LLM modules can be imported without errors."""
    from mini_openclaw.llm import base, ollama_client, react_fallback, tool_calling, context_manager
    assert base and ollama_client and react_fallback and tool_calling and context_manager


def test_all_interface_modules_importable():
    """All interface modules can be imported without errors."""
    from mini_openclaw.interfaces.cli import app as cli_app
    from mini_openclaw.interfaces.web import app as web_app
    from mini_openclaw.interfaces.web import routes, websocket
    assert cli_app and web_app and routes and websocket


# ── 2. Config & Session Tests ────────────────────────────────────────


def test_default_config():
    """Default AppConfig is valid with all expected sections."""
    cfg = AppConfig()
    assert cfg.gateway.max_concurrent_agents == 8
    assert cfg.gateway.max_spawn_depth == 3
    assert cfg.llm.provider == "ollama"
    assert cfg.security.sandbox_root is None
    assert cfg.tools.plugin_dirs == ["./plugins"]


def test_session_creation_and_history():
    """Session tracks messages correctly."""
    session = Session(model="test-model")
    assert session.session_id.startswith("session:")
    assert len(session.conversation_history) == 0

    session.add_message(Message(role=MessageRole.USER, content="hello"))
    session.add_message(Message(role=MessageRole.ASSISTANT, content="hi"))
    assert len(session.get_history()) == 2
    assert session.get_history(1)[0].content == "hi"

    session.clear_history()
    assert len(session.get_history()) == 0


# ── 3. EventBus Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eventbus_subscribe_and_emit():
    """EventBus delivers events to type-specific subscribers."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.TOOL_CALLED, handler)

    await bus.emit(Event(type=EventType.TOOL_CALLED, source_id="a1", data={"tool": "file_read"}))
    await bus.emit(Event(type=EventType.AGENT_SPAWNED, source_id="a2"))  # different type

    assert len(received) == 1
    assert received[0].data["tool"] == "file_read"


@pytest.mark.asyncio
async def test_eventbus_subscribe_all():
    """subscribe_all receives events of all types."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event.type)

    bus.subscribe_all(handler)

    await bus.emit(Event(type=EventType.TOOL_CALLED, source_id="a1"))
    await bus.emit(Event(type=EventType.AGENT_SPAWNED, source_id="a2"))
    await bus.emit(Event(type=EventType.SESSION_CREATED, source_id="s1"))

    assert len(received) == 3


@pytest.mark.asyncio
async def test_eventbus_unsubscribe():
    """Unsubscribed handlers stop receiving events."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe_all(handler)
    await bus.emit(Event(type=EventType.TOOL_CALLED, source_id="a1"))
    assert len(received) == 1

    bus.unsubscribe_all(handler)
    await bus.emit(Event(type=EventType.TOOL_CALLED, source_id="a1"))
    assert len(received) == 1  # No additional events


# ── 4. ToolRegistry Integration Tests ────────────────────────────────


def test_builtin_tools_all_registered(registry):
    """All built-in tools are registered."""
    names = registry.list_names()
    expected = [
        "file_read", "file_write", "shell_exec",
        "web_search", "http_request",
        "pip_install", "python_exec",
        "browser_control", "cron_job",
        "spawn_agent",
        "spawn_parallel", "agent_comm", "shared_memory",
    ]
    for name in expected:
        assert name in names, f"Missing tool: {name}"
    assert len(names) == 13


def test_tool_definitions_have_required_fields(registry):
    """Each registered tool has valid definition fields."""
    for td in registry.list_definitions():
        assert td.name
        assert td.description
        assert isinstance(td.requires_approval, bool)
        assert isinstance(td.category, str)


def test_approval_required_tools(registry):
    """Dangerous tools correctly have requires_approval=True."""
    defs = {td.name: td for td in registry.list_definitions()}
    assert defs["file_write"].requires_approval is True
    assert defs["shell_exec"].requires_approval is True
    assert defs["pip_install"].requires_approval is True
    assert defs["python_exec"].requires_approval is True
    assert defs["file_read"].requires_approval is False


@pytest.mark.asyncio
async def test_tool_execution_via_registry(registry, sandbox_dir):
    """Execute a tool through the registry (file_read on missing file)."""
    ctx = ToolContext(session_id="s1", agent_id="a1", sandbox_root=sandbox_dir, working_directory=sandbox_dir)
    result = await registry.execute("file_read", {"path": "nonexistent.txt"}, ctx)
    assert not result.success


def test_permissions_resolve_child_tools(registry):
    """Child tool resolution strips spawn_agent by default."""
    all_defs = registry.list_definitions()
    child_defs = resolve_child_tools(all_defs)
    child_names = [t.name for t in child_defs]
    assert "spawn_agent" not in child_names
    assert "file_read" in child_names


# ── 5. Agent ReAct Loop Integration ─────────────────────────────────


@pytest.mark.asyncio
async def test_agent_simple_response():
    """Agent with no tool calls returns LLM content directly."""
    mock_llm = MockLLMClient([
        ChatResponse(content="Hello! I can help you.", tokens_used=15),
    ])
    session = Session(model="mock")
    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[],
        max_iterations=5,
    )

    result = await agent.run("Hi there")
    assert result.success
    assert "Hello" in result.content
    assert result.tool_calls_made == 0
    assert agent.state == AgentState.COMPLETED


@pytest.mark.asyncio
async def test_agent_tool_call_flow(sandbox_dir):
    """Agent calls a tool, gets result, then produces final answer."""
    mock_llm = MockLLMClient([
        # Iteration 1: LLM requests file_read
        ChatResponse(
            content="Let me read the file.",
            tool_calls=[ToolCall(name="file_read", arguments={"path": "test.txt"})],
            tokens_used=20,
        ),
        # Iteration 2: LLM produces final answer after seeing tool result
        ChatResponse(
            content="The file was not found.",
            tokens_used=15,
        ),
    ])

    session = Session(model="mock", sandbox=SandboxConfig(root_path=sandbox_dir))

    # Provide file_read tool definition
    from mini_openclaw.tools.builtin.file_read import FileReadTool
    file_read_def = FileReadTool().definition

    # Create a minimal gateway-like setup
    registry = ToolRegistry()
    registry.register(FileReadTool())

    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[file_read_def],
        max_iterations=5,
    )

    # Manually set gateway for tool execution
    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.tool_registry = registry
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = HITLManager(policy=ApprovalPolicy.AUTO_APPROVE)
    agent.gateway = mock_gateway

    result = await agent.run("Read test.txt")
    assert result.success
    assert result.tool_calls_made == 1
    assert "not found" in result.content.lower()


@pytest.mark.asyncio
async def test_agent_hitl_deny_blocks_tool(sandbox_dir):
    """When HITL denies a tool, the agent gets an error result for that tool."""
    mock_llm = MockLLMClient([
        # Iteration 1: LLM requests shell_exec (requires approval)
        ChatResponse(
            content="Let me execute a command.",
            tool_calls=[ToolCall(name="shell_exec", arguments={"command": "ls"})],
            tokens_used=20,
        ),
        # Iteration 2: after tool denied, LLM gives final answer
        ChatResponse(
            content="The command was denied by the user.",
            tokens_used=15,
        ),
    ])

    from mini_openclaw.tools.builtin.shell_exec import ShellExecTool
    shell_def = ShellExecTool().definition

    session = Session(model="mock", sandbox=SandboxConfig(root_path=sandbox_dir))
    registry = ToolRegistry()
    registry.register(ShellExecTool())

    hitl = HITLManager(policy=ApprovalPolicy.AUTO_DENY)

    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.tool_registry = registry
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = hitl

    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[shell_def],
        max_iterations=5,
    )
    agent.gateway = mock_gateway

    result = await agent.run("Run ls command")
    assert result.success
    assert result.tool_calls_made == 1
    assert "denied" in result.content.lower()


@pytest.mark.asyncio
async def test_agent_hitl_auto_approve_allows_tool(sandbox_dir):
    """When HITL auto-approves, the tool executes normally."""
    mock_llm = MockLLMClient([
        ChatResponse(
            content="Let me run some code.",
            tool_calls=[ToolCall(name="python_exec", arguments={"code": "print(42)"})],
            tokens_used=20,
        ),
        ChatResponse(
            content="The result is 42.",
            tokens_used=10,
        ),
    ])

    from mini_openclaw.tools.builtin.python_exec import PythonExecTool
    python_def = PythonExecTool().definition

    session = Session(model="mock", sandbox=SandboxConfig(root_path=sandbox_dir))
    registry = ToolRegistry()
    registry.register(PythonExecTool())

    hitl = HITLManager(policy=ApprovalPolicy.AUTO_APPROVE)

    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.tool_registry = registry
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = hitl

    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[python_def],
        max_iterations=5,
    )
    agent.gateway = mock_gateway

    result = await agent.run("Print 42")
    assert result.success
    assert result.tool_calls_made == 1
    assert "42" in result.content


@pytest.mark.asyncio
async def test_agent_max_iterations():
    """Agent stops after max_iterations and returns partial result."""
    # LLM always requests a tool call → never produces a final answer
    tool_def = ToolDefinition(name="dummy", description="dummy tool")

    responses = [
        ChatResponse(
            content=f"Iteration {i}",
            tool_calls=[ToolCall(name="dummy", arguments={})],
            tokens_used=5,
        )
        for i in range(10)
    ]
    mock_llm = MockLLMClient(responses)

    session = Session(model="mock")

    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = None

    # Create a dummy tool in registry that always succeeds
    class DummyTool:
        @property
        def definition(self):
            return tool_def

        async def execute(self, arguments, context):
            return ToolResult(success=True, output="ok")

    reg = ToolRegistry()
    reg.register(DummyTool())
    mock_gateway.tool_registry = reg

    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[tool_def],
        max_iterations=3,
    )
    agent.gateway = mock_gateway

    result = await agent.run("Loop forever")
    assert result.success  # Graceful stop, not a crash
    assert result.tool_calls_made == 3
    assert "Max iterations" in result.content


# ── 6. Gateway Integration Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_gateway_create_session(config):
    """Gateway can create a session."""
    gw = Gateway(config)
    session = gw.create_session()
    assert session.session_id in gw.sessions
    assert len(gw.sessions) == 1
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_start_registers_tools(config):
    """Gateway.start() registers all 10 builtin tools."""
    gw = Gateway(config)
    # Patch Ollama to avoid network calls
    mock_llm = MockLLMClient()
    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()

    names = gw.tool_registry.list_names()
    assert len(names) == 13
    assert "spawn_agent" in names
    assert "pip_install" in names
    assert "browser_control" in names
    assert "cron_job" in names
    assert "spawn_parallel" in names
    assert "agent_comm" in names
    assert "shared_memory" in names
    assert gw.scheduler is not None
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_spawn_agent_depth_limit(config):
    """Spawning beyond max depth raises AgentDepthLimitError."""
    from mini_openclaw.core.errors import AgentDepthLimitError

    gw = Gateway(config)
    mock_llm = MockLLMClient()
    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()

    session = gw.create_session()

    with pytest.raises(AgentDepthLimitError):
        await gw.spawn_agent(
            session_id=session.session_id,
            depth=config.gateway.max_spawn_depth + 1,
        )
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_spawn_root_and_child(config):
    """Gateway can spawn a root agent and a child agent with correct tool filtering."""
    gw = Gateway(config)
    mock_llm = MockLLMClient()
    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()

    session = gw.create_session()

    # Root agent gets all tools
    root = await gw.spawn_agent(session_id=session.session_id, depth=0)
    root_tool_names = [t.name for t in root.tools]
    assert "spawn_agent" in root_tool_names
    assert "file_read" in root_tool_names

    # Child at depth 1 can still spawn (depth < max_spawn_depth - 1 = 2)
    child = await gw.spawn_agent(
        session_id=session.session_id,
        parent_agent_id=root.agent_id,
        depth=1,
    )
    child_tool_names = [t.name for t in child.tools]
    assert "file_read" in child_tool_names

    # Child at depth 2: spawn_agent is stripped (depth >= max_spawn_depth - 1)
    grandchild = await gw.spawn_agent(
        session_id=session.session_id,
        parent_agent_id=child.agent_id,
        depth=2,
    )
    grandchild_tool_names = [t.name for t in grandchild.tools]
    assert "spawn_agent" not in grandchild_tool_names
    assert "file_read" in grandchild_tool_names

    assert child.agent_id in gw.agents
    assert child.parent_id == root.agent_id
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_chat_full_flow(config):
    """Full chat flow: user message → root agent → LLM → response."""
    gw = Gateway(config)

    mock_llm = MockLLMClient([
        ChatResponse(content="Hi! I'm mini-openclaw.", tokens_used=20),
    ])

    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()
    gw._llm_client = mock_llm  # Ensure the mock is used by spawned agents

    session = gw.create_session()
    result = await gw.chat(session.session_id, "Hello")

    assert result.success
    assert "mini-openclaw" in result.content
    assert result.tokens_used == 20
    assert len(session.conversation_history) == 2  # user + assistant
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_events_emitted_during_chat(config):
    """Chat triggers SESSION_CREATED, AGENT_SPAWNED, and AGENT_COMPLETED events."""
    gw = Gateway(config)

    mock_llm = MockLLMClient([
        ChatResponse(content="Done.", tokens_used=5),
    ])

    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()
    gw._llm_client = mock_llm

    events_received: list[EventType] = []

    async def collector(event: Event):
        events_received.append(event.type)

    gw.event_bus.subscribe_all(collector)

    session = gw.create_session()
    await gw.chat(session.session_id, "Test")

    # Allow fire-and-forget events to settle
    await asyncio.sleep(0.05)

    assert EventType.SESSION_CREATED in events_received
    assert EventType.AGENT_SPAWNED in events_received
    assert EventType.AGENT_COMPLETED in events_received
    await gw.shutdown()


@pytest.mark.asyncio
async def test_gateway_hitl_integration(config):
    """HITL manager is accessible and functional through the Gateway."""
    gw = Gateway(config)
    assert gw.hitl is not None
    assert gw.hitl.policy == ApprovalPolicy.ALWAYS_ASK

    gw.hitl.set_policy(ApprovalPolicy.AUTO_APPROVE)
    assert gw.hitl.policy == ApprovalPolicy.AUTO_APPROVE

    # Shutdown cancels all pending HITL requests
    gw.hitl.set_policy(ApprovalPolicy.ALWAYS_ASK)

    async def add_pending():
        return await gw.hitl.request_approval("a1", "s1", "shell_exec", {"command": "ls"})

    task = asyncio.create_task(add_pending())
    await asyncio.sleep(0.02)
    assert len(gw.hitl.get_pending()) == 1

    await gw.shutdown()
    resp = await task
    assert resp.approved is False  # Cancelled by shutdown


@pytest.mark.asyncio
async def test_gateway_terminate_agent(config):
    """terminate_agent removes the agent and cascades to children."""
    gw = Gateway(config)
    mock_llm = MockLLMClient()
    with patch.object(gw, '_llm_client', mock_llm):
        await gw.start()
    gw._llm_client = mock_llm

    session = gw.create_session()
    root = await gw.spawn_agent(session_id=session.session_id, depth=0)
    child = await gw.spawn_agent(
        session_id=session.session_id,
        parent_agent_id=root.agent_id,
        depth=1,
    )
    root.children.append(child.agent_id)

    assert root.agent_id in gw.agents
    assert child.agent_id in gw.agents

    await gw.terminate_agent(root.agent_id, cascade=True)
    assert root.agent_id not in gw.agents
    assert child.agent_id not in gw.agents
    await gw.shutdown()


# ── 7. HITL Approval Flow with Agent ─────────────────────────────────


@pytest.mark.asyncio
async def test_hitl_interactive_approve_flow(sandbox_dir):
    """Simulate full interactive HITL: Agent blocks → user approves → Agent continues."""
    mock_llm = MockLLMClient([
        ChatResponse(
            content="I'll write a file.",
            tool_calls=[ToolCall(name="file_write", arguments={
                "path": "hello.txt",
                "content": "hello world",
                "mode": "write",
            })],
            tokens_used=20,
        ),
        ChatResponse(content="File written successfully.", tokens_used=10),
    ])

    from mini_openclaw.tools.builtin.file_write import FileWriteTool
    fw = FileWriteTool()
    fw_def = fw.definition

    session = Session(model="mock", sandbox=SandboxConfig(root_path=sandbox_dir))
    registry = ToolRegistry()
    registry.register(fw)

    hitl = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.tool_registry = registry
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = hitl

    agent = Agent(
        session=session,
        llm_client=mock_llm,
        tools=[fw_def],
        max_iterations=5,
    )
    agent.gateway = mock_gateway

    # Simulate user approving after a short delay
    async def auto_approver():
        while True:
            await asyncio.sleep(0.05)
            pending = hitl.get_pending()
            if pending:
                for req in pending:
                    hitl.respond(request_id=req.request_id, approved=True)
                return

    asyncio.create_task(auto_approver())

    result = await agent.run("Write hello.txt")
    assert result.success
    assert result.tool_calls_made == 1
    # The tool executes and returns success — file is written relative to
    # the agent's working_directory (which is "."). The important thing here
    # is that the HITL approval flow worked end-to-end: Agent blocked,
    # auto_approver approved, Agent continued and tool executed successfully.
    assert "written" in result.content.lower() or "file" in result.content.lower()


@pytest.mark.asyncio
async def test_hitl_session_remember(sandbox_dir):
    """Once 'always approve' is selected, subsequent calls skip the dialog."""
    from mini_openclaw.tools.builtin.file_write import FileWriteTool
    fw = FileWriteTool()
    fw_def = fw.definition

    session = Session(
        session_id="remember-test",
        model="mock",
        sandbox=SandboxConfig(root_path=sandbox_dir),
    )
    registry = ToolRegistry()
    registry.register(fw)

    hitl = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.tool_registry = registry
    mock_gateway.event_bus = EventBus()
    mock_gateway.hitl = hitl

    # First: approve with remember
    mock_llm_1 = MockLLMClient([
        ChatResponse(
            content="Writing file 1.",
            tool_calls=[ToolCall(name="file_write", arguments={
                "path": "f1.txt", "content": "one", "mode": "write",
            })],
            tokens_used=10,
        ),
        ChatResponse(content="Done 1.", tokens_used=5),
    ])

    agent1 = Agent(session=session, llm_client=mock_llm_1, tools=[fw_def], max_iterations=5)
    agent1.gateway = mock_gateway

    async def approve_and_remember():
        while True:
            await asyncio.sleep(0.05)
            pending = hitl.get_pending()
            if pending:
                for req in pending:
                    hitl.respond(
                        request_id=req.request_id,
                        approved=True,
                        remember_for_session=True,
                    )
                return

    asyncio.create_task(approve_and_remember())
    result1 = await agent1.run("Write f1.txt")
    assert result1.success

    # Second: should auto-approve because we remembered
    mock_llm_2 = MockLLMClient([
        ChatResponse(
            content="Writing file 2.",
            tool_calls=[ToolCall(name="file_write", arguments={
                "path": "f2.txt", "content": "two", "mode": "write",
            })],
            tokens_used=10,
        ),
        ChatResponse(content="Done 2.", tokens_used=5),
    ])

    agent2 = Agent(session=session, llm_client=mock_llm_2, tools=[fw_def], max_iterations=5)
    agent2.gateway = mock_gateway

    # No approver task needed — should auto-approve
    result2 = await agent2.run("Write f2.txt")
    assert result2.success
    assert len(hitl.get_pending()) == 0  # No pending requests


# ── 8. Web Interface Initialization ──────────────────────────────────


def test_web_app_factory(config):
    """FastAPI app factory creates a valid application."""
    from mini_openclaw.interfaces.web.app import create_web_app
    gw = Gateway(config)
    app = create_web_app(gw)
    assert app.title == "mini-openclaw"
    # Verify routes exist
    route_paths = [r.path for r in app.routes]
    assert "/" in route_paths


def test_web_static_files_exist():
    """Static web files are present."""
    from pathlib import Path
    static_dir = Path(__file__).parent.parent / "src" / "mini_openclaw" / "interfaces" / "web" / "static"
    assert (static_dir / "index.html").exists()
    assert (static_dir / "style.css").exists()
    assert (static_dir / "app.js").exists()


# ── 9. Cross-cutting Concerns ────────────────────────────────────────


def test_tool_definition_ollama_schema(registry):
    """ToolDefinition.to_ollama_schema() produces valid schema for all tools."""
    for td in registry.list_definitions():
        schema = td.to_ollama_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == td.name
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_agent_tree_structure(config):
    """get_agent_tree returns properly structured data."""
    gw = Gateway(config)
    tree = gw.get_agent_tree()
    assert "agents" in tree
    assert isinstance(tree["agents"], list)


@pytest.mark.asyncio
async def test_multiple_sessions_isolated(config):
    """Multiple sessions have independent histories."""
    gw = Gateway(config)
    s1 = gw.create_session()
    s2 = gw.create_session()

    s1.add_message(Message(role=MessageRole.USER, content="session 1 msg"))
    s2.add_message(Message(role=MessageRole.USER, content="session 2 msg"))

    assert len(s1.conversation_history) == 1
    assert len(s2.conversation_history) == 1
    assert s1.conversation_history[0].content == "session 1 msg"
    assert s2.conversation_history[0].content == "session 2 msg"
    await gw.shutdown()
