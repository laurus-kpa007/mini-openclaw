"""Tests for Dynamic Agent Spawning features:
- Inter-agent communication (AgentCommHub)
- Result aggregation (ResultAggregator)
- Health monitoring (HealthMonitor)
- Shared memory (SharedMemory)
- Agent role templates (RoleRegistry)
- Tool discovery (ToolDiscovery)
"""

import asyncio
import pytest

from mini_openclaw.config import AppConfig
from mini_openclaw.core.agent import Agent, AgentResult, AgentState
from mini_openclaw.core.agent_comm import AgentCommHub, AgentMessage, MessagePriority
from mini_openclaw.core.agent_roles import BUILTIN_ROLES, AgentRole, RoleRegistry
from mini_openclaw.core.events import EventType
from mini_openclaw.core.health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
    HealthStatus,
    RecoveryAction,
)
from mini_openclaw.core.result_aggregator import (
    AggregatedResult,
    AggregationStrategy,
    ChildTask,
    ResultAggregator,
)
from mini_openclaw.core.shared_memory import AccessLevel, SharedMemory
from mini_openclaw.core.tool_discovery import ToolDiscovery
from mini_openclaw.tools.registry import ToolRegistry


# ====================
# AgentCommHub Tests
# ====================


class TestAgentCommHub:
    def setup_method(self):
        self.hub = AgentCommHub()

    def test_register_agent(self):
        mailbox = self.hub.register_agent("agent-1")
        assert mailbox is not None
        assert mailbox.agent_id == "agent-1"

    def test_register_with_parent(self):
        self.hub.register_agent("root")
        self.hub.register_agent("child-1", parent_id="root")
        # Both should be in the family group
        root = self.hub._find_root("child-1")
        assert root == "root"

    @pytest.mark.asyncio
    async def test_send_direct_message(self):
        self.hub.register_agent("sender")
        self.hub.register_agent("receiver")

        msg = AgentMessage(
            sender_id="sender",
            receiver_id="receiver",
            content="Hello from sender",
        )
        delivered = await self.hub.send(msg)
        assert delivered is True

        mailbox = self.hub.get_mailbox("receiver")
        assert mailbox.pending_count == 1

        received = await mailbox.get(timeout=1.0)
        assert received is not None
        assert received.content == "Hello from sender"
        assert received.sender_id == "sender"

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_agent(self):
        msg = AgentMessage(
            sender_id="sender",
            receiver_id="ghost",
            content="Hello",
        )
        delivered = await self.hub.send(msg)
        assert delivered is False

    @pytest.mark.asyncio
    async def test_send_to_parent(self):
        self.hub.register_agent("parent")
        self.hub.register_agent("child", parent_id="parent")

        delivered = await self.hub.send_to_parent("child", "progress update")
        assert delivered is True

        mailbox = self.hub.get_mailbox("parent")
        msg = await mailbox.get(timeout=1.0)
        assert msg is not None
        assert msg.content == "progress update"
        assert msg.sender_id == "child"

    @pytest.mark.asyncio
    async def test_send_to_parent_no_parent(self):
        self.hub.register_agent("orphan")
        delivered = await self.hub.send_to_parent("orphan", "hello?")
        assert delivered is False

    @pytest.mark.asyncio
    async def test_broadcast(self):
        self.hub.register_agent("root")
        self.hub.register_agent("child-1", parent_id="root")
        self.hub.register_agent("child-2", parent_id="root")

        count = await self.hub.broadcast("root", "family:root", "team update")
        assert count == 2  # Both children receive it

        for cid in ["child-1", "child-2"]:
            mailbox = self.hub.get_mailbox(cid)
            msg = await mailbox.get(timeout=1.0)
            assert msg is not None
            assert msg.content == "team update"

    @pytest.mark.asyncio
    async def test_message_priority_ordering(self):
        self.hub.register_agent("receiver")

        # Send low priority first, then urgent
        msg_low = AgentMessage(
            sender_id="a",
            receiver_id="receiver",
            content="low priority",
            priority=MessagePriority.LOW,
        )
        msg_urgent = AgentMessage(
            sender_id="b",
            receiver_id="receiver",
            content="urgent",
            priority=MessagePriority.URGENT,
        )
        await self.hub.send(msg_low)
        await self.hub.send(msg_urgent)

        mailbox = self.hub.get_mailbox("receiver")
        first = await mailbox.get(timeout=1.0)
        assert first.content == "urgent"
        second = await mailbox.get(timeout=1.0)
        assert second.content == "low priority"

    def test_unregister_agent(self):
        self.hub.register_agent("agent-1")
        self.hub.unregister_agent("agent-1")
        assert self.hub.get_mailbox("agent-1") is None

    @pytest.mark.asyncio
    async def test_get_all_messages(self):
        self.hub.register_agent("r")
        for i in range(5):
            msg = AgentMessage(sender_id="s", receiver_id="r", content=f"msg-{i}")
            await self.hub.send(msg)

        mailbox = self.hub.get_mailbox("r")
        messages = await mailbox.get_all()
        assert len(messages) == 5
        assert mailbox.is_empty


# ====================
# SharedMemory Tests
# ====================


class TestSharedMemory:
    def setup_method(self):
        self.mem = SharedMemory("test-session")
        self.mem.register_agent("root")
        self.mem.register_agent("child-1", parent_id="root")
        self.mem.register_agent("child-2", parent_id="root")
        self.mem.register_agent("outsider")

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        await self.mem.put("key1", "value1", owner_id="root")
        result = await self.mem.get("key1", requester_id="root")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_family_access(self):
        await self.mem.put("findings", "important data", owner_id="root", access_level=AccessLevel.FAMILY)
        # Child in same tree can access
        result = await self.mem.get("findings", requester_id="child-1")
        assert result == "important data"
        # Outsider cannot
        result = await self.mem.get("findings", requester_id="outsider")
        assert result is None

    @pytest.mark.asyncio
    async def test_public_access(self):
        await self.mem.put("public-key", "open data", owner_id="root", access_level=AccessLevel.PUBLIC)
        result = await self.mem.get("public-key", requester_id="outsider")
        assert result == "open data"

    @pytest.mark.asyncio
    async def test_private_access(self):
        await self.mem.put("secret", "my data", owner_id="child-1", access_level=AccessLevel.PRIVATE)
        # Owner can access
        result = await self.mem.get("secret", requester_id="child-1")
        assert result == "my data"
        # Sibling cannot
        result = await self.mem.get("secret", requester_id="child-2")
        assert result is None

    @pytest.mark.asyncio
    async def test_version_increment(self):
        entry = await self.mem.put("key", "v1", owner_id="root")
        assert entry.version == 1
        entry = await self.mem.put("key", "v2", owner_id="root")
        assert entry.version == 2

    @pytest.mark.asyncio
    async def test_delete(self):
        await self.mem.put("temp", "data", owner_id="root")
        deleted = await self.mem.delete("temp", requester_id="root")
        assert deleted is True
        result = await self.mem.get("temp", requester_id="root")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_by_non_owner(self):
        await self.mem.put("owned", "data", owner_id="root", access_level=AccessLevel.FAMILY)
        deleted = await self.mem.delete("owned", requester_id="child-1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_search_by_tag(self):
        await self.mem.put("res-1", "data1", owner_id="root", tags=["research"])
        await self.mem.put("res-2", "data2", owner_id="root", tags=["research", "important"])
        await self.mem.put("code-1", "data3", owner_id="root", tags=["code"])

        results = await self.mem.search_by_tag("research", requester_id="child-1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_keys(self):
        await self.mem.put("prefix:a", "1", owner_id="root")
        await self.mem.put("prefix:b", "2", owner_id="root")
        await self.mem.put("other", "3", owner_id="root")

        keys = await self.mem.list_keys(requester_id="root", prefix="prefix:")
        assert len(keys) == 2
        assert all(k.startswith("prefix:") for k in keys)

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        await self.mem.put("temp", "data", owner_id="root", ttl=0.01)
        # Immediately accessible
        result = await self.mem.get("temp", requester_id="root")
        assert result == "data"
        # Wait for expiration
        await asyncio.sleep(0.02)
        result = await self.mem.get("temp", requester_id="root")
        assert result is None


# ====================
# HealthMonitor Tests
# ====================


class TestHealthMonitor:
    def setup_method(self):
        self.config = AppConfig()
        from mini_openclaw.core.gateway import Gateway
        self.gateway = Gateway(self.config)
        self.monitor_config = HealthMonitorConfig(
            heartbeat_interval=1.0,
            unresponsive_threshold=2.0,
            dead_threshold=5.0,
            max_restarts=2,
            check_interval=1.0,
        )
        self.monitor = HealthMonitor(self.gateway, self.monitor_config)

    def test_register_agent(self):
        self.monitor.register_agent("agent-1")
        health = self.monitor.get_health("agent-1")
        assert health is not None
        assert health.status == HealthStatus.HEALTHY

    def test_heartbeat_updates(self):
        self.monitor.register_agent("agent-1")
        self.monitor.heartbeat("agent-1", metadata={"iteration": 1})
        health = self.monitor.get_health("agent-1")
        assert health.heartbeat_count == 1

    def test_record_tokens(self):
        self.monitor.register_agent("agent-1")
        self.monitor.record_tokens("agent-1", 500)
        self.monitor.record_tokens("agent-1", 300)
        health = self.monitor.get_health("agent-1")
        assert health.tokens_consumed == 800

    def test_record_failure(self):
        self.monitor.register_agent("agent-1")
        self.monitor.record_failure("agent-1")
        self.monitor.record_failure("agent-1")
        health = self.monitor.get_health("agent-1")
        assert health.consecutive_failures == 2

    def test_unregister_agent(self):
        self.monitor.register_agent("agent-1")
        self.monitor.unregister_agent("agent-1")
        assert self.monitor.get_health("agent-1") is None

    def test_get_summary(self):
        self.monitor.register_agent("agent-1")
        self.monitor.register_agent("agent-2")
        summary = self.monitor.get_summary()
        assert summary["total_agents"] == 2
        assert "agent-1" in summary["agents"]
        assert "agent-2" in summary["agents"]

    def test_determine_recovery_dead(self):
        from mini_openclaw.core.health_monitor import AgentHealthRecord
        record = AgentHealthRecord(agent_id="test", status=HealthStatus.DEAD, restart_count=0)
        action = self.monitor._determine_recovery(record)
        assert action == RecoveryAction.RESTART

    def test_determine_recovery_dead_max_restarts(self):
        from mini_openclaw.core.health_monitor import AgentHealthRecord
        record = AgentHealthRecord(agent_id="test", status=HealthStatus.DEAD, restart_count=3)
        action = self.monitor._determine_recovery(record)
        assert action == RecoveryAction.TERMINATE

    def test_determine_recovery_healthy(self):
        from mini_openclaw.core.health_monitor import AgentHealthRecord
        record = AgentHealthRecord(agent_id="test", status=HealthStatus.HEALTHY)
        action = self.monitor._determine_recovery(record)
        assert action == RecoveryAction.NONE


# ====================
# AgentRoles Tests
# ====================


class TestAgentRoles:
    def setup_method(self):
        self.registry = RoleRegistry()

    def test_builtin_roles_exist(self):
        names = self.registry.list_names()
        assert "researcher" in names
        assert "coder" in names
        assert "reviewer" in names
        assert "sysadmin" in names
        assert "analyst" in names
        assert "planner" in names

    def test_get_role(self):
        role = self.registry.get("researcher")
        assert role is not None
        assert role.name == "researcher"
        assert role.tool_allowlist is not None
        assert "web_search" in role.tool_allowlist

    def test_custom_role(self):
        custom = AgentRole(
            name="translator",
            description="Language translation specialist",
            system_prompt="You are a translator.",
            tool_allowlist=["web_search"],
        )
        self.registry.register(custom)
        assert self.registry.get("translator") is not None

    def test_unregister_custom_role(self):
        custom = AgentRole(
            name="temp_role",
            description="Temporary",
            system_prompt="Temp",
        )
        self.registry.register(custom)
        self.registry.unregister("temp_role")
        assert self.registry.get("temp_role") is None

    def test_cannot_unregister_builtin(self):
        self.registry.unregister("researcher")
        assert self.registry.get("researcher") is not None

    def test_coder_has_file_tools(self):
        role = self.registry.get("coder")
        assert "file_read" in role.tool_allowlist
        assert "file_write" in role.tool_allowlist

    def test_reviewer_is_read_only(self):
        role = self.registry.get("reviewer")
        assert "file_read" in role.tool_allowlist
        assert "file_write" not in role.tool_allowlist


# ====================
# ToolDiscovery Tests
# ====================


class TestToolDiscovery:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register_builtin_tools()
        self.discovery = ToolDiscovery(self.registry)

    def test_recommend_web_search(self):
        recs = self.discovery.recommend_tools("search the web for Python tutorials")
        tool_names = [r.tool.name for r in recs]
        assert "web_search" in tool_names

    def test_recommend_file_ops(self):
        recs = self.discovery.recommend_tools("read the configuration file and modify it")
        tool_names = [r.tool.name for r in recs]
        assert "file_read" in tool_names
        assert "file_write" in tool_names

    def test_recommend_shell(self):
        recs = self.discovery.recommend_tools("run the build command")
        tool_names = [r.tool.name for r in recs]
        assert "shell_exec" in tool_names

    def test_recommend_python(self):
        recs = self.discovery.recommend_tools("analyze the data using python")
        tool_names = [r.tool.name for r in recs]
        assert "python_exec" in tool_names

    def test_max_tools_limit(self):
        recs = self.discovery.recommend_tools(
            "search web, read files, write code, run commands, browse websites",
            max_tools=3,
        )
        assert len(recs) <= 3

    def test_suggest_role_researcher(self):
        role = self.discovery.suggest_role("research the latest AI trends and gather information")
        assert role == "researcher"

    def test_suggest_role_coder(self):
        role = self.discovery.suggest_role("implement a new login feature and write code")
        assert role == "coder"

    def test_suggest_role_reviewer(self):
        role = self.discovery.suggest_role("review the pull request and check code quality")
        assert role == "reviewer"

    def test_suggest_role_none(self):
        role = self.discovery.suggest_role("hello world")
        assert role is None

    def test_get_tools_for_role(self):
        tools = self.discovery.get_tools_for_role("researcher")
        tool_names = [t.name for t in tools]
        assert "web_search" in tool_names


# ====================
# ResultAggregator Tests
# ====================


class TestResultAggregator:
    def test_child_task_creation(self):
        task = ChildTask(
            task_description="Research AI",
            tool_allowlist=["web_search"],
        )
        assert task.task_description == "Research AI"
        assert task.tool_allowlist == ["web_search"]
        assert task.priority == 0

    def test_aggregated_result(self):
        result = AggregatedResult(
            success=True,
            summary="Test summary",
            strategy_used="wait_all",
            children_total=3,
            children_succeeded=2,
            children_failed=1,
        )
        assert result.success is True
        assert result.children_total == 3

    def test_aggregation_strategy_values(self):
        assert AggregationStrategy.WAIT_ALL.value == "wait_all"
        assert AggregationStrategy.FIRST_SUCCESS.value == "first_success"
        assert AggregationStrategy.MAJORITY_VOTE.value == "majority_vote"
        assert AggregationStrategy.PRIORITY_CHAIN.value == "priority_chain"


# ====================
# Integration: New EventTypes
# ====================


class TestNewEventTypes:
    def test_comm_events_exist(self):
        assert EventType.AGENT_MESSAGE_SENT.value == "agent_message_sent"
        assert EventType.AGENT_MESSAGE_RECEIVED.value == "agent_message_received"
        assert EventType.AGENT_BROADCAST.value == "agent_broadcast"

    def test_health_events_exist(self):
        assert EventType.AGENT_HEALTH_CHANGED.value == "agent_health_changed"
        assert EventType.AGENT_RECOVERY_ACTION.value == "agent_recovery_action"

    def test_memory_events_exist(self):
        assert EventType.SHARED_MEMORY_UPDATED.value == "shared_memory_updated"

    def test_parallel_events_exist(self):
        assert EventType.PARALLEL_SPAWN_STARTED.value == "parallel_spawn_started"
        assert EventType.PARALLEL_SPAWN_COMPLETED.value == "parallel_spawn_completed"


# ====================
# Integration: Gateway with new features
# ====================


class TestGatewayNewFeatures:
    def setup_method(self):
        self.config = AppConfig()
        from mini_openclaw.core.gateway import Gateway
        self.gateway = Gateway(self.config)

    def test_gateway_has_comm_hub(self):
        assert self.gateway.comm_hub is not None

    def test_gateway_has_health_monitor(self):
        assert self.gateway.health_monitor is not None

    def test_gateway_has_role_registry(self):
        assert self.gateway.role_registry is not None
        assert len(self.gateway.role_registry.list_names()) >= 6

    def test_session_creates_shared_memory(self):
        session = self.gateway.create_session()
        mem = self.gateway.get_shared_memory(session.session_id)
        assert mem is not None

    def test_agent_tree_includes_health(self):
        # Verify the _agent_to_dict method includes health field
        from mini_openclaw.core.agent import Agent
        agent = Agent(agent_id="test-agent")
        self.gateway.agents["test-agent"] = agent
        self.gateway.health_monitor.register_agent("test-agent")

        tree = self.gateway._agent_to_dict(agent)
        assert "health" in tree
        assert tree["health"] == "healthy"

        # Cleanup
        del self.gateway.agents["test-agent"]

    def test_new_tools_registered(self):
        """Verify the new tools are available in the registry."""
        self.gateway.tool_registry.register_builtin_tools()
        names = self.gateway.tool_registry.list_names()
        assert "spawn_parallel" in names
        assert "agent_comm" in names
        assert "shared_memory" in names
