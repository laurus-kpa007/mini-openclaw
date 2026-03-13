"""Inter-agent communication protocol via async mailboxes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """A message sent between agents."""
    sender_id: str
    receiver_id: str
    content: str
    msg_type: str = "info"  # info, request, response, broadcast
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: str | None = None  # Links request/response pairs
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AgentMailbox:
    """
    Async mailbox for an individual agent.
    Uses a priority queue so urgent messages are delivered first.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._queue: asyncio.PriorityQueue[tuple[int, float, AgentMessage]] = (
            asyncio.PriorityQueue()
        )
        self._unread_count = 0

    async def put(self, message: AgentMessage) -> None:
        # Lower number = higher priority (URGENT=3 maps to sort key -3)
        priority_key = -message.priority.value
        ts = message.timestamp.timestamp()
        await self._queue.put((priority_key, ts, message))
        self._unread_count += 1

    async def get(self, timeout: float | None = None) -> AgentMessage | None:
        """Get next message, optionally with timeout."""
        try:
            if timeout is not None:
                _, _, msg = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                _, _, msg = self._queue.get_nowait()
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return None
        self._unread_count = max(0, self._unread_count - 1)
        return msg

    async def get_all(self) -> list[AgentMessage]:
        """Drain all pending messages."""
        messages: list[AgentMessage] = []
        while not self._queue.empty():
            try:
                _, _, msg = self._queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        self._unread_count = 0
        return messages

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()


class AgentCommHub:
    """
    Central communication hub that manages mailboxes and message routing.
    Supports direct messaging, parent/child messaging, and broadcasts.
    """

    def __init__(self) -> None:
        self._mailboxes: dict[str, AgentMailbox] = {}
        self._broadcast_groups: dict[str, set[str]] = defaultdict(set)
        # Track parent-child relationships for scoped broadcasts
        self._parent_map: dict[str, str | None] = {}

    def register_agent(self, agent_id: str, parent_id: str | None = None) -> AgentMailbox:
        """Register an agent and create its mailbox."""
        if agent_id not in self._mailboxes:
            self._mailboxes[agent_id] = AgentMailbox(agent_id)
        self._parent_map[agent_id] = parent_id
        # Auto-join session-level broadcast group if parent exists
        if parent_id:
            # Find the root agent to determine the "family" group
            root = self._find_root(agent_id)
            group_name = f"family:{root}"
            self._broadcast_groups[group_name].add(agent_id)
            # Ensure parent is also in the group
            self._broadcast_groups[group_name].add(parent_id)
        return self._mailboxes[agent_id]

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's mailbox and group memberships."""
        self._mailboxes.pop(agent_id, None)
        self._parent_map.pop(agent_id, None)
        for group in self._broadcast_groups.values():
            group.discard(agent_id)

    async def send(self, message: AgentMessage) -> bool:
        """Send a direct message to a specific agent."""
        mailbox = self._mailboxes.get(message.receiver_id)
        if not mailbox:
            logger.warning(
                "Cannot deliver message to %s: no mailbox", message.receiver_id
            )
            return False
        await mailbox.put(message)
        logger.debug(
            "Message from %s → %s (type=%s)",
            message.sender_id, message.receiver_id, message.msg_type,
        )
        return True

    async def send_to_parent(self, agent_id: str, content: str, **kwargs: Any) -> bool:
        """Send a message to an agent's parent."""
        parent_id = self._parent_map.get(agent_id)
        if not parent_id:
            return False
        msg = AgentMessage(
            sender_id=agent_id,
            receiver_id=parent_id,
            content=content,
            msg_type=kwargs.get("msg_type", "info"),
            priority=kwargs.get("priority", MessagePriority.NORMAL),
            metadata=kwargs.get("metadata", {}),
        )
        return await self.send(msg)

    async def send_to_children(
        self, agent_id: str, content: str, children_ids: list[str], **kwargs: Any
    ) -> int:
        """Send a message to all children of an agent. Returns count delivered."""
        delivered = 0
        for child_id in children_ids:
            msg = AgentMessage(
                sender_id=agent_id,
                receiver_id=child_id,
                content=content,
                msg_type=kwargs.get("msg_type", "info"),
                priority=kwargs.get("priority", MessagePriority.NORMAL),
                metadata=kwargs.get("metadata", {}),
            )
            if await self.send(msg):
                delivered += 1
        return delivered

    async def broadcast(
        self, sender_id: str, group_name: str, content: str, **kwargs: Any
    ) -> int:
        """Broadcast a message to all agents in a group. Returns count delivered."""
        members = self._broadcast_groups.get(group_name, set())
        delivered = 0
        for member_id in members:
            if member_id == sender_id:
                continue  # Don't send to self
            msg = AgentMessage(
                sender_id=sender_id,
                receiver_id=member_id,
                content=content,
                msg_type="broadcast",
                priority=kwargs.get("priority", MessagePriority.NORMAL),
                metadata=kwargs.get("metadata", {}),
            )
            if await self.send(msg):
                delivered += 1
        return delivered

    def join_group(self, agent_id: str, group_name: str) -> None:
        self._broadcast_groups[group_name].add(agent_id)

    def leave_group(self, agent_id: str, group_name: str) -> None:
        self._broadcast_groups[group_name].discard(agent_id)

    def get_mailbox(self, agent_id: str) -> AgentMailbox | None:
        return self._mailboxes.get(agent_id)

    def _find_root(self, agent_id: str) -> str:
        """Walk up parent chain to find the root agent."""
        current = agent_id
        visited: set[str] = set()
        while current in self._parent_map and self._parent_map[current] is not None:
            if current in visited:
                break  # Cycle protection
            visited.add(current)
            current = self._parent_map[current]  # type: ignore[assignment]
        return current
