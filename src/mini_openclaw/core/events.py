"""Async event bus for intra-Gateway communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(Enum):
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STATE_CHANGED = "agent_state_changed"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    TOOL_APPROVAL_REQUESTED = "tool_approval_requested"
    TOOL_APPROVAL_RESPONSE = "tool_approval_response"
    LLM_CHUNK = "llm_chunk"
    LLM_RESPONSE = "llm_response"
    MESSAGE_ADDED = "message_added"
    SESSION_CREATED = "session_created"
    # Inter-agent communication
    AGENT_MESSAGE_SENT = "agent_message_sent"
    AGENT_MESSAGE_RECEIVED = "agent_message_received"
    AGENT_BROADCAST = "agent_broadcast"
    # Health monitoring
    AGENT_HEALTH_CHANGED = "agent_health_changed"
    AGENT_RECOVERY_ACTION = "agent_recovery_action"
    # Shared memory
    SHARED_MEMORY_UPDATED = "shared_memory_updated"
    # Parallel spawning
    PARALLEL_SPAWN_STARTED = "parallel_spawn_started"
    PARALLEL_SPAWN_COMPLETED = "parallel_spawn_completed"


@dataclass
class Event:
    type: EventType
    source_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Callback type: async function taking an Event
EventCallback = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub for real-time UI updates and inter-component messaging."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventCallback]] = defaultdict(list)
        self._global_subscribers: list[EventCallback] = []

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribe to all event types."""
        self._global_subscribers.append(callback)

    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> None:
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            pass

    def unsubscribe_all(self, callback: EventCallback) -> None:
        try:
            self._global_subscribers.remove(callback)
        except ValueError:
            pass

    async def emit(self, event: Event) -> None:
        """Dispatch event to all matching subscribers (non-blocking)."""
        callbacks = list(self._subscribers.get(event.type, []))
        callbacks.extend(self._global_subscribers)
        for callback in callbacks:
            try:
                await callback(event)
            except Exception:
                logger.exception("Error in event callback for %s", event.type.value)

    def emit_nowait(self, event: Event) -> None:
        """Schedule event emission without awaiting (fire-and-forget)."""
        loop = asyncio.get_event_loop()
        loop.create_task(self.emit(event))
