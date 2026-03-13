"""Agent health monitoring with heartbeats, timeouts, and auto-recovery."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from mini_openclaw.core.events import Event, EventType

if TYPE_CHECKING:
    from mini_openclaw.core.gateway import Gateway

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # Responding but slow
    UNRESPONSIVE = "unresponsive"  # No heartbeat within threshold
    DEAD = "dead"              # Exceeded max unresponsive time


class RecoveryAction(Enum):
    NONE = "none"
    RESTART = "restart"
    TERMINATE = "terminate"
    ESCALATE = "escalate"  # Notify parent agent


@dataclass
class AgentHealthRecord:
    """Health state tracking for a single agent."""
    agent_id: str
    status: HealthStatus = HealthStatus.HEALTHY
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    heartbeat_count: int = 0
    restart_count: int = 0
    consecutive_failures: int = 0
    iteration_count: int = 0
    tokens_consumed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMonitorConfig:
    heartbeat_interval: float = 5.0       # Seconds between heartbeat checks
    unresponsive_threshold: float = 30.0  # Seconds before marking unresponsive
    dead_threshold: float = 120.0         # Seconds before marking dead
    max_restarts: int = 2                 # Max restart attempts per agent
    auto_recovery: bool = True            # Enable automatic recovery actions
    check_interval: float = 10.0          # How often the monitor runs its sweep


class HealthMonitor:
    """
    Monitors agent health via heartbeats and activity tracking.
    Detects stuck/dead agents and initiates recovery.
    """

    def __init__(self, gateway: Gateway, config: HealthMonitorConfig | None = None) -> None:
        self._gateway = gateway
        self.config = config or HealthMonitorConfig()
        self._records: dict[str, AgentHealthRecord] = {}
        self._monitor_task: asyncio.Task | None = None
        self._running = False
        # Callbacks for custom recovery logic
        self._recovery_handlers: dict[RecoveryAction, Any] = {}

    async def start(self) -> None:
        """Start the background health monitoring loop."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started (interval=%.1fs)", self.config.check_interval)

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    def register_agent(self, agent_id: str) -> None:
        """Start tracking an agent's health."""
        self._records[agent_id] = AgentHealthRecord(agent_id=agent_id)
        logger.debug("Tracking health for agent: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Stop tracking an agent."""
        self._records.pop(agent_id, None)

    def heartbeat(self, agent_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a heartbeat from an agent."""
        record = self._records.get(agent_id)
        if not record:
            return
        now = datetime.now(timezone.utc)
        record.last_heartbeat = now
        record.last_activity = now
        record.heartbeat_count += 1
        if record.status in (HealthStatus.UNRESPONSIVE, HealthStatus.DEGRADED):
            record.status = HealthStatus.HEALTHY
            record.consecutive_failures = 0
        if metadata:
            record.metadata.update(metadata)

    def record_activity(self, agent_id: str, activity_type: str = "tool_call") -> None:
        """Record any activity (tool call, LLM response, etc.)."""
        record = self._records.get(agent_id)
        if record:
            record.last_activity = datetime.now(timezone.utc)
            if activity_type == "iteration":
                record.iteration_count += 1

    def record_tokens(self, agent_id: str, tokens: int) -> None:
        """Track token consumption."""
        record = self._records.get(agent_id)
        if record:
            record.tokens_consumed += tokens

    def record_failure(self, agent_id: str) -> None:
        """Record a failure for an agent."""
        record = self._records.get(agent_id)
        if record:
            record.consecutive_failures += 1

    def get_health(self, agent_id: str) -> AgentHealthRecord | None:
        return self._records.get(agent_id)

    def get_all_health(self) -> dict[str, AgentHealthRecord]:
        return dict(self._records)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all agent health statuses."""
        summary: dict[str, int] = {s.value: 0 for s in HealthStatus}
        for record in self._records.values():
            summary[record.status.value] += 1
        return {
            "total_agents": len(self._records),
            "statuses": summary,
            "agents": {
                aid: {
                    "status": r.status.value,
                    "last_heartbeat": r.last_heartbeat.isoformat(),
                    "heartbeat_count": r.heartbeat_count,
                    "restart_count": r.restart_count,
                    "iterations": r.iteration_count,
                    "tokens": r.tokens_consumed,
                }
                for aid, r in self._records.items()
            },
        }

    async def _monitor_loop(self) -> None:
        """Background loop that checks agent health periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval)
                await self._check_all_agents()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health monitor sweep error")

    async def _check_all_agents(self) -> None:
        """Sweep all tracked agents and update statuses."""
        now = datetime.now(timezone.utc)

        for agent_id, record in list(self._records.items()):
            agent = self._gateway.agents.get(agent_id)
            if not agent:
                self.unregister_agent(agent_id)
                continue

            # Skip already completed/terminated agents
            from mini_openclaw.core.agent import AgentState
            if agent.state in (AgentState.COMPLETED, AgentState.FAILED, AgentState.TERMINATED):
                continue

            elapsed = (now - record.last_heartbeat).total_seconds()
            old_status = record.status

            if elapsed > self.config.dead_threshold:
                record.status = HealthStatus.DEAD
            elif elapsed > self.config.unresponsive_threshold:
                record.status = HealthStatus.UNRESPONSIVE
            elif elapsed > self.config.unresponsive_threshold / 2:
                record.status = HealthStatus.DEGRADED
            else:
                record.status = HealthStatus.HEALTHY

            # Emit event on status change
            if record.status != old_status:
                self._gateway.event_bus.emit_nowait(Event(
                    type=EventType.AGENT_STATE_CHANGED,
                    source_id=agent_id,
                    data={
                        "health_status": record.status.value,
                        "old_health_status": old_status.value,
                        "elapsed_since_heartbeat": elapsed,
                    },
                ))
                logger.warning(
                    "Agent %s health: %s → %s (%.1fs since heartbeat)",
                    agent_id, old_status.value, record.status.value, elapsed,
                )

            # Auto-recovery
            if self.config.auto_recovery:
                action = self._determine_recovery(record)
                if action != RecoveryAction.NONE:
                    await self._execute_recovery(agent_id, record, action)

    def _determine_recovery(self, record: AgentHealthRecord) -> RecoveryAction:
        """Decide what recovery action to take based on health status."""
        if record.status == HealthStatus.DEAD:
            if record.restart_count < self.config.max_restarts:
                return RecoveryAction.RESTART
            return RecoveryAction.TERMINATE
        if record.status == HealthStatus.UNRESPONSIVE:
            return RecoveryAction.ESCALATE
        return RecoveryAction.NONE

    async def _execute_recovery(
        self, agent_id: str, record: AgentHealthRecord, action: RecoveryAction
    ) -> None:
        """Execute a recovery action for an unhealthy agent."""
        logger.info("Recovery action for %s: %s", agent_id, action.value)

        if action == RecoveryAction.TERMINATE:
            await self._gateway.terminate_agent(agent_id, cascade=True)
            self._gateway.event_bus.emit_nowait(Event(
                type=EventType.AGENT_FAILED,
                source_id=agent_id,
                data={"reason": "health_monitor_termination", "restarts": record.restart_count},
            ))

        elif action == RecoveryAction.RESTART:
            record.restart_count += 1
            record.last_heartbeat = datetime.now(timezone.utc)
            record.status = HealthStatus.HEALTHY
            # Reset the agent's heartbeat to give it another chance
            logger.info(
                "Restart attempt %d/%d for agent %s",
                record.restart_count, self.config.max_restarts, agent_id,
            )

        elif action == RecoveryAction.ESCALATE:
            # Notify parent agent via event
            agent = self._gateway.agents.get(agent_id)
            if agent and agent.parent_id:
                self._gateway.event_bus.emit_nowait(Event(
                    type=EventType.AGENT_STATE_CHANGED,
                    source_id=agent_id,
                    data={
                        "escalation": True,
                        "parent_id": agent.parent_id,
                        "health_status": record.status.value,
                        "message": f"Child agent {agent_id} is unresponsive",
                    },
                ))
