"""Shared memory / knowledge store for inter-agent state sharing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    PUBLIC = "public"          # Any agent can read/write
    FAMILY = "family"          # Only agents in the same tree (root + descendants)
    PRIVATE = "private"        # Only the owner agent


@dataclass
class MemoryEntry:
    """A single entry in shared memory."""
    key: str
    value: Any
    owner_id: str              # Agent that created this entry
    access_level: AccessLevel = AccessLevel.FAMILY
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    ttl: float | None = None   # Time-to-live in seconds (None = permanent)


class SharedMemory:
    """
    A session-scoped key-value store that allows agents to share state.
    Supports access control, tagging, and TTL expiration.

    Usage patterns:
    - Research agent stores findings → Coder agent reads them
    - Parent stores subtask list → Children update progress
    - Any agent stores intermediate results for siblings
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._store: dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
        # Track agent family trees for FAMILY access control
        self._parent_map: dict[str, str | None] = {}

    def register_agent(self, agent_id: str, parent_id: str | None = None) -> None:
        """Register an agent for access control resolution."""
        self._parent_map[agent_id] = parent_id

    def unregister_agent(self, agent_id: str) -> None:
        self._parent_map.pop(agent_id, None)

    async def put(
        self,
        key: str,
        value: Any,
        owner_id: str,
        access_level: AccessLevel = AccessLevel.FAMILY,
        tags: list[str] | None = None,
        ttl: float | None = None,
    ) -> MemoryEntry:
        """Store or update a value in shared memory."""
        async with self._lock:
            existing = self._store.get(key)
            if existing:
                # Check write permission
                if not self._can_access(owner_id, existing):
                    raise PermissionError(
                        f"Agent {owner_id} cannot write to key '{key}' "
                        f"owned by {existing.owner_id} (access={existing.access_level.value})"
                    )
                existing.value = value
                existing.updated_at = datetime.now(timezone.utc)
                existing.version += 1
                if tags is not None:
                    existing.tags = tags
                return existing

            entry = MemoryEntry(
                key=key,
                value=value,
                owner_id=owner_id,
                access_level=access_level,
                tags=tags or [],
                ttl=ttl,
            )
            self._store[key] = entry
            logger.debug("SharedMemory[%s]: %s set by %s", self.session_id, key, owner_id)
            return entry

    async def get(self, key: str, requester_id: str) -> Any | None:
        """Get a value from shared memory, respecting access control."""
        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None

            # Check TTL
            if entry.ttl is not None:
                elapsed = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
                if elapsed > entry.ttl:
                    del self._store[key]
                    return None

            if not self._can_access(requester_id, entry):
                logger.warning(
                    "Access denied: %s tried to read '%s' owned by %s",
                    requester_id, key, entry.owner_id,
                )
                return None

            return entry.value

    async def get_entry(self, key: str, requester_id: str) -> MemoryEntry | None:
        """Get the full MemoryEntry (including metadata) for a key."""
        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if not self._can_access(requester_id, entry):
                return None
            return entry

    async def delete(self, key: str, requester_id: str) -> bool:
        """Delete an entry. Only owner or PUBLIC entries can be deleted."""
        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return False
            if entry.owner_id != requester_id and entry.access_level != AccessLevel.PUBLIC:
                return False
            del self._store[key]
            return True

    async def search_by_tag(self, tag: str, requester_id: str) -> list[MemoryEntry]:
        """Find all entries with a given tag that the requester can access."""
        async with self._lock:
            results = []
            for entry in self._store.values():
                if tag in entry.tags and self._can_access(requester_id, entry):
                    results.append(entry)
            return results

    async def list_keys(self, requester_id: str, prefix: str | None = None) -> list[str]:
        """List all accessible keys, optionally filtered by prefix."""
        async with self._lock:
            keys = []
            for key, entry in self._store.items():
                if prefix and not key.startswith(prefix):
                    continue
                if self._can_access(requester_id, entry):
                    keys.append(key)
            return keys

    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired_keys = []
            for key, entry in self._store.items():
                if entry.ttl is not None:
                    elapsed = (now - entry.created_at).total_seconds()
                    if elapsed > entry.ttl:
                        expired_keys.append(key)
            for key in expired_keys:
                del self._store[key]
            return len(expired_keys)

    def _can_access(self, requester_id: str, entry: MemoryEntry) -> bool:
        """Check if an agent can access an entry."""
        if entry.access_level == AccessLevel.PUBLIC:
            return True
        if entry.owner_id == requester_id:
            return True
        if entry.access_level == AccessLevel.PRIVATE:
            return False
        # FAMILY: requester must be in the same tree as owner
        if entry.access_level == AccessLevel.FAMILY:
            owner_root = self._find_root(entry.owner_id)
            requester_root = self._find_root(requester_id)
            return owner_root == requester_root
        return False

    def _find_root(self, agent_id: str) -> str:
        """Walk up parent chain to find root."""
        current = agent_id
        visited: set[str] = set()
        while current in self._parent_map and self._parent_map[current] is not None:
            if current in visited:
                break
            visited.add(current)
            current = self._parent_map[current]  # type: ignore[assignment]
        return current
