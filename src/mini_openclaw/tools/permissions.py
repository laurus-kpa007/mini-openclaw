"""Tool access control via allow/deny lists."""

from __future__ import annotations

from mini_openclaw.tools.base import ToolDefinition


# Tools removed from child agents by default
DEFAULT_CHILD_DENYLIST = ["spawn_agent"]


def resolve_tools(
    all_tools: list[ToolDefinition],
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
) -> list[ToolDefinition]:
    """
    Filter tools based on allow/deny lists.
    - allowlist: if set, ONLY these tools are included
    - denylist: if set, these tools are excluded
    - allowlist takes precedence if both are set
    """
    if allowlist is not None:
        allow_set = set(allowlist)
        return [t for t in all_tools if t.name in allow_set]

    if denylist is not None:
        deny_set = set(denylist)
        return [t for t in all_tools if t.name not in deny_set]

    return list(all_tools)


def resolve_child_tools(
    parent_tools: list[ToolDefinition],
    child_allowlist: list[str] | None = None,
    child_denylist: list[str] | None = None,
    allow_spawning: bool = False,
) -> list[ToolDefinition]:
    """
    Resolve tools for a child agent.
    By default, child agents cannot spawn further agents.
    """
    effective_denylist = list(child_denylist or [])
    if not allow_spawning and "spawn_agent" not in (child_allowlist or []):
        effective_denylist.extend(DEFAULT_CHILD_DENYLIST)

    return resolve_tools(
        parent_tools,
        allowlist=child_allowlist,
        denylist=effective_denylist if effective_denylist else None,
    )
