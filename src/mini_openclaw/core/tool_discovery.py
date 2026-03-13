"""Dynamic tool discovery - agents can request tools at runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mini_openclaw.tools.base import ToolDefinition

if TYPE_CHECKING:
    from mini_openclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolRecommendation:
    """A recommended tool with relevance score."""
    tool: ToolDefinition
    score: float  # 0.0 - 1.0 relevance
    reason: str


# Keyword → tool name mapping for intent-based discovery
TASK_TOOL_MAP: dict[str, list[str]] = {
    # File operations
    "read": ["file_read"],
    "file": ["file_read", "file_write"],
    "write": ["file_write"],
    "create": ["file_write"],
    "edit": ["file_read", "file_write"],
    "modify": ["file_read", "file_write"],
    "save": ["file_write"],
    # Shell / system
    "run": ["shell_exec"],
    "execute": ["shell_exec", "python_exec"],
    "command": ["shell_exec"],
    "install": ["pip_install", "shell_exec"],
    "process": ["shell_exec"],
    "compile": ["shell_exec"],
    "build": ["shell_exec"],
    # Web / network
    "search": ["web_search"],
    "google": ["web_search"],
    "find": ["web_search", "file_read"],
    "browse": ["browser_control", "web_search"],
    "scrape": ["browser_control", "http_request"],
    "download": ["http_request"],
    "api": ["http_request"],
    "fetch": ["http_request"],
    "url": ["http_request", "browser_control"],
    "website": ["browser_control", "web_search"],
    # Code
    "python": ["python_exec"],
    "code": ["python_exec", "file_write"],
    "script": ["python_exec", "shell_exec"],
    "analyze": ["python_exec", "file_read"],
    "calculate": ["python_exec"],
    "data": ["python_exec", "file_read"],
    # Orchestration
    "delegate": ["spawn_agent"],
    "parallel": ["spawn_agent"],
    "subtask": ["spawn_agent"],
    # Scheduling
    "schedule": ["cron_job"],
    "recurring": ["cron_job"],
    "periodic": ["cron_job"],
    "cron": ["cron_job"],
}


class ToolDiscovery:
    """
    Enables agents to discover and request tools dynamically based on task needs.

    Instead of giving agents ALL tools upfront (which confuses smaller LLMs),
    this service recommends relevant tools based on the task description.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def recommend_tools(
        self,
        task_description: str,
        max_tools: int = 5,
        available_only: bool = True,
    ) -> list[ToolRecommendation]:
        """
        Analyze a task description and recommend relevant tools.
        Uses keyword matching against the task description.
        """
        task_lower = task_description.lower()
        tool_scores: dict[str, float] = {}
        tool_reasons: dict[str, list[str]] = {}

        # Score tools based on keyword matches
        for keyword, tool_names in TASK_TOOL_MAP.items():
            if keyword in task_lower:
                for tool_name in tool_names:
                    tool_scores[tool_name] = tool_scores.get(tool_name, 0) + 1.0
                    if tool_name not in tool_reasons:
                        tool_reasons[tool_name] = []
                    tool_reasons[tool_name].append(keyword)

        # Normalize scores
        max_score = max(tool_scores.values()) if tool_scores else 1.0
        for name in tool_scores:
            tool_scores[name] /= max_score

        # Build recommendations
        registered_names = set(self._registry.list_names()) if available_only else None
        recommendations: list[ToolRecommendation] = []

        all_defs = {d.name: d for d in self._registry.list_definitions()}

        for tool_name, score in sorted(tool_scores.items(), key=lambda x: -x[1]):
            if available_only and registered_names and tool_name not in registered_names:
                continue
            tool_def = all_defs.get(tool_name)
            if not tool_def:
                continue

            keywords = tool_reasons.get(tool_name, [])
            reason = f"Matched keywords: {', '.join(keywords)}"
            recommendations.append(ToolRecommendation(
                tool=tool_def,
                score=score,
                reason=reason,
            ))

            if len(recommendations) >= max_tools:
                break

        return recommendations

    def get_tools_for_role(self, role_name: str) -> list[ToolDefinition]:
        """Get the tool definitions for a given role's allowlist."""
        from mini_openclaw.core.agent_roles import BUILTIN_ROLES
        role = BUILTIN_ROLES.get(role_name)
        if not role or not role.tool_allowlist:
            return self._registry.list_definitions()

        all_defs = {d.name: d for d in self._registry.list_definitions()}
        return [all_defs[name] for name in role.tool_allowlist if name in all_defs]

    def suggest_role(self, task_description: str) -> str | None:
        """Suggest the best agent role for a given task."""
        from mini_openclaw.core.agent_roles import BUILTIN_ROLES

        task_lower = task_description.lower()

        # Simple keyword-based role suggestion
        role_keywords: dict[str, list[str]] = {
            "researcher": ["research", "search", "find out", "look up", "investigate", "gather information"],
            "coder": ["write code", "implement", "develop", "program", "fix bug", "debug", "refactor", "coding"],
            "reviewer": ["review", "check code", "audit", "inspect", "code quality"],
            "sysadmin": ["configure", "deploy", "server", "docker", "kubernetes", "infrastructure", "devops"],
            "analyst": ["analyze", "statistics", "data", "trend", "pattern", "chart", "visualize", "metric"],
            "planner": ["plan", "break down", "decompose", "organize", "roadmap", "strategy"],
        }

        best_role = None
        best_score = 0

        for role_name, keywords in role_keywords.items():
            if role_name not in BUILTIN_ROLES:
                continue
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_role = role_name

        return best_role if best_score > 0 else None
