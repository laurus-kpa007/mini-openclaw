"""Custom exception hierarchy for mini-openclaw."""


class MiniOpenClawError(Exception):
    """Base exception for all mini-openclaw errors."""


class ConfigError(MiniOpenClawError):
    """Configuration-related errors."""


class AgentError(MiniOpenClawError):
    """Agent lifecycle errors."""


class AgentSpawnError(AgentError):
    """Failed to spawn a child agent."""


class AgentDepthLimitError(AgentSpawnError):
    """Spawn depth limit exceeded."""


class AgentConcurrencyError(AgentSpawnError):
    """Too many concurrent agents."""


class ToolError(MiniOpenClawError):
    """Tool-related errors."""


class ToolNotFoundError(ToolError):
    """Requested tool not found in registry."""


class ToolExecutionError(ToolError):
    """Tool execution failed."""


class ToolPermissionError(ToolError):
    """Agent does not have access to the requested tool."""


class LLMError(MiniOpenClawError):
    """LLM communication errors."""


class LLMConnectionError(LLMError):
    """Cannot connect to LLM backend."""


class LLMResponseError(LLMError):
    """Invalid or unexpected LLM response."""


class SessionError(MiniOpenClawError):
    """Session-related errors."""


class SecurityError(MiniOpenClawError):
    """Security violation (sandbox escape, blocked command, etc.)."""
