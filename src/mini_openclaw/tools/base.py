"""Tool protocol, definitions, and shared data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    name: str
    type: str  # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    category: str = "general"
    requires_approval: bool = False

    def to_ollama_schema(self) -> dict[str, Any]:
        """Convert to Ollama's expected tool format for /api/chat."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None


@dataclass
class ToolContext:
    """Execution context passed to every tool call."""
    session_id: str
    agent_id: str
    sandbox_root: str | None = None
    allowed_hosts: list[str] | None = None
    working_directory: str = "."


@runtime_checkable
class Tool(Protocol):
    """Protocol that all tools must implement."""

    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...
