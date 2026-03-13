"""Configuration loading from YAML + environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    max_concurrent_agents: int = 8
    max_spawn_depth: int = 3
    max_children_per_agent: int = 5
    max_iterations_per_agent: int = 20


class LLMConfig(BaseModel):
    provider: str = "ollama"  # "ollama" or "lmstudio"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    context_window: int = 8192
    timeout: float = 120.0
    api_key: str = "lm-studio"  # API key for LM Studio (default is fine)


class SessionConfig(BaseModel):
    default_token_budget: int = 8192
    max_turns: int = 50
    max_tool_output_chars: int = 4000


class SecurityConfig(BaseModel):
    sandbox_root: str | None = None
    allowed_shell_commands: list[str] | None = None
    blocked_shell_commands: list[str] = Field(default_factory=lambda: ["rm -rf /", "format", "mkfs"])
    allowed_hosts: list[str] | None = None
    require_approval_for: list[str] = Field(default_factory=lambda: ["file_write", "shell_exec"])


class ToolsConfig(BaseModel):
    plugin_dirs: list[str] = Field(default_factory=lambda: ["./plugins"])
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)


class CLIConfig(BaseModel):
    theme: str = "dark"


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class InterfaceConfig(BaseModel):
    cli: CLIConfig = Field(default_factory=CLIConfig)
    web: WebConfig = Field(default_factory=WebConfig)


class AppConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    path = Path(config_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
