"""Built-in shell execution tool with security restrictions."""

from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Commands that are always blocked
ALWAYS_BLOCKED = {
    "rm -rf /", "rm -rf /*", "format c:", "format c:/",
    "mkfs", "dd if=/dev/zero", ":(){:|:&};:",
}


class ShellExecTool:
    """Execute shell commands with sandbox and timeout restrictions."""

    def __init__(self, blocked_commands: list[str] | None = None) -> None:
        self._blocked = set(ALWAYS_BLOCKED)
        if blocked_commands:
            self._blocked.update(blocked_commands)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shell_exec",
            description=(
                "Execute a shell command and return its stdout/stderr. "
                "Use for running system commands, scripts, or CLI tools."
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The shell command to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30)",
                    required=False,
                    default=30,
                ),
            ],
            category="system",
            requires_approval=True,
        )

    def _is_blocked(self, command: str) -> bool:
        """Check if a command is in the blocklist."""
        cmd_lower = command.strip().lower()
        for blocked in self._blocked:
            if blocked.lower() in cmd_lower:
                return True
        return False

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = arguments.get("command", "")
        timeout = int(arguments.get("timeout", 30))

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        if self._is_blocked(command):
            return ToolResult(
                success=False,
                output="",
                error=f"Command blocked for security reasons: {command}",
            )

        cwd = context.working_directory
        if context.sandbox_root:
            cwd = context.sandbox_root

        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "bash", "-c", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            output_parts = []
            if stdout_str:
                output_parts.append(f"[stdout]\n{stdout_str}")
            if stderr_str:
                output_parts.append(f"[stderr]\n{stderr_str}")
            output_parts.append(f"[exit code: {proc.returncode}]")

            return ToolResult(
                success=proc.returncode == 0,
                output="\n".join(output_parts),
                error=stderr_str if proc.returncode != 0 else None,
            )
        except Exception as e:
            logger.exception("Shell execution failed")
            return ToolResult(success=False, output="", error=f"Execution error: {e}")
