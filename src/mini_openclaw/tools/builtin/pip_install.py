"""Built-in pip install tool for dynamic package installation."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class PipInstallTool:
    """Install Python packages via pip, making them available as tools at runtime."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="pip_install",
            description=(
                "Install Python packages using pip. Use this to install libraries "
                "needed for data processing, API access, or other tasks. "
                "The installed packages become available immediately."
            ),
            parameters=[
                ToolParameter(
                    name="packages",
                    type="array",
                    description='List of package specifiers (e.g. ["requests", "pandas>=2.0"])',
                ),
                ToolParameter(
                    name="upgrade",
                    type="boolean",
                    description="Whether to upgrade if already installed (default false)",
                    required=False,
                    default=False,
                ),
            ],
            category="system",
            requires_approval=True,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        packages = arguments.get("packages", [])
        upgrade = arguments.get("upgrade", False)

        if not packages:
            return ToolResult(success=False, output="", error="No packages specified")

        if isinstance(packages, str):
            packages = [packages]

        # Build pip command
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            if proc.returncode == 0:
                # Extract the "Successfully installed ..." line
                lines = stdout_str.strip().splitlines()
                success_lines = [l for l in lines if "Successfully" in l or "already satisfied" in l.lower()]
                summary = "\n".join(success_lines) if success_lines else f"Installed: {', '.join(packages)}"
                return ToolResult(
                    success=True,
                    output=f"{summary}\n\n[full output]\n{stdout_str[-500:]}",
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout_str[-500:] if stdout_str else "",
                    error=f"pip install failed (exit {proc.returncode}):\n{stderr_str[-500:]}",
                )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="pip install timed out after 120s",
            )
        except Exception as e:
            logger.exception("pip install failed")
            return ToolResult(success=False, output="", error=f"pip error: {e}")
