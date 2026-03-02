"""Built-in Python code execution tool."""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class PythonExecTool:
    """Execute Python code in a subprocess and return the output."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_exec",
            description=(
                "Execute Python code and return stdout/stderr. "
                "Use this for data processing, calculations, API calls, "
                "or any task that benefits from running Python directly. "
                "Installed packages (via pip_install) are available."
            ),
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
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

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        code = arguments.get("code", "")
        timeout = int(arguments.get("timeout", 30))

        if not code.strip():
            return ToolResult(success=False, output="", error="No code provided")

        # Write code to a temp file and execute in subprocess
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            cwd = context.working_directory
            if context.sandbox_root:
                cwd = context.sandbox_root

            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
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
                    error=f"Python execution timed out after {timeout}s",
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
            logger.exception("Python execution failed")
            return ToolResult(success=False, output="", error=f"Execution error: {e}")
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
