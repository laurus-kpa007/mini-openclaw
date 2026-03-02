"""Built-in tool for managing scheduled/cron jobs."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

if TYPE_CHECKING:
    from mini_openclaw.core.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Global reference set by Gateway when scheduler is initialized
_scheduler: Scheduler | None = None


def set_scheduler(scheduler: Scheduler) -> None:
    """Called by Gateway to make the scheduler accessible to this tool."""
    global _scheduler
    _scheduler = scheduler


class CronJobTool:
    """Create, list, pause, resume, and remove scheduled jobs."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="cron_job",
            description=(
                "Manage scheduled recurring tasks. "
                "Create jobs that run periodically (e.g., every hour, daily at 9am). "
                "Supports cron expressions (e.g., '0 9 * * *') and intervals (e.g., '1h', '30m'). "
                "Each job spawns a new agent that executes the given task with full tool access."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform",
                    enum=["create", "list", "remove", "pause", "resume", "history"],
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Job name (for 'create')",
                    required=False,
                ),
                ToolParameter(
                    name="task",
                    type="string",
                    description="Task description / prompt for the agent to execute (for 'create')",
                    required=False,
                ),
                ToolParameter(
                    name="schedule",
                    type="string",
                    description=(
                        "Schedule: cron expression ('0 9 * * *' = daily 9am, "
                        "'*/30 * * * *' = every 30min) or interval ('30s', '5m', '2h', '1d')"
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="job_id",
                    type="string",
                    description="Job ID (for remove/pause/resume/history)",
                    required=False,
                ),
                ToolParameter(
                    name="auto_approve",
                    type="boolean",
                    description="Auto-approve all HITL prompts for this job (default: false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="max_runs",
                    type="integer",
                    description="Max number of runs (0 = unlimited, default: 0)",
                    required=False,
                    default=0,
                ),
            ],
            category="system",
            requires_approval=True,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if _scheduler is None:
            return ToolResult(
                success=False, output="",
                error="Scheduler is not available. Use 'web' or 'chat' mode with scheduler enabled.",
            )

        action = arguments.get("action", "")

        if action == "create":
            return self._create(arguments)
        elif action == "list":
            return self._list()
        elif action == "remove":
            return self._remove(arguments)
        elif action == "pause":
            return self._pause(arguments)
        elif action == "resume":
            return self._resume(arguments)
        elif action == "history":
            return self._history(arguments)
        else:
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")

    def _create(self, args: dict[str, Any]) -> ToolResult:
        name = args.get("name", "")
        task = args.get("task", "")
        schedule = args.get("schedule", "")
        auto_approve = args.get("auto_approve", False)
        max_runs = int(args.get("max_runs", 0))

        if not task:
            return ToolResult(success=False, output="", error="'task' is required for creating a job")
        if not schedule:
            return ToolResult(success=False, output="", error="'schedule' is required for creating a job")
        if not name:
            name = task[:50]

        job = _scheduler.add_job(
            name=name,
            task=task,
            schedule=schedule,
            auto_approve=auto_approve,
            max_runs=max_runs,
        )

        return ToolResult(
            success=True,
            output=(
                f"Job created successfully!\n"
                f"  ID: {job.job_id}\n"
                f"  Name: {job.name}\n"
                f"  Schedule: {job.schedule}\n"
                f"  Auto-approve: {job.auto_approve}\n"
                f"  Max runs: {job.max_runs or 'unlimited'}\n"
                f"  Task: {job.task}"
            ),
        )

    def _list(self) -> ToolResult:
        jobs = _scheduler.list_jobs()
        if not jobs:
            return ToolResult(success=True, output="No scheduled jobs.")

        lines = [f"Scheduled Jobs ({len(jobs)}):"]
        for j in jobs:
            lines.append(
                f"  [{j.status.value}] {j.job_id}: {j.name}\n"
                f"    Schedule: {j.schedule} | Runs: {j.run_count}"
                + (f"/{j.max_runs}" if j.max_runs else "")
                + (f" | Last: {j.last_run.strftime('%Y-%m-%d %H:%M')}" if j.last_run else "")
            )
        return ToolResult(success=True, output="\n".join(lines))

    def _remove(self, args: dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id", "")
        if not job_id:
            return ToolResult(success=False, output="", error="'job_id' is required")
        ok = _scheduler.remove_job(job_id)
        if ok:
            return ToolResult(success=True, output=f"Job {job_id} removed.")
        return ToolResult(success=False, output="", error=f"Job '{job_id}' not found")

    def _pause(self, args: dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id", "")
        if not job_id:
            return ToolResult(success=False, output="", error="'job_id' is required")
        ok = _scheduler.pause_job(job_id)
        if ok:
            return ToolResult(success=True, output=f"Job {job_id} paused.")
        return ToolResult(success=False, output="", error=f"Job '{job_id}' not found or not active")

    def _resume(self, args: dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id", "")
        if not job_id:
            return ToolResult(success=False, output="", error="'job_id' is required")
        ok = _scheduler.resume_job(job_id)
        if ok:
            return ToolResult(success=True, output=f"Job {job_id} resumed.")
        return ToolResult(success=False, output="", error=f"Job '{job_id}' not found or not paused")

    def _history(self, args: dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id", "")
        if not job_id:
            return ToolResult(success=False, output="", error="'job_id' is required")
        job = _scheduler.get_job(job_id)
        if not job:
            return ToolResult(success=False, output="", error=f"Job '{job_id}' not found")

        if not job.history:
            return ToolResult(success=True, output=f"No runs yet for job '{job.name}'")

        lines = [f"Run history for '{job.name}' ({len(job.history)} runs):"]
        for run in reversed(job.history[-10:]):  # Show last 10
            status = "OK" if run.success else "FAIL"
            duration = ""
            if run.finished_at and run.started_at:
                secs = (run.finished_at - run.started_at).total_seconds()
                duration = f" ({secs:.1f}s)"
            lines.append(
                f"  [{status}] {run.run_id} @ {run.started_at.strftime('%Y-%m-%d %H:%M')}{duration}\n"
                f"    {run.result_summary[:200] or run.error[:200]}"
            )
        return ToolResult(success=True, output="\n".join(lines))
