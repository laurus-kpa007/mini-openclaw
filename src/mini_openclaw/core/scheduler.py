"""Task scheduler for recurring agent jobs (cron-like).

Allows users to register periodic tasks that automatically spawn agents
at specified intervals. Each job runs a full agent ReAct loop with the
original user message, using the gateway's tools and HITL policy.

Flow:
  User: "매일 9시에 코스피 지수를 검색해서 report.txt에 저장해줘"
  → Agent calls cron_job(schedule="0 9 * * *", task="코스피 지수를 ...")
  → Scheduler registers the job
  → At 09:00 daily, Scheduler creates a session + agent → runs the task
  → Results stored in job history
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mini_openclaw.core.gateway import Gateway

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"  # one-shot jobs
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRun:
    """Record of a single execution of a scheduled job."""
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    success: bool = False
    result_summary: str = ""
    error: str = ""


@dataclass
class ScheduledJob:
    """A registered recurring task."""
    job_id: str
    name: str
    task: str  # The user message / prompt to execute
    schedule: str  # Cron expression or interval string
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: JobStatus = JobStatus.ACTIVE
    auto_approve: bool = False  # Whether to auto-approve HITL for this job
    tool_allowlist: list[str] | None = None  # Restrict which tools the job can use
    max_runs: int = 0  # 0 = unlimited
    run_count: int = 0
    last_run: datetime | None = None
    history: list[JobRun] = field(default_factory=list)
    max_history: int = 20  # Keep last N runs


def parse_interval_seconds(schedule: str) -> int | None:
    """
    Parse simple interval strings into seconds.
    Supports: "30s", "5m", "2h", "1d", and cron-like expressions.
    Returns None for cron expressions (handled separately).
    """
    schedule = schedule.strip().lower()

    # Simple interval formats
    if schedule.endswith("s") and schedule[:-1].isdigit():
        return int(schedule[:-1])
    if schedule.endswith("m") and schedule[:-1].isdigit():
        return int(schedule[:-1]) * 60
    if schedule.endswith("h") and schedule[:-1].isdigit():
        return int(schedule[:-1]) * 3600
    if schedule.endswith("d") and schedule[:-1].isdigit():
        return int(schedule[:-1]) * 86400

    return None  # Assume cron expression


def cron_matches_now(cron_expr: str, now: datetime) -> bool:
    """
    Simple cron matcher supporting: minute hour day month weekday
    Supports * (any), specific numbers, and */N (every N).
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False

    fields = [now.minute, now.hour, now.day, now.month, now.weekday()]
    # weekday: cron uses 0=Sun..6=Sat, Python uses 0=Mon..6=Sun
    # Convert Python weekday to cron: Mon=1..Sun=0
    fields[4] = (fields[4] + 1) % 7

    for field_val, pattern in zip(fields, parts):
        if pattern == "*":
            continue
        if pattern.startswith("*/"):
            step = int(pattern[2:])
            if step > 0 and field_val % step != 0:
                return False
        elif "," in pattern:
            allowed = {int(x) for x in pattern.split(",")}
            if field_val not in allowed:
                return False
        else:
            if field_val != int(pattern):
                return False

    return True


class Scheduler:
    """
    Manages scheduled jobs. Runs a background loop that checks
    every minute whether any jobs should execute.
    """

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway
        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def jobs(self) -> dict[str, ScheduledJob]:
        return self._jobs

    def add_job(
        self,
        name: str,
        task: str,
        schedule: str,
        auto_approve: bool = False,
        tool_allowlist: list[str] | None = None,
        max_runs: int = 0,
    ) -> ScheduledJob:
        """Register a new scheduled job."""
        job_id = f"job:{uuid.uuid4().hex[:8]}"
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            task=task,
            schedule=schedule,
            auto_approve=auto_approve,
            tool_allowlist=tool_allowlist,
            max_runs=max_runs,
        )
        self._jobs[job_id] = job
        logger.info("Scheduled job '%s' (%s): %s", name, schedule, job_id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            del self._jobs[job_id]
            logger.info("Removed job: %s", job_id)
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.ACTIVE:
            job.status = JobStatus.PAUSED
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.status = JobStatus.ACTIVE
            return True
        return False

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> ScheduledJob | None:
        return self._jobs.get(job_id)

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop — checks jobs every 30 seconds."""
        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Scheduler tick error")
            await asyncio.sleep(30)

    async def _tick(self) -> None:
        """Check all active jobs and run those that are due."""
        now = datetime.now(timezone.utc)

        for job in list(self._jobs.values()):
            if job.status != JobStatus.ACTIVE:
                continue

            if self._is_due(job, now):
                # Don't block the scheduler — run in background
                asyncio.create_task(self._execute_job(job))

    def _is_due(self, job: ScheduledJob, now: datetime) -> bool:
        """Check if a job should run now."""
        # Check max_runs limit
        if job.max_runs > 0 and job.run_count >= job.max_runs:
            job.status = JobStatus.COMPLETED
            return False

        interval = parse_interval_seconds(job.schedule)

        if interval is not None:
            # Interval-based: check time since last run
            if job.last_run is None:
                return True
            elapsed = (now - job.last_run).total_seconds()
            return elapsed >= interval
        else:
            # Cron-based: check if current minute matches
            # Avoid running twice in the same minute
            if job.last_run and (now - job.last_run).total_seconds() < 60:
                return False
            return cron_matches_now(job.schedule, now)

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job by spawning an agent."""
        run_id = f"run:{uuid.uuid4().hex[:8]}"
        run = JobRun(run_id=run_id, started_at=datetime.now(timezone.utc))

        logger.info("Executing job '%s' (%s), run: %s", job.name, job.job_id, run_id)

        # Set auto-approve if configured
        original_policy = None
        if job.auto_approve:
            from mini_openclaw.core.hitl import ApprovalPolicy
            original_policy = self.gateway.hitl.policy
            self.gateway.hitl.set_policy(ApprovalPolicy.AUTO_APPROVE)

        try:
            session = self.gateway.create_session()
            result = await self.gateway.chat(session.session_id, job.task)

            run.success = result.success
            run.result_summary = result.content[:500] if result.content else ""
            run.finished_at = datetime.now(timezone.utc)

            logger.info(
                "Job '%s' run %s: success=%s, tools=%d",
                job.name, run_id, result.success, result.tool_calls_made,
            )

        except Exception as e:
            logger.exception("Job '%s' run %s failed", job.name, run_id)
            run.success = False
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc)

        finally:
            # Restore HITL policy
            if original_policy is not None:
                self.gateway.hitl.set_policy(original_policy)

        # Update job state
        job.run_count += 1
        job.last_run = datetime.now(timezone.utc)
        job.history.append(run)
        if len(job.history) > job.max_history:
            job.history = job.history[-job.max_history:]

        # Check if max_runs reached
        if job.max_runs > 0 and job.run_count >= job.max_runs:
            job.status = JobStatus.COMPLETED
            logger.info("Job '%s' completed after %d runs", job.name, job.run_count)
