"""Tests for the scheduler and cron_job tool."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_openclaw.core.scheduler import (
    JobStatus,
    JobRun,
    Scheduler,
    ScheduledJob,
    cron_matches_now,
    parse_interval_seconds,
)
from mini_openclaw.tools.builtin.cron_job import CronJobTool, set_scheduler, _scheduler
from mini_openclaw.tools.base import ToolContext, ToolResult


# ── parse_interval_seconds ───────────────────────────────────────────


def test_parse_seconds():
    assert parse_interval_seconds("30s") == 30
    assert parse_interval_seconds("120s") == 120


def test_parse_minutes():
    assert parse_interval_seconds("5m") == 300
    assert parse_interval_seconds("1m") == 60


def test_parse_hours():
    assert parse_interval_seconds("2h") == 7200
    assert parse_interval_seconds("1h") == 3600


def test_parse_days():
    assert parse_interval_seconds("1d") == 86400
    assert parse_interval_seconds("7d") == 604800


def test_parse_cron_returns_none():
    """Cron expressions return None (handled separately)."""
    assert parse_interval_seconds("0 9 * * *") is None
    assert parse_interval_seconds("*/5 * * * *") is None


def test_parse_with_whitespace():
    assert parse_interval_seconds("  30s  ") == 30
    assert parse_interval_seconds(" 2H ") == 7200  # case insensitive


# ── cron_matches_now ─────────────────────────────────────────────────


def test_cron_wildcard_matches_all():
    """'* * * * *' matches any time."""
    dt = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)  # Monday
    assert cron_matches_now("* * * * *", dt) is True


def test_cron_specific_minute_and_hour():
    """'30 14 * * *' matches 14:30."""
    dt = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
    assert cron_matches_now("30 14 * * *", dt) is True
    dt_wrong = datetime(2026, 3, 2, 14, 31, tzinfo=timezone.utc)
    assert cron_matches_now("30 14 * * *", dt_wrong) is False


def test_cron_step_pattern():
    """'*/15 * * * *' matches minute 0, 15, 30, 45."""
    dt_0 = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
    dt_15 = datetime(2026, 3, 2, 10, 15, tzinfo=timezone.utc)
    dt_7 = datetime(2026, 3, 2, 10, 7, tzinfo=timezone.utc)
    assert cron_matches_now("*/15 * * * *", dt_0) is True
    assert cron_matches_now("*/15 * * * *", dt_15) is True
    assert cron_matches_now("*/15 * * * *", dt_7) is False


def test_cron_day_of_week():
    """'0 9 * * 1' matches Monday at 9:00 (cron weekday: 1=Mon)."""
    # 2026-03-02 is a Monday
    dt_mon = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    assert cron_matches_now("0 9 * * 1", dt_mon) is True
    # 2026-03-03 is a Tuesday
    dt_tue = datetime(2026, 3, 3, 9, 0, tzinfo=timezone.utc)
    assert cron_matches_now("0 9 * * 1", dt_tue) is False


def test_cron_comma_list():
    """'0 9 * * 1,3,5' matches Mon, Wed, Fri."""
    dt_mon = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    dt_wed = datetime(2026, 3, 4, 9, 0, tzinfo=timezone.utc)
    dt_thu = datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)
    assert cron_matches_now("0 9 * * 1,3,5", dt_mon) is True
    assert cron_matches_now("0 9 * * 1,3,5", dt_wed) is True
    assert cron_matches_now("0 9 * * 1,3,5", dt_thu) is False


def test_cron_invalid_format():
    """Invalid cron expressions return False."""
    dt = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    assert cron_matches_now("bad", dt) is False
    assert cron_matches_now("1 2 3", dt) is False


# ── Scheduler job CRUD ───────────────────────────────────────────────


@pytest.fixture
def mock_gateway():
    gw = MagicMock()
    gw.hitl = MagicMock()
    gw.create_session = MagicMock()
    gw.chat = AsyncMock()
    return gw


@pytest.fixture
def scheduler(mock_gateway):
    return Scheduler(mock_gateway)


def test_add_job(scheduler):
    job = scheduler.add_job(name="Test", task="do something", schedule="1h")
    assert job.job_id.startswith("job:")
    assert job.name == "Test"
    assert job.task == "do something"
    assert job.schedule == "1h"
    assert job.status == JobStatus.ACTIVE
    assert job.job_id in scheduler.jobs


def test_remove_job(scheduler):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    assert scheduler.remove_job(job.job_id) is True
    assert job.job_id not in scheduler.jobs


def test_remove_nonexistent_job(scheduler):
    assert scheduler.remove_job("nonexistent") is False


def test_pause_and_resume_job(scheduler):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    assert scheduler.pause_job(job.job_id) is True
    assert job.status == JobStatus.PAUSED

    # Can't pause an already paused job
    assert scheduler.pause_job(job.job_id) is False

    assert scheduler.resume_job(job.job_id) is True
    assert job.status == JobStatus.ACTIVE

    # Can't resume an already active job
    assert scheduler.resume_job(job.job_id) is False


def test_list_jobs(scheduler):
    scheduler.add_job(name="A", task="t1", schedule="1h")
    scheduler.add_job(name="B", task="t2", schedule="2h")
    jobs = scheduler.list_jobs()
    assert len(jobs) == 2
    names = {j.name for j in jobs}
    assert names == {"A", "B"}


def test_get_job(scheduler):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    assert scheduler.get_job(job.job_id) is job
    assert scheduler.get_job("nonexistent") is None


def test_add_job_with_max_runs(scheduler):
    job = scheduler.add_job(name="Test", task="t", schedule="1h", max_runs=5)
    assert job.max_runs == 5


# ── Scheduler _is_due ────────────────────────────────────────────────


def test_is_due_interval_first_run(scheduler):
    """First run of interval job is always due."""
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    now = datetime.now(timezone.utc)
    assert scheduler._is_due(job, now) is True


def test_is_due_interval_not_elapsed(scheduler):
    """Interval job is not due if interval hasn't elapsed."""
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    now = datetime.now(timezone.utc)
    job.last_run = now  # Just ran
    assert scheduler._is_due(job, now) is False


def test_is_due_max_runs_reached(scheduler):
    """Job with max_runs reached is marked COMPLETED."""
    job = scheduler.add_job(name="Test", task="t", schedule="1h", max_runs=3)
    job.run_count = 3
    now = datetime.now(timezone.utc)
    assert scheduler._is_due(job, now) is False
    assert job.status == JobStatus.COMPLETED


# ── Scheduler start/stop ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler):
    """Scheduler can start and stop without errors."""
    await scheduler.start()
    assert scheduler._running is True
    assert scheduler._task is not None

    await scheduler.stop()
    assert scheduler._running is False


@pytest.mark.asyncio
async def test_scheduler_double_start(scheduler):
    """Starting twice does not create duplicate tasks."""
    await scheduler.start()
    task1 = scheduler._task
    await scheduler.start()  # second call should be no-op
    assert scheduler._task is task1
    await scheduler.stop()


# ── CronJobTool ──────────────────────────────────────────────────────


@pytest.fixture
def cron_tool(scheduler):
    set_scheduler(scheduler)
    return CronJobTool()


@pytest.fixture
def tool_ctx():
    return ToolContext(session_id="s1", agent_id="a1")


def test_cron_tool_definition(cron_tool):
    d = cron_tool.definition
    assert d.name == "cron_job"
    assert d.requires_approval is True
    assert d.category == "system"
    param_names = {p.name for p in d.parameters}
    assert "action" in param_names
    assert "schedule" in param_names
    assert "task" in param_names


@pytest.mark.asyncio
async def test_cron_tool_create(cron_tool, scheduler, tool_ctx):
    result = await cron_tool.execute({
        "action": "create",
        "name": "My Job",
        "task": "Search KOSPI index",
        "schedule": "1h",
    }, tool_ctx)
    assert result.success
    assert "Job created" in result.output
    assert len(scheduler.list_jobs()) == 1


@pytest.mark.asyncio
async def test_cron_tool_create_missing_task(cron_tool, tool_ctx):
    result = await cron_tool.execute({
        "action": "create",
        "schedule": "1h",
    }, tool_ctx)
    assert not result.success
    assert "task" in result.error.lower()


@pytest.mark.asyncio
async def test_cron_tool_create_missing_schedule(cron_tool, tool_ctx):
    result = await cron_tool.execute({
        "action": "create",
        "task": "do something",
    }, tool_ctx)
    assert not result.success
    assert "schedule" in result.error.lower()


@pytest.mark.asyncio
async def test_cron_tool_list_empty(cron_tool, tool_ctx):
    result = await cron_tool.execute({"action": "list"}, tool_ctx)
    assert result.success
    assert "No scheduled jobs" in result.output


@pytest.mark.asyncio
async def test_cron_tool_list_with_jobs(cron_tool, scheduler, tool_ctx):
    scheduler.add_job(name="A", task="t1", schedule="1h")
    scheduler.add_job(name="B", task="t2", schedule="30m")
    result = await cron_tool.execute({"action": "list"}, tool_ctx)
    assert result.success
    assert "2" in result.output


@pytest.mark.asyncio
async def test_cron_tool_remove(cron_tool, scheduler, tool_ctx):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    result = await cron_tool.execute({
        "action": "remove",
        "job_id": job.job_id,
    }, tool_ctx)
    assert result.success
    assert "removed" in result.output.lower()
    assert len(scheduler.list_jobs()) == 0


@pytest.mark.asyncio
async def test_cron_tool_pause_resume(cron_tool, scheduler, tool_ctx):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")

    result = await cron_tool.execute({
        "action": "pause",
        "job_id": job.job_id,
    }, tool_ctx)
    assert result.success
    assert job.status == JobStatus.PAUSED

    result = await cron_tool.execute({
        "action": "resume",
        "job_id": job.job_id,
    }, tool_ctx)
    assert result.success
    assert job.status == JobStatus.ACTIVE


@pytest.mark.asyncio
async def test_cron_tool_history_empty(cron_tool, scheduler, tool_ctx):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    result = await cron_tool.execute({
        "action": "history",
        "job_id": job.job_id,
    }, tool_ctx)
    assert result.success
    assert "No runs" in result.output


@pytest.mark.asyncio
async def test_cron_tool_history_with_runs(cron_tool, scheduler, tool_ctx):
    job = scheduler.add_job(name="Test", task="t", schedule="1h")
    job.history.append(JobRun(
        run_id="run:001",
        started_at=datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 3, 2, 9, 1, tzinfo=timezone.utc),
        success=True,
        result_summary="KOSPI: 2850",
    ))
    result = await cron_tool.execute({
        "action": "history",
        "job_id": job.job_id,
    }, tool_ctx)
    assert result.success
    assert "KOSPI" in result.output
    assert "OK" in result.output


@pytest.mark.asyncio
async def test_cron_tool_unknown_action(cron_tool, tool_ctx):
    result = await cron_tool.execute({"action": "invalid"}, tool_ctx)
    assert not result.success
    assert "Unknown action" in result.error


@pytest.mark.asyncio
async def test_cron_tool_no_scheduler(tool_ctx):
    """When scheduler is not set, tool returns an error."""
    import mini_openclaw.tools.builtin.cron_job as mod
    original = mod._scheduler
    mod._scheduler = None
    try:
        tool = CronJobTool()
        result = await tool.execute({"action": "list"}, tool_ctx)
        assert not result.success
        assert "not available" in result.error.lower()
    finally:
        mod._scheduler = original
