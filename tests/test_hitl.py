"""Tests for the HITL (Human-in-the-Loop) approval system."""

import asyncio

import pytest

from mini_openclaw.core.hitl import (
    ApprovalPolicy,
    ApprovalResponse,
    HITLManager,
    _describe_tool_call,
)


def test_auto_approve_policy():
    """AUTO_APPROVE returns immediately without blocking."""
    mgr = HITLManager(policy=ApprovalPolicy.AUTO_APPROVE)
    loop = asyncio.new_event_loop()
    resp = loop.run_until_complete(
        mgr.request_approval("a1", "s1", "shell_exec", {"command": "ls"})
    )
    loop.close()
    assert resp.approved is True


def test_auto_deny_policy():
    """AUTO_DENY returns immediately with denied."""
    mgr = HITLManager(policy=ApprovalPolicy.AUTO_DENY)
    loop = asyncio.new_event_loop()
    resp = loop.run_until_complete(
        mgr.request_approval("a1", "s1", "shell_exec", {"command": "ls"})
    )
    loop.close()
    assert resp.approved is False
    assert "Auto-denied" in resp.reason


@pytest.mark.asyncio
async def test_session_pre_approval():
    """Once a tool is approved for a session, future requests auto-approve."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)
    mgr.approve_tool_for_session("s1", "shell_exec")

    assert mgr.is_pre_approved("s1", "shell_exec") is True
    assert mgr.is_pre_approved("s1", "file_write") is False
    assert mgr.is_pre_approved("s2", "shell_exec") is False

    # Pre-approved tools return immediately
    resp = await mgr.request_approval("a1", "s1", "shell_exec", {"command": "ls"})
    assert resp.approved is True


@pytest.mark.asyncio
async def test_respond_approve():
    """User approves a pending request via respond()."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    async def approve_after_delay():
        await asyncio.sleep(0.05)
        pending = mgr.get_pending()
        assert len(pending) == 1
        req = pending[0]
        mgr.respond(request_id=req.request_id, approved=True)

    approval_task = asyncio.create_task(
        mgr.request_approval("a1", "s1", "pip_install", {"packages": ["requests"]})
    )
    await asyncio.sleep(0.01)  # Let request register
    asyncio.create_task(approve_after_delay())

    resp = await approval_task
    assert resp.approved is True
    assert len(mgr.get_pending()) == 0


@pytest.mark.asyncio
async def test_respond_deny():
    """User denies a pending request via respond()."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    async def deny_after_delay():
        await asyncio.sleep(0.05)
        pending = mgr.get_pending()
        assert len(pending) == 1
        req = pending[0]
        mgr.respond(request_id=req.request_id, approved=False, reason="too risky")

    approval_task = asyncio.create_task(
        mgr.request_approval("a1", "s1", "shell_exec", {"command": "rm -rf /"})
    )
    await asyncio.sleep(0.01)
    asyncio.create_task(deny_after_delay())

    resp = await approval_task
    assert resp.approved is False
    assert "too risky" in resp.reason


@pytest.mark.asyncio
async def test_respond_remember_for_session():
    """'Always approve' sets session-scoped pre-approval."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    async def approve_always():
        await asyncio.sleep(0.05)
        pending = mgr.get_pending()
        req = pending[0]
        mgr.respond(
            request_id=req.request_id,
            approved=True,
            remember_for_session=True,
        )

    approval_task = asyncio.create_task(
        mgr.request_approval("a1", "s1", "pip_install", {"packages": ["numpy"]})
    )
    await asyncio.sleep(0.01)
    asyncio.create_task(approve_always())

    resp = await approval_task
    assert resp.approved is True

    # Now the same tool should be pre-approved for session s1
    assert mgr.is_pre_approved("s1", "pip_install") is True
    resp2 = await mgr.request_approval("a2", "s1", "pip_install", {"packages": ["pandas"]})
    assert resp2.approved is True


@pytest.mark.asyncio
async def test_respond_invalid_request_id():
    """Responding to a non-existent request returns False."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)
    ok = mgr.respond(request_id="nonexistent", approved=True)
    assert ok is False


@pytest.mark.asyncio
async def test_cancel_all():
    """cancel_all resolves pending futures as denied."""
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)

    approval_task = asyncio.create_task(
        mgr.request_approval("a1", "s1", "shell_exec", {"command": "ls"})
    )
    await asyncio.sleep(0.01)  # Let request register

    assert len(mgr.get_pending()) == 1
    mgr.cancel_all()
    assert len(mgr.get_pending()) == 0

    resp = await approval_task
    assert resp.approved is False
    assert "shutdown" in resp.reason.lower()


def test_set_policy():
    mgr = HITLManager(policy=ApprovalPolicy.ALWAYS_ASK)
    assert mgr.policy == ApprovalPolicy.ALWAYS_ASK
    mgr.set_policy(ApprovalPolicy.AUTO_APPROVE)
    assert mgr.policy == ApprovalPolicy.AUTO_APPROVE


def test_describe_tool_call():
    """Verify human-readable descriptions are generated."""
    assert "shell command" in _describe_tool_call("shell_exec", {"command": "ls -la"}).lower()
    assert "install" in _describe_tool_call("pip_install", {"packages": ["numpy"]}).lower()
    assert "write" in _describe_tool_call("file_write", {"path": "/tmp/a.txt", "content": "hi"}).lower()
    assert "python" in _describe_tool_call("python_exec", {"code": "print(42)"}).lower()
    assert "unknown_tool" in _describe_tool_call("unknown_tool", {"arg": 1}).lower()
