"""Human-in-the-Loop (HITL) approval system.

When a tool with requires_approval=True is called, the Agent pauses execution
and emits a TOOL_APPROVAL_REQUESTED event. The UI (CLI/Web) presents the request
to the user. The user approves or denies, and the response flows back through
an asyncio.Future to unblock the Agent.

Flow:
  Agent._execute_tool()
    -> tool.requires_approval? -> gateway.hitl.request_approval(...)
       -> emit TOOL_APPROVAL_REQUESTED event (UI picks this up)
       -> await future (Agent blocks here)
          <- UI calls gateway.hitl.respond(request_id, approved, ...)
             <- future resolved -> Agent continues or skips
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalPolicy(Enum):
    """Determines how approval requests are handled."""
    ALWAYS_ASK = "always_ask"       # Always ask user (default)
    AUTO_APPROVE = "auto_approve"   # Approve everything automatically
    AUTO_DENY = "auto_deny"         # Deny everything automatically


@dataclass
class ApprovalRequest:
    """A pending approval request from an agent."""
    request_id: str
    agent_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str  # Human-readable description of what will happen
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    future: asyncio.Future | None = field(default=None, repr=False)


@dataclass
class ApprovalResponse:
    """User's response to an approval request."""
    request_id: str
    approved: bool
    modified_arguments: dict[str, Any] | None = None  # User can modify args
    reason: str = ""  # Optional reason for denial


class HITLManager:
    """
    Manages the approval lifecycle for tool calls that require user confirmation.

    The Gateway owns this manager. Agents call request_approval() which blocks
    until the UI calls respond(). The UI subscribes to TOOL_APPROVAL_REQUESTED
    events to know when to prompt the user.
    """

    def __init__(self, policy: ApprovalPolicy = ApprovalPolicy.ALWAYS_ASK) -> None:
        self.policy = policy
        self._pending: dict[str, ApprovalRequest] = {}
        # Tools that the user has approved for this session (session-scoped memory)
        self._session_approved: dict[str, set[str]] = {}  # session_id -> {tool_names}

    def set_policy(self, policy: ApprovalPolicy) -> None:
        self.policy = policy

    def approve_tool_for_session(self, session_id: str, tool_name: str) -> None:
        """Remember that the user approved this tool for the rest of the session."""
        if session_id not in self._session_approved:
            self._session_approved[session_id] = set()
        self._session_approved[session_id].add(tool_name)

    def is_pre_approved(self, session_id: str, tool_name: str) -> bool:
        """Check if a tool was previously approved for this session."""
        return tool_name in self._session_approved.get(session_id, set())

    async def request_approval(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        description: str = "",
    ) -> ApprovalResponse:
        """
        Request approval for a tool call. Blocks until the user responds.

        Returns ApprovalResponse with approved=True/False and optionally modified args.
        """
        # Auto-approve/deny based on policy
        if self.policy == ApprovalPolicy.AUTO_APPROVE:
            return ApprovalResponse(request_id="", approved=True)
        if self.policy == ApprovalPolicy.AUTO_DENY:
            return ApprovalResponse(request_id="", approved=False, reason="Auto-denied by policy")

        # Check session-scoped pre-approval
        if self.is_pre_approved(session_id, tool_name):
            return ApprovalResponse(request_id="", approved=True)

        # Create the request with a Future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = f"approval:{uuid.uuid4().hex[:8]}"

        request = ApprovalRequest(
            request_id=request_id,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            description=description or _describe_tool_call(tool_name, arguments),
            future=future,
        )
        self._pending[request_id] = request

        logger.info(
            "HITL approval requested: %s tool=%s agent=%s",
            request_id, tool_name, agent_id,
        )

        # The caller (UI) will be notified via EventBus - the event is emitted
        # by the Agent, not here, so we just await the future
        try:
            response = await asyncio.wait_for(future, timeout=300.0)  # 5min timeout
            return response
        except asyncio.TimeoutError:
            logger.warning("Approval request %s timed out", request_id)
            return ApprovalResponse(
                request_id=request_id,
                approved=False,
                reason="Approval timed out (5 minutes)",
            )
        finally:
            self._pending.pop(request_id, None)

    def respond(
        self,
        request_id: str,
        approved: bool,
        modified_arguments: dict[str, Any] | None = None,
        reason: str = "",
        remember_for_session: bool = False,
    ) -> bool:
        """
        Respond to a pending approval request. Called by the UI.
        Returns True if the request was found and resolved.
        """
        request = self._pending.get(request_id)
        if not request or not request.future:
            logger.warning("Approval request %s not found or already resolved", request_id)
            return False

        response = ApprovalResponse(
            request_id=request_id,
            approved=approved,
            modified_arguments=modified_arguments,
            reason=reason,
        )

        # Remember approval for session if requested
        if approved and remember_for_session:
            self.approve_tool_for_session(request.session_id, request.tool_name)

        request.future.set_result(response)
        logger.info(
            "HITL approval response: %s approved=%s remember=%s",
            request_id, approved, remember_for_session,
        )
        return True

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all pending approval requests (for UI display)."""
        return list(self._pending.values())

    def cancel_all(self) -> None:
        """Cancel all pending requests (e.g., on shutdown)."""
        for request in self._pending.values():
            if request.future and not request.future.done():
                request.future.set_result(
                    ApprovalResponse(
                        request_id=request.request_id,
                        approved=False,
                        reason="System shutdown",
                    )
                )
        self._pending.clear()


def _describe_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Generate a human-readable description of a tool call."""
    if tool_name == "shell_exec":
        cmd = arguments.get("command", "?")
        return f"Execute shell command: {cmd}"
    elif tool_name == "file_write":
        path = arguments.get("path", "?")
        mode = arguments.get("mode", "write")
        size = len(arguments.get("content", ""))
        return f"{'Append to' if mode == 'append' else 'Write to'} file: {path} ({size} chars)"
    elif tool_name == "pip_install":
        packages = arguments.get("packages", [])
        return f"Install Python packages: {', '.join(packages) if isinstance(packages, list) else packages}"
    elif tool_name == "python_exec":
        code = arguments.get("code", "?")
        preview = code[:100] + "..." if len(code) > 100 else code
        return f"Execute Python code: {preview}"
    else:
        args_str = ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:3])
        return f"Call {tool_name}({args_str})"
