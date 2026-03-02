"""WebSocket handlers for real-time agent interaction."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mini_openclaw.core.events import Event, EventType

logger = logging.getLogger(__name__)


def create_ws_router() -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/sessions/{session_id}")
    async def session_websocket(websocket: WebSocket, session_id: str):
        """
        Bidirectional WebSocket for real-time interaction.

        Client -> Server:
            {"type": "message", "content": "user text"}
            {"type": "command", "command": "/tools"}
            {"type": "approval_response", "request_id": "...", "approved": bool, "remember": bool}

        Server -> Client:
            {"type": "assistant", "content": "...", "agent_id": "..."}
            {"type": "tool_call", "tool": "...", "args": {...}, "agent_id": "..."}
            {"type": "tool_result", "tool": "...", "success": bool, "agent_id": "..."}
            {"type": "agent_spawned", "agent_id": "...", "parent_id": "...", "depth": int}
            {"type": "agent_completed", "agent_id": "...", "success": bool}
            {"type": "approval_request", "request_id": "...", "tool": "...", "description": "...", "arguments": {...}}
            {"type": "error", "message": "..."}
        """
        await websocket.accept()
        gateway = websocket.app.state.gateway

        # Ensure session exists
        if session_id not in gateway.sessions:
            session = gateway.create_session()
            session_id = session.session_id
            await websocket.send_json({"type": "session_created", "session_id": session_id})

        # Event callback to forward events to WebSocket
        async def forward_event(event: Event):
            try:
                msg = _event_to_ws_message(event)
                if msg:
                    await websocket.send_json(msg)
            except Exception:
                pass  # Connection may have closed

        gateway.event_bus.subscribe_all(forward_event)

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "message":
                    content = data.get("content", "")
                    if content:
                        try:
                            result = await gateway.chat(session_id, content)
                            await websocket.send_json({
                                "type": "assistant",
                                "content": result.content,
                                "success": result.success,
                                "tokens_used": result.tokens_used,
                                "tool_calls_made": result.tool_calls_made,
                                "children_spawned": result.children_spawned,
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e),
                            })

                elif msg_type == "approval_response":
                    # User responded to an HITL approval request
                    request_id = data.get("request_id", "")
                    approved = data.get("approved", False)
                    remember = data.get("remember", False)
                    reason = data.get("reason", "") if not approved else ""
                    if request_id and gateway.hitl:
                        ok = gateway.hitl.respond(
                            request_id=request_id,
                            approved=approved,
                            remember_for_session=remember,
                            reason=reason,
                        )
                        if not ok:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Approval request '{request_id}' not found or already resolved.",
                            })

                elif msg_type == "command":
                    cmd = data.get("command", "")
                    if cmd == "/tools":
                        tools = [
                            {
                                "name": td.name,
                                "description": td.description,
                                "requires_approval": td.requires_approval,
                            }
                            for td in gateway.tool_registry.list_definitions()
                        ]
                        await websocket.send_json({"type": "tools_list", "tools": tools})
                    elif cmd == "/agents":
                        tree = gateway.get_agent_tree()
                        await websocket.send_json({"type": "agents_tree", "tree": tree})
                    elif cmd == "/hitl":
                        pending = []
                        if gateway.hitl:
                            for req in gateway.hitl.get_pending():
                                pending.append({
                                    "request_id": req.request_id,
                                    "tool_name": req.tool_name,
                                    "description": req.description,
                                    "arguments": req.arguments,
                                })
                        await websocket.send_json({"type": "hitl_pending", "requests": pending})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session %s", session_id)
        finally:
            gateway.event_bus.unsubscribe_all(forward_event)

    return router


def _event_to_ws_message(event: Event) -> dict | None:
    """Convert a Gateway event to a WebSocket message."""
    if event.type == EventType.TOOL_CALLED:
        return {
            "type": "tool_call",
            "tool": event.data.get("tool", ""),
            "args": event.data.get("arguments", {}),
            "agent_id": event.source_id,
        }
    elif event.type == EventType.TOOL_RESULT:
        return {
            "type": "tool_result",
            "tool": event.data.get("tool", ""),
            "success": event.data.get("success", False),
            "preview": event.data.get("output_preview", ""),
            "agent_id": event.source_id,
        }
    elif event.type == EventType.AGENT_SPAWNED:
        return {
            "type": "agent_spawned",
            "agent_id": event.source_id,
            "parent_id": event.data.get("parent_id"),
            "depth": event.data.get("depth", 0),
            "tools": event.data.get("tools", []),
        }
    elif event.type == EventType.AGENT_COMPLETED:
        return {
            "type": "agent_completed",
            "agent_id": event.source_id,
            "success": event.data.get("success", True),
        }
    elif event.type == EventType.AGENT_FAILED:
        return {
            "type": "agent_failed",
            "agent_id": event.source_id,
            "error": event.data.get("error", ""),
        }
    elif event.type == EventType.TOOL_APPROVAL_REQUESTED:
        args = event.data.get("arguments", {})
        args_str = str(args)
        if len(args_str) > 300:
            args_str = args_str[:300] + "..."
        return {
            "type": "approval_request",
            "request_id": event.data.get("request_id", ""),
            "tool": event.data.get("tool", ""),
            "description": event.data.get("description", ""),
            "arguments": args,
            "agent_id": event.source_id,
        }
    elif event.type == EventType.TOOL_APPROVAL_RESPONSE:
        return {
            "type": "approval_resolved",
            "request_id": event.data.get("request_id", ""),
            "approved": event.data.get("approved", False),
            "tool": event.data.get("tool", ""),
        }
    return None
