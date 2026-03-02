"""REST API routes for mini-openclaw web interface."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


class MessageRequest(BaseModel):
    content: str


class SessionResponse(BaseModel):
    session_id: str
    model: str
    created_at: str


class ChatResponse(BaseModel):
    success: bool
    content: str
    tokens_used: int
    tool_calls_made: int
    children_spawned: list[str]


def create_router() -> APIRouter:
    router = APIRouter()

    def _get_gateway(request: Request):
        return request.app.state.gateway

    @router.post("/sessions", response_model=SessionResponse)
    async def create_session(request: Request):
        """Create a new chat session."""
        gw = _get_gateway(request)
        session = gw.create_session()
        return SessionResponse(
            session_id=session.session_id,
            model=session.model,
            created_at=session.created_at.isoformat(),
        )

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        """Get session details."""
        gw = _get_gateway(request)
        session = gw.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
            "message_count": len(session.conversation_history),
            "agent_ids": session.agent_ids,
        }

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        """End a session."""
        gw = _get_gateway(request)
        if session_id in gw.sessions:
            del gw.sessions[session_id]
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Session not found")

    @router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
    async def send_message(session_id: str, msg: MessageRequest, request: Request):
        """Send a message to an agent in a session."""
        gw = _get_gateway(request)
        if session_id not in gw.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        result = await gw.chat(session_id, msg.content)
        return ChatResponse(
            success=result.success,
            content=result.content,
            tokens_used=result.tokens_used,
            tool_calls_made=result.tool_calls_made,
            children_spawned=result.children_spawned,
        )

    @router.get("/sessions/{session_id}/history")
    async def get_history(session_id: str, request: Request):
        """Get conversation history for a session."""
        gw = _get_gateway(request)
        session = gw.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "agent_id": msg.agent_id,
            }
            for msg in session.conversation_history
        ]

    @router.get("/tools")
    async def list_tools(request: Request):
        """List all available tools."""
        gw = _get_gateway(request)
        return [
            {
                "name": td.name,
                "description": td.description,
                "category": td.category,
                "requires_approval": td.requires_approval,
                "parameters": [p.model_dump() for p in td.parameters],
            }
            for td in gw.tool_registry.list_definitions()
        ]

    @router.get("/agents")
    async def list_agents(request: Request):
        """List all active agents."""
        gw = _get_gateway(request)
        return gw.get_agent_tree()

    @router.get("/config")
    async def get_config(request: Request):
        """Get current configuration (safe subset)."""
        gw = _get_gateway(request)
        return {
            "llm": {
                "model": gw.config.llm.model,
                "provider": gw.config.llm.provider,
            },
            "gateway": gw.config.gateway.model_dump(),
            "tools": gw.tool_registry.list_names(),
        }

    return router
