"""FastAPI application factory for mini-openclaw web UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mini_openclaw.interfaces.web.routes import create_router
from mini_openclaw.interfaces.web.websocket import create_ws_router

if TYPE_CHECKING:
    from mini_openclaw.core.gateway import Gateway

STATIC_DIR = Path(__file__).parent / "static"


def create_web_app(gateway: Gateway) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="mini-openclaw",
        version="0.1.0",
        description="Dynamic Agent Spawning System - Web Interface",
    )

    # Store gateway in app state
    app.state.gateway = gateway

    # Include routers
    app.include_router(create_router(), prefix="/api")
    app.include_router(create_ws_router())

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def startup():
        await gateway.start()

    @app.on_event("shutdown")
    async def shutdown():
        await gateway.shutdown()

    @app.get("/")
    async def index():
        """Serve the main page."""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            from fastapi.responses import FileResponse
            return FileResponse(str(index_file))
        return {"message": "mini-openclaw API is running. Visit /docs for API docs."}

    return app
