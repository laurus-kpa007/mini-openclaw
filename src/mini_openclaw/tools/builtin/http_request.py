"""Built-in HTTP request tool with host restrictions."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class HttpRequestTool:
    """Make HTTP requests with optional host allowlist."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="http_request",
            description="Make an HTTP request to a URL and return the response body.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to request",
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method (GET, POST, PUT, DELETE)",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "DELETE"],
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="Optional HTTP headers as key-value pairs",
                    required=False,
                ),
                ToolParameter(
                    name="body",
                    type="string",
                    description="Optional request body (for POST/PUT)",
                    required=False,
                ),
            ],
            category="web",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = arguments.get("url", "")
        method = arguments.get("method", "GET").upper()
        headers = arguments.get("headers") or {}
        body = arguments.get("body")

        if not url:
            return ToolResult(success=False, output="", error="No URL provided")

        # Host allowlist check
        if context.allowed_hosts:
            parsed = urlparse(url)
            if parsed.hostname not in context.allowed_hosts:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Host '{parsed.hostname}' is not in the allowed hosts list",
                )

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    content=body if body else None,
                )

            # Truncate large responses
            body_text = resp.text
            if len(body_text) > 10000:
                body_text = body_text[:10000] + f"\n... [truncated, total {len(resp.text)} chars]"

            output = (
                f"HTTP {resp.status_code} {resp.reason_phrase}\n"
                f"Content-Type: {resp.headers.get('content-type', 'unknown')}\n"
                f"Content-Length: {len(resp.text)}\n\n"
                f"{body_text}"
            )

            return ToolResult(
                success=200 <= resp.status_code < 400,
                output=output,
                error=None if 200 <= resp.status_code < 400 else f"HTTP {resp.status_code}",
            )

        except httpx.TimeoutException:
            return ToolResult(success=False, output="", error="Request timed out (30s)")
        except Exception as e:
            logger.exception("HTTP request failed")
            return ToolResult(success=False, output="", error=f"Request error: {e}")
