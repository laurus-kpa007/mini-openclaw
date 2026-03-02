"""Built-in web search tool using DuckDuckGo Lite."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)

DDG_URL = "https://lite.duckduckgo.com/lite/"


class WebSearchTool:
    """Search the web using DuckDuckGo Lite (no API key required)."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for information using DuckDuckGo. Returns titles, URLs, and snippets.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return (default 5)",
                    required=False,
                    default=5,
                ),
            ],
            category="web",
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query", "")
        max_results = int(arguments.get("max_results", 5))

        if not query:
            return ToolResult(success=False, output="", error="No search query provided")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    DDG_URL,
                    data={"q": query, "kl": ""},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; mini-openclaw/0.1)"},
                )

            if resp.status_code != 200:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Search returned status {resp.status_code}",
                )

            results = self._parse_results(resp.text, max_results)
            if not results:
                return ToolResult(success=True, output=f"No results found for: {query}")

            output_lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                output_lines.append(f"{i}. {r['title']}")
                output_lines.append(f"   URL: {r['url']}")
                if r.get("snippet"):
                    output_lines.append(f"   {r['snippet']}")
                output_lines.append("")

            return ToolResult(success=True, output="\n".join(output_lines))

        except httpx.TimeoutException:
            return ToolResult(success=False, output="", error="Search request timed out")
        except Exception as e:
            logger.exception("Web search failed")
            return ToolResult(success=False, output="", error=f"Search error: {e}")

    @staticmethod
    def _parse_results(html: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo Lite HTML for results."""
        results = []
        # Extract links: DuckDuckGo Lite uses <a> tags with class "result-link"
        # Fallback: extract all links that look like results
        link_pattern = re.compile(
            r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result-link[^"]*"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        matches = link_pattern.findall(html)

        if not matches:
            # Fallback: extract links from the results table
            link_pattern2 = re.compile(
                r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            matches = link_pattern2.findall(html)

        # Extract snippets
        snippet_pattern = re.compile(
            r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>',
            re.IGNORECASE | re.DOTALL,
        )
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(matches[:max_results]):
            # Clean HTML tags
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

            if url and title_clean and not url.startswith("javascript:"):
                results.append({
                    "title": title_clean,
                    "url": url,
                    "snippet": snippet,
                })

        return results
