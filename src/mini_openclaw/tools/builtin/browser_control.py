"""Built-in browser control tool using Playwright."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from mini_openclaw.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class BrowserControlTool:
    """Control a headless browser to navigate, interact, and extract data from web pages."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser_control",
            description=(
                "Control a headless web browser (Playwright/Chromium). "
                "Supports navigation, clicking, typing, screenshots, and text extraction. "
                "Requires 'playwright' package — use pip_install first if not installed."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Browser action to perform",
                    enum=[
                        "goto",
                        "click",
                        "type",
                        "screenshot",
                        "get_text",
                        "get_html",
                        "evaluate",
                        "wait_for",
                        "select",
                        "scroll",
                    ],
                ),
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL to navigate to (for 'goto' action)",
                    required=False,
                ),
                ToolParameter(
                    name="selector",
                    type="string",
                    description="CSS selector for the target element (for click/type/get_text/wait_for/select)",
                    required=False,
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description="Text to type (for 'type' action)",
                    required=False,
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="Value for select dropdown, or JS expression for 'evaluate'",
                    required=False,
                ),
                ToolParameter(
                    name="save_path",
                    type="string",
                    description="File path to save screenshot (for 'screenshot', default: screenshot.png)",
                    required=False,
                    default="screenshot.png",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in milliseconds (default: 10000)",
                    required=False,
                    default=10000,
                ),
            ],
            category="browser",
            requires_approval=True,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "")
        timeout_ms = int(arguments.get("timeout", 10000))

        if not action:
            return ToolResult(success=False, output="", error="No action specified")

        # Build the Playwright script based on action
        script = _build_script(action, arguments, context)
        if script is None:
            return ToolResult(
                success=False, output="",
                error=f"Unknown action: {action}",
            )

        # Execute script in a subprocess (isolates Playwright from our event loop)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(script)
                temp_path = f.name

            cwd = context.sandbox_root or context.working_directory

            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            timeout_s = max(timeout_ms / 1000 + 10, 30)  # script timeout + buffer
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False, output="",
                    error=f"Browser action timed out after {timeout_s:.0f}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            if proc.returncode == 0:
                # Truncate large output
                output = stdout_str
                if len(output) > 8000:
                    output = output[:8000] + f"\n... [truncated, total {len(stdout_str)} chars]"
                return ToolResult(success=True, output=output)
            else:
                # Check if playwright is not installed
                if "No module named 'playwright'" in stderr_str:
                    return ToolResult(
                        success=False, output="",
                        error=(
                            "Playwright is not installed. "
                            "Use pip_install(['playwright']) first, "
                            "then shell_exec('playwright install chromium')."
                        ),
                    )
                return ToolResult(
                    success=False,
                    output=stdout_str[:1000] if stdout_str else "",
                    error=stderr_str[:1000],
                )

        except Exception as e:
            logger.exception("Browser control failed")
            return ToolResult(success=False, output="", error=f"Browser error: {e}")
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _build_script(action: str, args: dict[str, Any], context: ToolContext) -> str | None:
    """Build a self-contained Playwright Python script for the given action."""
    url = args.get("url", "")
    selector = args.get("selector", "")
    text = args.get("text", "")
    value = args.get("value", "")
    save_path = args.get("save_path", "screenshot.png")
    timeout_ms = int(args.get("timeout", 10000))

    # Common preamble: launch browser, open page
    preamble = f"""
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={{"width": 1280, "height": 720}},
        user_agent="mini-openclaw/0.1 browser_control",
    )
    page = ctx.new_page()
    page.set_default_timeout({timeout_ms})
"""

    close = """
    browser.close()
"""

    if action == "goto":
        if not url:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded")
    title = page.title()
    url_after = page.url
    text_content = page.inner_text("body")
    if len(text_content) > 3000:
        text_content = text_content[:3000] + "..."
    print(f"Navigated to: {{url_after}}")
    print(f"Title: {{title}}")
    print(f"---PAGE TEXT---")
    print(text_content)
""" + close

    elif action == "click":
        if not selector:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    page.click({selector!r})
    page.wait_for_load_state("domcontentloaded")
    print(f"Clicked: {selector!r}")
    print(f"Current URL: {{page.url}}")
    print(f"Title: {{page.title()}}")
""" + close

    elif action == "type":
        if not selector or not text:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    page.fill({selector!r}, {text!r})
    print(f"Typed into {selector!r}: {text!r}")
""" + close

    elif action == "screenshot":
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    page.screenshot(path={save_path!r}, full_page=True)
    print(f"Screenshot saved to: {save_path!r}")
    print(f"Page title: {{page.title()}}")
    print(f"URL: {{page.url}}")
""" + close

    elif action == "get_text":
        if not selector and not url:
            return None
        sel = selector or "body"
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    text = page.inner_text({sel!r})
    if len(text) > 5000:
        text = text[:5000] + "..."
    print(text)
""" + close

    elif action == "get_html":
        if not url and not selector:
            return None
        sel = selector or "body"
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    html = page.inner_html({sel!r})
    if len(html) > 5000:
        html = html[:5000] + "..."
    print(html)
""" + close

    elif action == "evaluate":
        if not value:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    result = page.evaluate({value!r})
    print(json.dumps(result, ensure_ascii=False, default=str))
""" + close

    elif action == "wait_for":
        if not selector:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    page.wait_for_selector({selector!r}, state="visible")
    print(f"Element found: {selector!r}")
    print(f"Text: {{page.inner_text({selector!r})[:500]}}")
""" + close

    elif action == "select":
        if not selector or not value:
            return None
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    page.select_option({selector!r}, {value!r})
    print(f"Selected {value!r} in {selector!r}")
""" + close

    elif action == "scroll":
        direction = value or "down"
        return preamble + f"""
    page.goto({url!r}, wait_until="domcontentloaded") if {bool(url)} else None
    if {direction!r} == "down":
        page.evaluate("window.scrollBy(0, 800)")
    elif {direction!r} == "up":
        page.evaluate("window.scrollBy(0, -800)")
    elif {direction!r} == "bottom":
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    elif {direction!r} == "top":
        page.evaluate("window.scrollTo(0, 0)")
    print(f"Scrolled {direction!r}")
    print(f"Page height: {{page.evaluate('document.body.scrollHeight')}}")
""" + close

    return None
