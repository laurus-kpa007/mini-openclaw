"""Tests for the browser_control tool."""

import pytest
import tempfile

from mini_openclaw.tools.base import ToolContext
from mini_openclaw.tools.builtin.browser_control import BrowserControlTool, _build_script


@pytest.fixture
def sandbox_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def tool_context(sandbox_dir):
    return ToolContext(
        session_id="test-session",
        agent_id="test-agent",
        sandbox_root=sandbox_dir,
        working_directory=sandbox_dir,
    )


def test_definition():
    tool = BrowserControlTool()
    defn = tool.definition
    assert defn.name == "browser_control"
    assert defn.requires_approval is True
    assert defn.category == "browser"
    param_names = [p.name for p in defn.parameters]
    assert "action" in param_names
    assert "url" in param_names
    assert "selector" in param_names
    assert "text" in param_names


def test_action_enum():
    tool = BrowserControlTool()
    action_param = next(p for p in tool.definition.parameters if p.name == "action")
    expected_actions = [
        "goto", "click", "type", "screenshot", "get_text",
        "get_html", "evaluate", "wait_for", "select", "scroll",
    ]
    for action in expected_actions:
        assert action in action_param.enum


@pytest.mark.asyncio
async def test_no_action(tool_context):
    tool = BrowserControlTool()
    result = await tool.execute({"action": ""}, tool_context)
    assert not result.success
    assert "no action" in result.error.lower()


@pytest.mark.asyncio
async def test_unknown_action(tool_context):
    tool = BrowserControlTool()
    result = await tool.execute({"action": "fly"}, tool_context)
    assert not result.success
    assert "unknown action" in result.error.lower()


def test_build_script_goto():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script("goto", {"url": "https://example.com", "timeout": 5000}, ctx)
    assert script is not None
    assert "page.goto" in script
    assert "example.com" in script
    assert "sync_playwright" in script


def test_build_script_click():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script("click", {"selector": "#btn", "timeout": 5000}, ctx)
    assert script is not None
    assert "page.click" in script
    assert "#btn" in script


def test_build_script_type():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script("type", {"selector": "#input", "text": "hello", "timeout": 5000}, ctx)
    assert script is not None
    assert "page.fill" in script
    assert "hello" in script


def test_build_script_screenshot():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script(
        "screenshot",
        {"url": "https://example.com", "save_path": "out.png", "timeout": 5000},
        ctx,
    )
    assert script is not None
    assert "page.screenshot" in script
    assert "out.png" in script


def test_build_script_evaluate():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script(
        "evaluate",
        {"url": "https://example.com", "value": "document.title", "timeout": 5000},
        ctx,
    )
    assert script is not None
    assert "page.evaluate" in script
    assert "document.title" in script


def test_build_script_scroll():
    ctx = ToolContext(session_id="s", agent_id="a")
    script = _build_script("scroll", {"url": "https://example.com", "value": "down", "timeout": 5000}, ctx)
    assert script is not None
    assert "scrollBy" in script


def test_build_script_missing_params():
    """Missing required params return None."""
    ctx = ToolContext(session_id="s", agent_id="a")
    # goto without url
    assert _build_script("goto", {"timeout": 5000}, ctx) is None
    # click without selector
    assert _build_script("click", {"timeout": 5000}, ctx) is None
    # type without text
    assert _build_script("type", {"selector": "#x", "timeout": 5000}, ctx) is None
    # evaluate without value
    assert _build_script("evaluate", {"timeout": 5000}, ctx) is None
    # select without value
    assert _build_script("select", {"selector": "#x", "timeout": 5000}, ctx) is None


@pytest.mark.asyncio
async def test_playwright_not_installed_error(tool_context):
    """When playwright is not installed, return a helpful error message."""
    tool = BrowserControlTool()
    result = await tool.execute(
        {"action": "goto", "url": "https://example.com"},
        tool_context,
    )
    # If playwright IS installed, this will succeed; if not, it should give a clear error.
    # We test both outcomes are handled gracefully.
    if not result.success:
        assert "playwright" in result.error.lower() or "error" in result.error.lower()
    else:
        assert "example.com" in result.output.lower() or "navigated" in result.output.lower()
