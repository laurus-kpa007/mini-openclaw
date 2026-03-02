"""Tests for pip_install and python_exec tools."""

import pytest
import tempfile

from mini_openclaw.tools.base import ToolContext
from mini_openclaw.tools.builtin.pip_install import PipInstallTool
from mini_openclaw.tools.builtin.python_exec import PythonExecTool


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


# --- PipInstallTool ---

def test_pip_install_definition():
    tool = PipInstallTool()
    defn = tool.definition
    assert defn.name == "pip_install"
    assert defn.requires_approval is True
    assert defn.category == "system"
    param_names = [p.name for p in defn.parameters]
    assert "packages" in param_names
    assert "upgrade" in param_names


@pytest.mark.asyncio
async def test_pip_install_no_packages(tool_context):
    tool = PipInstallTool()
    result = await tool.execute({"packages": []}, tool_context)
    assert not result.success
    assert "no packages" in result.error.lower()


@pytest.mark.asyncio
async def test_pip_install_string_package(tool_context):
    """String input for packages is coerced to a list."""
    tool = PipInstallTool()
    # Install a lightweight built-in that's always available
    result = await tool.execute({"packages": "pip"}, tool_context)
    assert result.success


@pytest.mark.asyncio
async def test_pip_install_already_satisfied(tool_context):
    """Installing an already-installed package should succeed."""
    tool = PipInstallTool()
    result = await tool.execute({"packages": ["pip"]}, tool_context)
    assert result.success
    assert "satisfied" in result.output.lower() or "installed" in result.output.lower()


# --- PythonExecTool ---

def test_python_exec_definition():
    tool = PythonExecTool()
    defn = tool.definition
    assert defn.name == "python_exec"
    assert defn.requires_approval is True
    assert defn.category == "system"
    param_names = [p.name for p in defn.parameters]
    assert "code" in param_names
    assert "timeout" in param_names


@pytest.mark.asyncio
async def test_python_exec_simple(tool_context):
    tool = PythonExecTool()
    result = await tool.execute({"code": "print('hello from python')"}, tool_context)
    assert result.success
    assert "hello from python" in result.output


@pytest.mark.asyncio
async def test_python_exec_empty_code(tool_context):
    tool = PythonExecTool()
    result = await tool.execute({"code": ""}, tool_context)
    assert not result.success
    assert "no code" in result.error.lower()


@pytest.mark.asyncio
async def test_python_exec_error(tool_context):
    tool = PythonExecTool()
    result = await tool.execute({"code": "raise ValueError('test error')"}, tool_context)
    assert not result.success
    assert "test error" in result.output or "test error" in (result.error or "")


@pytest.mark.asyncio
async def test_python_exec_timeout(tool_context):
    """Code exceeding the timeout is killed."""
    tool = PythonExecTool()
    result = await tool.execute(
        {"code": "import time; time.sleep(100)", "timeout": 1},
        tool_context,
    )
    assert not result.success
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_python_exec_multiline(tool_context):
    """Multi-line code works correctly."""
    tool = PythonExecTool()
    code = "x = 10\ny = 20\nprint(x + y)"
    result = await tool.execute({"code": code}, tool_context)
    assert result.success
    assert "30" in result.output
