"""CLI entry point for mini-openclaw."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.tree import Tree

from mini_openclaw.config import load_config
from mini_openclaw.core.events import Event, EventType
from mini_openclaw.core.gateway import Gateway

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.option("--config", "config_path", default="config.yaml", help="Config file path")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option(
    "--auto-approve", is_flag=True, default=False,
    help="Auto-approve all tool calls (skip HITL prompts)",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool, auto_approve: bool) -> None:
    """mini-openclaw: Dynamic Agent Spawning System"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    ctx.obj["auto_approve"] = auto_approve


def _setup_hitl_policy(gateway: Gateway, auto_approve: bool) -> None:
    """Configure HITL policy based on CLI flags."""
    from mini_openclaw.core.hitl import ApprovalPolicy
    if auto_approve:
        gateway.hitl.set_policy(ApprovalPolicy.AUTO_APPROVE)


@cli.command()
@click.argument("message")
@click.pass_context
def run(ctx: click.Context, message: str) -> None:
    """Run a single message through the agent (non-interactive)."""
    config = ctx.obj["config"]
    auto_approve = ctx.obj["auto_approve"]
    asyncio.run(_run_once(config, message, auto_approve))


async def _run_once(config, message: str, auto_approve: bool) -> None:
    """Execute a single message and print the result."""
    gateway = Gateway(config)
    _setup_hitl_policy(gateway, auto_approve)

    # Subscribe to events for real-time display
    async def on_event(event: Event) -> None:
        if event.type == EventType.TOOL_CALLED:
            tool = event.data.get("tool", "")
            args_preview = str(event.data.get("arguments", {}))[:80]
            console.print(f"  [dim]tool call:[/dim] [cyan]{tool}[/cyan]({args_preview})")
        elif event.type == EventType.TOOL_RESULT:
            tool = event.data.get("tool", "")
            success = event.data.get("success", False)
            icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            console.print(f"  [dim]result:[/dim] {icon} {tool}")
        elif event.type == EventType.AGENT_SPAWNED:
            parent = event.data.get("parent_id", "root")
            depth = event.data.get("depth", 0)
            tools = event.data.get("tools", [])
            console.print(
                f"  [yellow]agent spawned[/yellow] (depth={depth}, parent={parent}, "
                f"tools={tools})"
            )
        elif event.type == EventType.AGENT_COMPLETED:
            tc = event.data.get("tool_calls", 0)
            children = event.data.get("children", [])
            console.print(
                f"  [green]agent completed[/green] (tool_calls={tc}, children={len(children)})"
            )
        elif event.type == EventType.TOOL_APPROVAL_REQUESTED:
            # In 'run' mode without --auto-approve, prompt in terminal
            if not auto_approve:
                await _handle_cli_approval(gateway, event)

    gateway.event_bus.subscribe_all(on_event)

    try:
        await gateway.start()
        console.print(Panel(f"[bold]{message}[/bold]", title="User", border_style="blue"))

        session = gateway.create_session()
        result = await gateway.chat(session.session_id, message)

        console.print()
        if result.success:
            console.print(Panel(result.content, title="Assistant", border_style="green"))
        else:
            console.print(Panel(result.content, title="Error", border_style="red"))

        # Summary
        console.print(
            f"\n[dim]Tokens: {result.tokens_used} | "
            f"Tool calls: {result.tool_calls_made} | "
            f"Children spawned: {len(result.children_spawned)}[/dim]"
        )
    finally:
        await gateway.shutdown()


async def _handle_cli_approval(gateway: Gateway, event: Event) -> None:
    """Handle HITL approval in CLI mode by prompting the user."""
    request_id = event.data.get("request_id", "")
    tool_name = event.data.get("tool", "")
    description = event.data.get("description", "")
    arguments = event.data.get("arguments", {})

    console.print()
    console.print(Panel(
        f"[bold yellow]APPROVAL REQUIRED[/bold yellow]\n\n"
        f"Tool: [cyan]{tool_name}[/cyan]\n"
        f"Description: {description}\n"
        f"Arguments: {arguments}",
        title="HITL",
        border_style="yellow",
    ))

    # Run input in a thread to not block the event loop
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: console.input(
            "[yellow]Approve? (y)es / (n)o / (a)lways for this session > [/yellow]"
        ).strip().lower(),
    )

    approved = response in ("y", "yes", "a", "always")
    remember = response in ("a", "always")

    gateway.hitl.respond(
        request_id=request_id,
        approved=approved,
        remember_for_session=remember,
        reason="" if approved else "Denied by user",
    )

    if approved:
        icon = "[green]APPROVED[/green]"
        if remember:
            icon += " [dim](remembered for session)[/dim]"
    else:
        icon = "[red]DENIED[/red]"
    console.print(f"  {icon}")


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start an interactive chat session."""
    config = ctx.obj["config"]
    auto_approve = ctx.obj["auto_approve"]
    asyncio.run(_chat_loop(config, auto_approve))


async def _chat_loop(config, auto_approve: bool) -> None:
    """Interactive chat loop."""
    gateway = Gateway(config)
    _setup_hitl_policy(gateway, auto_approve)

    async def on_event(event: Event) -> None:
        if event.type == EventType.TOOL_CALLED:
            tool = event.data.get("tool", "")
            console.print(f"  [dim]calling:[/dim] [cyan]{tool}[/cyan]")
        elif event.type == EventType.AGENT_SPAWNED:
            depth = event.data.get("depth", 0)
            console.print(f"  [yellow]spawning child agent (depth={depth})[/yellow]")
        elif event.type == EventType.TOOL_APPROVAL_REQUESTED:
            if not auto_approve:
                await _handle_cli_approval(gateway, event)

    gateway.event_bus.subscribe_all(on_event)

    try:
        await gateway.start()
        session = gateway.create_session()

        console.print(
            Panel(
                f"mini-openclaw v0.1.0 | model: {config.llm.model}\n"
                f"Tools: {', '.join(gateway.tool_registry.list_names())}\n"
                f"HITL: {'auto-approve' if auto_approve else 'interactive'}\n"
                "Type 'exit' or 'quit' to end, '/tools' '/jobs' '/hitl' for info",
                title="mini-openclaw",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold blue]You>[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/quit"):
                break
            if user_input == "/tools":
                for td in gateway.tool_registry.list_definitions():
                    approval = " [yellow](requires approval)[/yellow]" if td.requires_approval else ""
                    console.print(f"  [cyan]{td.name}[/cyan]: {td.description}{approval}")
                continue
            if user_input == "/agents":
                tree_data = gateway.get_agent_tree()
                console.print(tree_data)
                continue
            if user_input == "/hitl":
                pending = gateway.hitl.get_pending()
                if pending:
                    for req in pending:
                        console.print(
                            f"  [yellow]{req.request_id}[/yellow]: "
                            f"{req.tool_name} - {req.description}"
                        )
                else:
                    console.print("  [dim]No pending approval requests[/dim]")
                continue
            if user_input == "/jobs":
                if gateway.scheduler:
                    jobs = gateway.scheduler.list_jobs()
                    if jobs:
                        console.print(f"[bold]Scheduled Jobs ({len(jobs)}):[/bold]")
                        for j in jobs:
                            status_color = {"active": "green", "paused": "yellow"}.get(j.status.value, "dim")
                            console.print(
                                f"  [{status_color}]{j.status.value}[/{status_color}] "
                                f"[cyan]{j.job_id}[/cyan]: {j.name}\n"
                                f"    Schedule: {j.schedule} | Runs: {j.run_count}"
                                + (f"/{j.max_runs}" if j.max_runs else "")
                            )
                    else:
                        console.print("  [dim]No scheduled jobs[/dim]")
                else:
                    console.print("  [dim]Scheduler not available[/dim]")
                continue

            result = await gateway.chat(session.session_id, user_input)
            console.print()
            if result.success:
                console.print(Panel(result.content, title="Assistant", border_style="green"))
            else:
                console.print(Panel(result.content, title="Error", border_style="red"))

    finally:
        await gateway.shutdown()


@cli.command()
@click.pass_context
def tui(ctx: click.Context) -> None:
    """Launch the Textual TUI interface."""
    config = ctx.obj["config"]
    auto_approve = ctx.obj["auto_approve"]
    gateway = Gateway(config)
    _setup_hitl_policy(gateway, auto_approve)

    from mini_openclaw.interfaces.cli.app import MiniOpenClawTUI
    app = MiniOpenClawTUI(gateway)
    app.run()


@cli.command()
@click.option("--host", default=None, help="Web server host")
@click.option("--port", default=None, type=int, help="Web server port")
@click.pass_context
def web(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Launch the Web UI."""
    import uvicorn
    from mini_openclaw.interfaces.web.app import create_web_app

    config = ctx.obj["config"]
    auto_approve = ctx.obj["auto_approve"]
    gateway = Gateway(config)
    _setup_hitl_policy(gateway, auto_approve)
    app = create_web_app(gateway)

    uvicorn.run(
        app,
        host=host or config.interface.web.host,
        port=port or config.interface.web.port,
        log_level="info",
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
