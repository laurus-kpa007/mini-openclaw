"""Textual TUI application for mini-openclaw with HITL support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static, Tree

from mini_openclaw.core.events import Event, EventType

if TYPE_CHECKING:
    from mini_openclaw.core.gateway import Gateway


class AgentTreeWidget(Tree):
    """Displays the live agent hierarchy."""

    def __init__(self, *args, **kwargs):
        super().__init__("Agents", *args, **kwargs)
        self._agent_nodes: dict[str, any] = {}

    def add_agent(self, agent_id: str, parent_id: str | None, depth: int, tools: list[str]):
        label = f"[cyan]Agent[/] d={depth} [{', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}]"
        if parent_id and parent_id in self._agent_nodes:
            node = self._agent_nodes[parent_id].add(label, data=agent_id)
        else:
            node = self.root.add(label, data=agent_id)
        self._agent_nodes[agent_id] = node
        node.expand()
        self.root.expand()

    def update_agent_state(self, agent_id: str, new_state: str):
        if agent_id in self._agent_nodes:
            node = self._agent_nodes[agent_id]
            state_colors = {
                "running": "green",
                "waiting_for_tool": "yellow",
                "waiting_for_child": "blue",
                "completed": "dim green",
                "failed": "red",
                "terminated": "dim red",
            }
            color = state_colors.get(new_state, "white")
            node.label = f"[{color}]{new_state}[/] {node.label}"


class ApprovalScreen(ModalScreen[tuple[bool, bool]]):
    """Modal dialog for HITL approval requests."""

    CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 70;
        height: auto;
        max-height: 20;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #approval-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    #approval-desc {
        margin-bottom: 1;
    }
    #approval-buttons {
        margin-top: 1;
        height: 3;
    }
    #approval-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        tool_name: str,
        description: str,
        arguments: dict,
    ) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.description = description
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("APPROVAL REQUIRED", id="approval-title")
            yield Label(f"Tool: [cyan]{self.tool_name}[/cyan]")
            yield Label(f"{self.description}", id="approval-desc")
            args_str = str(self.arguments)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            yield Label(f"[dim]Args: {args_str}[/dim]")
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", variant="success", id="btn-approve")
                yield Button("Always (session)", variant="warning", id="btn-always")
                yield Button("Deny", variant="error", id="btn-deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-approve":
            self.dismiss((True, False))
        elif event.button.id == "btn-always":
            self.dismiss((True, True))
        elif event.button.id == "btn-deny":
            self.dismiss((False, False))


class MiniOpenClawTUI(App):
    """Textual TUI for interactive agent chat with HITL approval dialogs."""

    CSS = """
    #main-area { width: 3fr; }
    #sidebar { width: 1fr; min-width: 30; border-left: solid $accent; }
    #chat-log { height: 1fr; }
    #tool-log { height: 12; border-top: solid $accent; }
    #agent-tree { height: 1fr; }
    #input-box { dock: bottom; }
    #sidebar-label { text-style: bold; padding: 0 1; background: $accent; }
    #tool-label { text-style: bold; padding: 0 1; background: $accent; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    TITLE = "mini-openclaw"

    def __init__(self, gateway: Gateway) -> None:
        super().__init__()
        self.gateway = gateway
        self._session_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="main-area"):
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
                yield Input(
                    placeholder="Type a message... (/help, /tools, /quit)",
                    id="input-box",
                )
            with Vertical(id="sidebar"):
                yield Static("Agent Tree", id="sidebar-label")
                yield AgentTreeWidget(id="agent-tree")
                yield Static("Tool Activity", id="tool-label")
                yield RichLog(id="tool-log", wrap=True, highlight=True, markup=True, max_lines=100)
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize gateway and subscribe to events."""
        await self.gateway.start()
        session = self.gateway.create_session()
        self._session_id = session.session_id

        # Subscribe to events
        self.gateway.event_bus.subscribe_all(self._on_gateway_event)

        chat_log = self.query_one("#chat-log", RichLog)
        tools = ", ".join(self.gateway.tool_registry.list_names())
        hitl_status = "auto-approve" if self.gateway.hitl.policy.value == "auto_approve" else "interactive"
        chat_log.write(
            f"[bold cyan]mini-openclaw v0.1.0[/]\n"
            f"Model: {self.gateway.config.llm.model}\n"
            f"Tools: {tools}\n"
            f"HITL: {hitl_status}\n"
            f"Type /help for commands\n"
        )

    async def _on_gateway_event(self, event: Event) -> None:
        """Handle Gateway events for UI updates."""
        tool_log = self.query_one("#tool-log", RichLog)
        agent_tree = self.query_one("#agent-tree", AgentTreeWidget)

        if event.type == EventType.AGENT_SPAWNED:
            agent_tree.add_agent(
                agent_id=event.source_id,
                parent_id=event.data.get("parent_id"),
                depth=event.data.get("depth", 0),
                tools=event.data.get("tools", []),
            )
        elif event.type == EventType.AGENT_STATE_CHANGED:
            agent_tree.update_agent_state(
                event.source_id, event.data.get("new_state", "")
            )
        elif event.type == EventType.TOOL_CALLED:
            tool = event.data.get("tool", "?")
            args = str(event.data.get("arguments", {}))[:60]
            tool_log.write(f"[cyan]{tool}[/]({args})")
        elif event.type == EventType.TOOL_RESULT:
            tool = event.data.get("tool", "?")
            ok = event.data.get("success", False)
            icon = "[green]OK[/]" if ok else "[red]FAIL[/]"
            tool_log.write(f"  -> {icon} {tool}")
        elif event.type == EventType.TOOL_APPROVAL_REQUESTED:
            await self._handle_approval_request(event)

    async def _handle_approval_request(self, event: Event) -> None:
        """Show HITL approval modal dialog."""
        request_id = event.data.get("request_id", "")
        tool_name = event.data.get("tool", "")
        description = event.data.get("description", "")
        arguments = event.data.get("arguments", {})

        tool_log = self.query_one("#tool-log", RichLog)
        tool_log.write(f"[yellow]APPROVAL NEEDED:[/] {tool_name}")

        def on_dismiss(result: tuple[bool, bool]) -> None:
            approved, remember = result
            self.gateway.hitl.respond(
                request_id=request_id,
                approved=approved,
                remember_for_session=remember,
                reason="" if approved else "Denied by user",
            )
            icon = "[green]APPROVED[/]" if approved else "[red]DENIED[/]"
            if remember:
                icon += " [dim](always)[/]"
            tool_log.write(f"  -> {icon} {tool_name}")

        screen = ApprovalScreen(
            tool_name=tool_name,
            description=description,
            arguments=arguments,
        )
        self.push_screen(screen, on_dismiss)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        text = event.value.strip()
        event.input.value = ""

        if not text:
            return

        chat_log = self.query_one("#chat-log", RichLog)

        # Slash commands
        if text.startswith("/"):
            await self._handle_command(text, chat_log)
            return

        # Regular message
        chat_log.write(f"\n[bold blue]You>[/] {text}")

        if not self._session_id:
            chat_log.write("[red]No active session[/]")
            return

        # Run in background to not block UI
        self.run_worker(self._send_message(text, chat_log))

    async def _send_message(self, text: str, chat_log: RichLog) -> None:
        """Send message to gateway and display result."""
        try:
            result = await self.gateway.chat(self._session_id, text)
            if result.success:
                chat_log.write(f"\n[bold green]Assistant>[/] {result.content}")
            else:
                chat_log.write(f"\n[bold red]Error>[/] {result.content}")

            chat_log.write(
                f"[dim](tokens: {result.tokens_used}, tools: {result.tool_calls_made}, "
                f"children: {len(result.children_spawned)})[/]"
            )
        except Exception as e:
            chat_log.write(f"\n[bold red]Error>[/] {e}")

    async def _handle_command(self, command: str, chat_log: RichLog) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/help":
            chat_log.write(
                "\n[bold]Commands:[/]\n"
                "  /help     - Show this help\n"
                "  /tools    - List available tools\n"
                "  /agents   - Show agent tree\n"
                "  /hitl     - Show pending approval requests\n"
                "  /clear    - Clear chat\n"
                "  /quit     - Exit"
            )
        elif cmd == "/tools":
            for td in self.gateway.tool_registry.list_definitions():
                approval = " [yellow](approval)[/]" if td.requires_approval else ""
                cat = f"[dim]{td.category}[/]"
                chat_log.write(f"  [cyan]{td.name}[/] {cat}: {td.description}{approval}")
        elif cmd == "/agents":
            tree_data = self.gateway.get_agent_tree()
            chat_log.write(f"\n{tree_data}")
        elif cmd == "/hitl":
            pending = self.gateway.hitl.get_pending()
            if pending:
                for req in pending:
                    chat_log.write(
                        f"  [yellow]{req.request_id}[/]: {req.tool_name} - {req.description}"
                    )
            else:
                chat_log.write("  [dim]No pending approval requests[/dim]")
        elif cmd == "/clear":
            chat_log.clear()
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            chat_log.write(f"[red]Unknown command: {cmd}[/]")

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", RichLog).clear()

    async def action_quit(self) -> None:
        await self.gateway.shutdown()
        self.exit()
