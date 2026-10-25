"""Terminal rendering for the call: transcripts, tool trace, answers and sources."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.orchestrator import TurnResult

console = Console()


def show_banner(model: str, tools: list[str], mode: str) -> None:
    """Print the startup banner."""
    console.print(
        Panel.fit(
            f"[bold]NovaTech Voice Agent[/]  ·  mode: [cyan]{mode}[/]  ·  model: [cyan]{model}[/]\n"
            f"tools: {', '.join(tools)}\n"
            "[dim]Say or type 'goodbye' / 'end call' to hang up.[/]",
            border_style="blue",
        )
    )


def show_user(text: str) -> None:
    """Print what the caller said (or typed)."""
    console.print(f"\n[bold green]You:[/] {text}")


def show_agent(text: str) -> None:
    """Print a line spoken by the agent outside a normal turn (greeting, goodbye)."""
    console.print(f"[bold magenta]Nova:[/] {text}")


def show_turn(result: TurnResult) -> None:
    """Print the tool trace, the answer and the knowledge-base sources for one turn."""
    if result.tool_calls:
        table = Table(title="Tool calls", title_justify="left", show_lines=False, expand=False)
        table.add_column("#", style="dim")
        table.add_column("tool", style="cyan")
        table.add_column("via")
        table.add_column("arguments")
        table.add_column("ms", justify="right")
        table.add_column("ok")
        for i, call in enumerate(result.tool_calls, 1):
            table.add_row(
                str(i),
                call.name,
                "RAG" if call.provider == "local" else call.provider.upper(),
                json.dumps(call.arguments, ensure_ascii=False),
                str(call.duration_ms),
                "[green]✓[/]" if call.ok else "[red]✗[/]",
            )
        console.print(table)
    else:
        console.print("[dim](no tools used)[/]")

    console.print(f"[bold magenta]Nova:[/] {result.answer}")
    if result.sources:
        console.print("[dim]Sources: " + "; ".join(result.sources) + "[/]")


def show_error(message: str) -> None:
    """Print a non-fatal error."""
    console.print(f"[bold red]Error:[/] {message}")
