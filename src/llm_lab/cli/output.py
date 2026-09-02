"""Shared rich console and formatting helpers for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console()


def heading(text: str) -> None:
    console.print(f"\n[bold cyan]{text}[/bold cyan]")


def teach(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def success(text: str) -> None:
    console.print(f"[green]✓[/green] {text}")


def warn(text: str) -> None:
    console.print(f"[yellow]{text}[/yellow]")


def info_panel(body: str, title: str | None = None) -> None:
    console.print(Panel(body, title=title, title_align="left", border_style="cyan"))
