"""`llm-lab models` — list configured providers and ping a model."""

from __future__ import annotations

import click
from rich.table import Table

from llm_lab.cli.output import console, success, warn
from llm_lab.config import LabConfig
from llm_lab.models.factory import make_client  # importing registers built-in providers
from llm_lab.models.registry import REGISTRY
from llm_lab.object_types import GenerationRequest

EXTRAS = {"openai": "openai", "bedrock-mantle": "openai", "anthropic": "anthropic"}
EXAMPLES = {
    "mock": "mock/echo",
    "ollama": "ollama/llama3.1",
    "openai": "openai/gpt-4o-mini",
    "bedrock": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock-mantle": "bedrock-mantle/google.gemma-4-31b",
    "anthropic": "anthropic/claude-sonnet-4-5",
}


@click.group()
def models() -> None:
    """Inspect providers and test model connectivity."""


@models.command("list")
def list_models():
    """List registered provider prefixes."""
    table = Table(title="Providers", title_justify="left", header_style="bold")
    for col in ("prefix", "example id", "needs extra"):
        table.add_column(col)
    for prefix in sorted(REGISTRY):
        table.add_row(prefix, EXAMPLES.get(prefix, f"{prefix}/<model>"), EXTRAS.get(prefix, "—"))
    console.print(table)


@models.command("ping")
@click.argument("model_id")
@click.option("--prompt", default="Say hello in three words.", show_default=True)
@click.option("--max-tokens", type=int, default=50, show_default=True)
def ping(model_id, prompt, max_tokens):
    """Send one short prompt to MODEL_ID and report latency/result."""
    try:
        client = make_client(model_id, LabConfig.from_env())
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=max_tokens))
    if resp.error:
        warn(f"{model_id}: {resp.error}")
        return
    success(f"{model_id}  ({resp.latency_s:.2f}s, {resp.usage.total_tokens} tok)")
    console.print(resp.text.strip() or "[dim](empty)[/dim]")
