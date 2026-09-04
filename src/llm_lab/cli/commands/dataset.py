"""`llm-lab dataset` — build the JSONL test set that `run compare` consumes."""

from __future__ import annotations

import click

from llm_lab.cli.output import success, teach
from llm_lab.config import LabConfig


@click.group()
def dataset() -> None:
    """Create JSONL datasets (from a CSV, or generated with an LLM)."""


@dataset.command("from-csv")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-o", "--out", required=True, type=click.Path(), help="Output .jsonl path.")
@click.option("--id-col", default=None, help="Column to use as each record's id (default: row index).")
def from_csv_cmd(csv_path, out, id_col):
    """Convert CSV rows to JSONL records (every column becomes a field)."""
    from llm_lab.datasets.build import from_csv

    try:
        n = from_csv(csv_path, out, id_col=id_col)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    success(f"Wrote {n} records to {out}")


@dataset.command("generate")
@click.option("--model", "model_id", required=True, help="Model id to generate the prompts with.")
@click.option("--instructions", required=True, help="Describe the task/domain to generate prompts for.")
@click.option("-n", "--num", type=int, default=20, show_default=True, help="How many prompts.")
@click.option("-o", "--out", required=True, type=click.Path(), help="Output .jsonl path.")
@click.option("--with-expected", is_flag=True, help="Also ask the model for an 'expected' answer per prompt.")
def generate_cmd(model_id, instructions, num, out, with_expected):
    """Generate a starter test set with an LLM (review before using)."""
    from llm_lab.datasets.build import generate
    from llm_lab.models.factory import make_client

    try:
        client = make_client(model_id, LabConfig.from_env())
        n = generate(client, instructions=instructions, n=num, out_path=out, with_expected=with_expected)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    success(f"Generated {n} records to {out}")
    teach("Generated prompts vary in quality — review and edit before running an experiment on them.")
