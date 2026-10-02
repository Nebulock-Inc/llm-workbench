"""`llm-workbench init` — scaffold a starter dataset, prompt, and eval into a directory.

The starter files ship inside the package (``llm_workbench/_starter``); this command
copies them out into a directory you own and edit, so a pip/uv install can run
the quickstart without cloning the repo.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import click

from llm_workbench.cli.output import heading, success, teach, warn


def _copy_tree(src, dest: Path, *, force: bool, written: list[Path], skipped: list[Path]) -> None:
    """Recursively copy a Traversable resource tree into dest (creating dirs)."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir(), key=lambda e: e.name):
        target = dest / entry.name
        if target.is_symlink():
            # Never follow a symlink in the destination tree. Writing through
            # one (or recursing into a linked directory) could clobber files
            # outside dest, even with --force. is_symlink() also catches
            # dangling links, which exists() would report as absent.
            skipped.append(target)
        elif entry.is_dir():
            _copy_tree(entry, target, force=force, written=written, skipped=skipped)
        elif target.exists() and not force:
            skipped.append(target)
        else:
            target.write_bytes(entry.read_bytes())
            written.append(target)


@click.command("init")
@click.argument("directory", default=".", required=False)
@click.option("--force", is_flag=True, help="Overwrite files that already exist.")
def init(directory: str, force: bool) -> None:
    """Scaffold a starter dataset, prompt, and eval into DIRECTORY (default: current dir).

    Writes a tiny, self-contained example — a prompt template (prompt.txt), a
    small dataset (dataset.jsonl), and one judge eval (evals_kit/example-helpfulness)
    — so you can run the quickstart against files you own and edit.
    """
    dest = Path(directory)
    starter = files("llm_workbench") / "_starter"

    written: list[Path] = []
    skipped: list[Path] = []
    _copy_tree(starter, dest, force=force, written=written, skipped=skipped)

    heading("Scaffolded starter files")
    for path in written:
        success(str(path))
    for path in skipped:
        warn(f"skipped (exists) {path}")

    if not written:
        teach("Nothing written — all starter files already exist. Re-run with --force to overwrite.")
        return

    teach(
        "\nTry it end to end (no API keys needed — the mock provider just echoes the prompt):\n"
        "  llm-workbench run compare \\\n"
        "      --prompt-template prompt.txt --dataset dataset.jsonl \\\n"
        "      --model mock/echo --eval example-helpfulness --out-dir runs/example\n\n"
        "Then score real answers with a local model (needs Ollama running):\n"
        "  llm-workbench run compare \\\n"
        "      --prompt-template prompt.txt --dataset dataset.jsonl \\\n"
        "      --model ollama/llama3.1 --eval example-helpfulness \\\n"
        "      --judge-model ollama/llama3.1 --out-dir runs/example"
    )
