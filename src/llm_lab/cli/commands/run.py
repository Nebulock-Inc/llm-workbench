"""`llm-lab run` — run comparison experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click

from llm_lab.cli.output import console, heading, success, teach, warn
from llm_lab.config import LabConfig
from llm_lab.evals.loader import discover
from llm_lab.evals.spec import EvalSpecError, load_spec
from llm_lab.runner.comparison import run_comparison
from llm_lab.tasks.loader import load_task, task_from_prompt_file


def _resolve_models(model_ids, models_csv):
    out = list(model_ids)
    if models_csv:
        out += [m.strip() for m in models_csv.split(",") if m.strip()]
    return out


def _resolve_evals(eval_names, eval_dir):
    if not eval_names:
        return discover(eval_dir)
    specs = []
    for name in eval_names:                      # a folder name under --eval-dir, or a path
        path = Path(name)
        if not (path / "eval.yaml").is_file():
            path = Path(eval_dir) / name
        specs.append(load_spec(path))
    return specs


@click.group()
def run() -> None:
    """Run comparison experiments."""


@run.command(
    "compare",
    epilog="Example:\n"
    "  llm-lab run compare --prompt-template examples/prompt.txt --dataset examples/dataset.jsonl \\\n"
    "      --model ollama/llama3.1 --model openai/gpt-4o-mini --eval example-helpfulness \\\n"
    "      --judge-model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0\n",
)
@click.option("--task", "task_spec", default=None, help="Custom task 'module:Class'. Omit to use --prompt-template.")
@click.option("--prompt-template", "prompt_template", type=click.Path(exists=True), default=None,
              help="Prompt template file for the default one-prompt-per-row task.")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="JSONL dataset.")
@click.option("--model", "model_ids", multiple=True, help="Model id like ollama/llama3.1 (repeatable).")
@click.option("--models", "models_csv", default=None, help="Comma-separated model ids.")
@click.option("--eval", "eval_names", multiple=True,
              help="Eval folder name under --eval-dir, or a path (repeatable). Default: all under --eval-dir.")
@click.option("--eval-dir", default="evals_kit", show_default=True)
@click.option("--judge-model", default=None, help="Model id for the LLM judge (enables judge scoring).")
@click.option("--out-dir", default=None, help="Where to write results (default runs/<timestamp>).")
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--limit", type=int, default=None, help="Only the first N records (smoke tests).")
def compare(task_spec, prompt_template, dataset, model_ids, models_csv, eval_names,
            eval_dir, judge_model, out_dir, resume, limit):
    """Run each model over a dataset and score the outputs."""
    models = _resolve_models(model_ids, models_csv)
    if not models:
        raise click.ClickException("Provide at least one --model, or --models a,b,c.")

    if task_spec and prompt_template:
        raise click.ClickException("Use only one of --task or --prompt-template.")
    try:
        if task_spec:
            task = load_task(task_spec)
        elif prompt_template:
            task = task_from_prompt_file(prompt_template)
        else:
            raise click.ClickException("Provide --task module:Class or --prompt-template FILE.")
    except (ImportError, AttributeError, ValueError) as exc:
        raise click.ClickException(f"Could not load task: {exc}") from exc

    try:
        specs = _resolve_evals(eval_names, eval_dir)
    except (EvalSpecError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not specs:
        raise click.ClickException("No evals to apply.")
    if any(s.has_judge for s in specs) and not judge_model:
        warn("Some evals have a judge rubric but no --judge-model given; judge scoring will be skipped.")

    out_dir = out_dir or f"runs/{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    heading(f"Comparing {len(models)} model(s) on {len(specs)} eval(s) → {out_dir}/")
    teach("models: " + ", ".join(models) + (f"   judge: {judge_model}" if judge_model else ""))

    def on_event(kind, model, rid):
        mark = {"done": "[green]·[/green]", "skip": "[dim]-[/dim]", "error": "[red]![/red]"}.get(kind, "?")
        console.print(f"{mark} {model}  {rid}", highlight=False)

    run_comparison(task=task, dataset_path=dataset, model_ids=models, specs=specs,
                   judge_model=judge_model, config=LabConfig.from_env(), out_dir=out_dir,
                   resume=resume, limit=limit, on_event=on_event)
    success(f"Done. Results in {out_dir}/results.jsonl and results.csv")
