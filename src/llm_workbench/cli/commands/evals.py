"""`llm-workbench evals` — create (guided wizard), validate, and list evals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click
from rich.syntax import Syntax
from rich.table import Table

from llm_workbench.cli.output import console, heading, info_panel, success, teach, warn
from llm_workbench.evals.loader import discover
from llm_workbench.evals.spec import EvalSpecError, load_spec, render_spec_files, write_spec

PRESETS: dict[str, list[tuple[int, str]]] = {
    "1-5": [
        (5, "Fully and accurately satisfies the criterion."),
        (4, "Satisfies the criterion with minor gaps."),
        (3, "Partially satisfies the criterion."),
        (2, "Barely satisfies the criterion."),
        (1, "Does not satisfy the criterion."),
    ],
    "binary": [
        (1, "Satisfies the criterion."),
        (0, "Does not satisfy the criterion."),
    ],
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "eval"


def _rubric_table(labels: dict[str, float]) -> Table:
    table = Table(title="Rubric (the judge picks exactly one level)", title_justify="left", header_style="bold")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("what this level means")
    for desc, score in sorted(labels.items(), key=lambda kv: -kv[1]):
        table.add_row(str(int(score)), desc)
    return table


def _build_rubric(preset: str, interactive: bool) -> dict[str, float]:
    labels: dict[str, float] = {}
    if preset == "custom":
        teach(
            "A custom rubric is a set of score → description levels. Enter the highest score first. "
            "Good levels are mutually exclusive and describe an OBSERVABLE difference the judge can see "
            "in the output — not vague adjectives. Leave the score blank to finish."
        )
        while True:
            raw = click.prompt("  score (blank to finish)", default="", show_default=False)
            if raw == "":
                break
            try:
                score = int(raw)
            except ValueError:
                warn("  score must be a whole number")
                continue
            labels[click.prompt("  description for this score")] = score
        if not labels:
            raise click.ClickException("A custom rubric needs at least one level.")
    else:
        teach(
            f"Starting from the {preset} preset. For each level, edit the wording so it describes YOUR "
            "criterion specifically (press Enter to keep the default). The score numbers are fixed."
        )
        for score, default_desc in PRESETS[preset]:
            desc = click.prompt(f"  [{score}]", default=default_desc) if interactive else default_desc
            labels[desc] = score
    return labels


DEFAULT_INSTRUCTIONS = (
    "You are an expert evaluator. Judge the OUTPUT against the scoring criteria you are given, "
    "using only the information below."
)


def _build_prompt(variables: list[str], instructions: str | None = None) -> str:
    blocks = "\n\n".join(f"<{v}>\n{{{v}}}\n</{v}>" for v in variables)
    preamble = (instructions or "").strip() or DEFAULT_INSTRUCTIONS
    return preamble + "\n\n" + blocks + "\n"


def _parse_scorers(items: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        typ, _, field = item.partition(":")
        entry: dict[str, Any] = {"type": typ.strip()}
        if field.strip():
            entry["expected_field"] = field.strip()
        out.append(entry)
    return out


@click.group()
def evals() -> None:
    """Create, validate, and list evals.

    An "eval" is a folder (eval.yaml + prompt.md) that says how to judge a model's
    output — a scoring rubric and/or deterministic checks.
    """


@evals.command(
    "new",
    epilog="Examples:\n"
    "  llm-workbench evals new                       # full guided wizard\n"
    "  llm-workbench evals new answer-quality --labels 1-5 --yes   # non-interactive (CI)\n",
)
@click.argument("name", required=False)
@click.option("--dir", "eval_dir", default="evals_kit", show_default=True, help="Folder to create the eval in.")
@click.option("--description", default=None, help="One-line description (skips that prompt).")
@click.option("--labels", "labels_preset", type=click.Choice(["1-5", "binary", "custom"]), default=None,
              help="Rubric scale (skips that prompt).")
@click.option("--var", "cli_vars", multiple=True, help="Judge variable name (repeatable).")
@click.option("--scorer", "cli_scorers", multiple=True, help="Programmatic scorer: 'type' or 'type:field' (repeatable).")
@click.option("--force", is_flag=True, help="Overwrite an existing eval folder.")
@click.option("--yes", is_flag=True, help="Non-interactive: accept defaults (requires NAME and --labels).")
def new(name, eval_dir, description, labels_preset, cli_vars, cli_scorers, force, yes):
    """Create a new eval, walking you through the rubric and judge prompt."""
    interactive = not yes

    if interactive:
        info_panel(
            "An eval scores your model's answers. It has two parts:\n"
            "  1. a RUBRIC — the scale an LLM 'judge' scores each answer on (e.g. 1-5)\n"
            "  2. a PROMPT — what the judge reads: the question, the answer, and any context\n"
            "You can also add deterministic checks (exact match, regex, JSON-valid) that need no model.\n"
            "We'll build these step by step, preview the files, and validate before saving.",
            title="Creating a new eval",
        )

    # --- Name -------------------------------------------------------------
    heading("1. Name")
    if not name:
        if yes:
            raise click.ClickException("NAME is required with --yes.")
        teach("A short name for what this eval measures. It becomes the folder name (slugified).")
        name = click.prompt("Eval name")
    slug = _slug(name)
    dest = Path(eval_dir) / slug
    if dest.exists() and (dest / "eval.yaml").exists() and not force:
        warn(f"{dest} already has an eval — re-run with --force to overwrite, or pick another name.")
        raise click.ClickException("eval already exists")

    # --- Description ------------------------------------------------------
    heading("2. Description")
    if description is None:
        teach("One line describing the quality this eval measures. Shown in `evals list`.")
        description = "" if yes else click.prompt("Description", default="", show_default=False)

    # --- Rubric -----------------------------------------------------------
    heading("3. Rubric")
    if labels_preset is None:
        teach(
            "Pick the scoring scale:\n"
            "  • 1-5    — graded quality; use when answers vary by degree\n"
            "  • binary — pass/fail (1/0); use for a yes-or-no property (e.g. 'refused correctly')\n"
            "  • custom — define your own score → description levels"
        )
        labels_preset = click.prompt("Scale", type=click.Choice(["1-5", "binary", "custom"]), default="1-5")
    labels = _build_rubric(labels_preset, interactive)
    if interactive:
        console.print(_rubric_table(labels))

    # --- Variables --------------------------------------------------------
    heading("4. Judge variables")
    if cli_vars:
        variables = list(cli_vars)
    elif yes:
        variables = ["input", "output"]
    else:
        teach(
            "Variables are placeholders the judge prompt fills in for each test case at run time.\n"
            "  • input  — the question/prompt given to the model under test\n"
            "  • output — the model's answer being judged\n"
            "Add more if the judge needs them (e.g. 'context', 'expected'). Each must appear in the prompt as {name}."
        )
        raw = click.prompt("Variables (comma-separated)", default="input, output")
        variables = [v.strip() for v in raw.split(",") if v.strip()]

    # --- Judge instructions -----------------------------------------------
    heading("5. Judge instructions")
    instructions: str | None = None
    wrote_instructions = False
    if interactive:
        teach(
            "The judge prompt needs instructions telling the judge WHAT to evaluate and how to apply the\n"
            "rubric. Two options:\n"
            "  • write a one-line instruction now (the {var} blocks are appended automatically), or\n"
            "  • skip and edit prompt.md yourself after the files are written — we'll print the path."
        )
        if click.confirm("Write the judge instructions now?", default=True):
            instructions = click.prompt("Judge instructions")
            wrote_instructions = bool(instructions.strip())
    prompt = _build_prompt(variables, instructions)

    # --- Programmatic scorers (optional) ----------------------------------
    heading("6. Programmatic checks (optional)")
    scorers = _parse_scorers(list(cli_scorers))
    if interactive and not scorers:
        teach(
            "Optional deterministic checks that run without an LLM and compare the output to a field in\n"
            "your dataset. Examples: exact_match:expected, contains, regex, json_schema, numeric_tolerance:expected.\n"
            "Use 'type' or 'type:field'. Leave blank to skip."
        )
        raw = click.prompt("Scorers (comma-separated)", default="", show_default=False)
        scorers = _parse_scorers([s.strip() for s in raw.split(",") if s.strip()])

    # --- Preview + confirm ------------------------------------------------
    yaml_text, prompt_text = render_spec_files(
        name=name, description=description, labels=labels, prompt=prompt, scorers=scorers,
    )
    if interactive:
        heading("7. Preview")
        console.print(Syntax(yaml_text, "yaml", theme="ansi_dark", word_wrap=True))
        console.print()
        info_panel(prompt_text.rstrip(), title=f"{dest}/prompt.md")
        if not click.confirm(f"\nWrite these to {dest}/ ?", default=True):
            warn("Aborted — nothing written.")
            return

    try:
        spec = write_spec(dest_dir=dest, name=name, description=description, labels=labels,
                          prompt=prompt, scorers=scorers, force=force)
    except EvalSpecError as exc:
        raise click.ClickException(str(exc)) from exc

    success(f"Created {dest}/eval.yaml and prompt.md")
    console.print(f"  scale=1-{int(spec.max_label_score)}  vars={list(spec.variables)}  scorers={len(spec.scorers)}")
    prompt_path = dest / "prompt.md"
    if wrote_instructions:
        teach(f"Next: refine the wording in {prompt_path}, then `llm-workbench evals validate --dir {eval_dir}`.")
    else:
        teach(
            f"You skipped writing instructions — the judge prompt is a placeholder. Write it now at:\n"
            f"  {prompt_path.resolve()}\n"
            f"Then run `llm-workbench evals validate --dir {eval_dir}`."
        )


@evals.command("validate")
@click.argument("path", required=False)
@click.option("--dir", "eval_dir", default="evals_kit", show_default=True)
@click.pass_context
def validate(ctx, path, eval_dir):
    """Validate one eval (PATH) or every eval under --dir."""
    if path:
        targets = [Path(path)]
    else:
        base = Path(eval_dir)
        targets = [p for p in sorted(base.glob("*")) if (p / "eval.yaml").is_file()] if base.is_dir() else []
    if not targets:
        raise click.ClickException(f"No evals found under {eval_dir!r}.")

    ok = 0
    for target in targets:
        try:
            spec = load_spec(target)
            success(f"{spec.name}  [dim]({target})[/dim]")
            ok += 1
        except EvalSpecError as exc:
            warn(f"✗ {target}: {str(exc).splitlines()[0]}")
    console.print(f"\n[bold]{ok}/{len(targets)} valid[/bold]")
    if ok < len(targets):
        ctx.exit(1)


@evals.command("list")
@click.option("--dir", "eval_dir", default="evals_kit", show_default=True)
def list_evals(eval_dir):
    """List evals under --dir."""
    try:
        specs = discover(eval_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if not specs:
        console.print("No evals found.")
        return
    table = Table(title=f"Evals in {eval_dir}/", title_justify="left", header_style="bold")
    for col in ("name", "scoring", "variables", "description"):
        table.add_column(col)
    for spec in specs:
        scoring = []
        if spec.has_judge:
            scoring.append(f"judge 1-{int(spec.max_label_score)}")
        if spec.scorers:
            scoring.append(f"{len(spec.scorers)} check(s)")
        table.add_row(spec.name, ", ".join(scoring) or "—", ", ".join(spec.variables) or "—", spec.description or "—")
    console.print(table)
