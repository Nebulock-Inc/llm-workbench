"""`llm-workbench calibrate` — calibrate the LLM judge against human judgment.

Calibration works by review: walk the judge's scored decisions from a run's results.jsonl,
show the input, the model's answer, and the judge's score + reasoning, and record your own
score on the same rubric (plus an optional note). Resumable: already-reviewed decisions are
skipped unless --re-review. At the end it prints judge-vs-human calibration (bias, exact &
within-1 agreement) per eval.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from llm_workbench.cli.output import console, heading, success, teach, warn
from llm_workbench.reviewing import (
    append_review,
    calibration_stats,
    decision_key,
    judge_decisions,
    load_reviews,
    write_calibration_report,
)


def _print_stats(reviews: list[dict], *, report_dir=None) -> None:
    rows = calibration_stats(reviews)
    if not rows:
        warn("No scored reviews yet — nothing to calibrate.")
        return
    table = Table(title="Judge ↔ human calibration", title_justify="left", header_style="bold")
    for col in ("eval", "n", "mean judge", "mean human", "bias (J−H)", "exact agree", "within 1"):
        table.add_column(col)
    for r in rows:
        style = "bold" if r["eval"] == "ALL" else ""
        table.add_row(
            r["eval"], str(r["n"]), f'{r["mean_judge"]:g}', f'{r["mean_human"]:g}',
            f'{r["bias"]:+g}', f'{r["exact_agree"] * 100:.0f}%', f'{r["within_1"] * 100:.0f}%',
            style=style,
        )
    console.print(table)
    teach("bias = mean judge − mean human (positive = judge scores higher than you).")

    if report_dir is not None:
        written = write_calibration_report(reviews, report_dir)
        if written:
            success(f"Wrote {written}/calibration.json and calibration.md")


def _render(d: dict, idx: int, total: int, *, full: bool, review_context=None) -> None:
    rec = d.get("input") or {}
    ev, model = d.get("eval"), d.get("model_id")
    heading(f"[{idx}/{total}]  {ev}   ·   {model}   ·   record {d.get('record_id')}")

    ctx = []
    if isinstance(rec, dict):
        for k in ("event_id", "org_id", "timestamp"):
            if rec.get(k):
                ctx.append(f"{k}={rec[k]}")
    if ctx:
        teach("  " + "   ".join(ctx))

    sql = d.get("prompt") or "(no query recorded)"
    console.print(Panel(sql, title="query run", title_align="left", border_style="blue"))

    # Optional task-supplied live context (e.g. re-running the query to show its results),
    # never persisted to disk. Failures here must not break the review loop.
    if review_context is not None:
        try:
            extra = review_context(d)
        except Exception as exc:  # noqa: BLE001 — surface, never crash the reviewer
            extra = f"(could not build live context: {exc})"
        if extra:
            console.print(Panel(
                str(extra), title="live context · not persisted",
                title_align="left", border_style="yellow",
            ))

    out = d.get("output") or ""
    if not full and len(out) > 1800:
        out = out[:1800] + f"\n… [{len(d['output']) - 1800} more chars — press 'o' for full]"
    console.print(Panel(out, title="model answer", title_align="left", border_style="green"))

    js, jl = d.get("judge_score"), d.get("judge_label")
    body = f"[bold]score {js}[/bold]" + (f"  ·  {jl}" if jl else "")
    if d.get("judge_reasoning"):
        body += f"\n\n{d['judge_reasoning']}"
    console.print(Panel(body, title="judge decision", title_align="left", border_style="magenta"))


_HELP = "enter a number = your score · [a]gree with judge · [s]kip · [o] full answer · [q]uit & save"


@click.command("calibrate")
@click.argument("results", type=click.Path(exists=True))
@click.option("--eval", "eval_filter", multiple=True, help="Restrict to these eval names (repeatable).")
@click.option("--model", "model_filter", multiple=True, help="Restrict to these model ids (repeatable).")
@click.option("--scorer", "scorer_filter", multiple=True, default=("judge",), show_default=True,
              help="Which scorer's decisions to review (repeatable).")
@click.option("--task", "task_spec", default=None,
              help="Task 'module:Class' whose optional review_context() supplies live, "
                   "non-persisted context (e.g. re-running a query to show its results).")
@click.option("--review-file", default=None, help="Where reviews are stored. Default: <run>/review.jsonl.")
@click.option("--re-review", is_flag=True, help="Include decisions already reviewed (default skips them).")
@click.option("--stats", "stats_only", is_flag=True, help="Just print calibration stats; no interactive review.")
@click.option("--out-dir", default=None,
              help="Where to write calibration.json/.md. Default: next to the review file.")
@click.option("--report/--no-report", default=True, show_default=True,
              help="Write calibration.json/.md alongside the console table.")
def calibrate(results, eval_filter, model_filter, scorer_filter, task_spec, review_file, re_review,
              stats_only, out_dir, report):
    """Calibrate the LLM judge by reviewing its decisions against your own judgment.

    Walks the judge's scored decisions, records your score on the same rubric, and reports
    judge-vs-human bias and agreement. RESULTS is a run directory or its results.jsonl.
    """
    results_path = Path(results)
    run_dir = results_path if results_path.is_dir() else results_path.parent
    review_path = Path(review_file) if review_file else run_dir / "review.jsonl"
    report_dir = (Path(out_dir) if out_dir else review_path.parent) if report else None

    review_context = None
    if task_spec:
        # Load .env the same way the run path does (LabConfig.from_env), so a task's
        # review_context() hook sees any credentials it needs (e.g. DB connection).
        from dotenv import find_dotenv, load_dotenv
        env_file = find_dotenv(usecwd=True)
        if env_file:
            load_dotenv(env_file)
        from llm_workbench.tasks.loader import load_task
        task = load_task(task_spec)
        review_context = getattr(task, "review_context", None)
        if not callable(review_context):
            warn(f"Task {task_spec} has no review_context() hook; no live context will be shown.")
            review_context = None

    existing = load_reviews(review_path)

    if stats_only:
        _print_stats(list(existing.values()), report_dir=report_dir)
        return

    decisions = judge_decisions(
        run_dir, scorers=tuple(scorer_filter),
        eval_filter=tuple(eval_filter), model_filter=tuple(model_filter),
    )
    if not decisions:
        raise click.ClickException(
            f"No judge decisions found in {run_dir}/results.jsonl for the given filters. "
            "(Was the run scored with a --judge-model?)"
        )

    pending = [d for d in decisions if re_review or decision_key(d) not in existing]
    heading(f"{len(decisions)} judge decision(s); {len(pending)} to review → {review_path}")
    teach(_HELP)
    if not pending:
        success("Everything is already reviewed.")
        _print_stats(list(existing.values()), report_dir=report_dir)
        return

    reviewed_now = 0
    full = False
    i = 0
    while i < len(pending):
        d = pending[i]
        _render(d, i + 1, len(pending), full=full, review_context=review_context)
        raw = click.prompt("your score", default="s", show_default=False).strip().lower()

        if raw == "q":
            break
        if raw == "s":
            i += 1
            full = False
            continue
        if raw == "o":
            full = True
            continue  # re-render same item with full output
        if raw == "a":
            human_score = d["judge_score"]
        else:
            try:
                human_score = float(raw)
                if human_score.is_integer():
                    human_score = int(human_score)
            except ValueError:
                warn("Enter a number, or a / s / o / q.")
                continue

        note = click.prompt("note (optional)", default="", show_default=False).strip()
        append_review(review_path, {
            "model_id": d["model_id"], "record_id": d["record_id"], "eval": d["eval"],
            "scorer": d["scorer"], "judge_score": d["judge_score"], "judge_label": d["judge_label"],
            "human_score": human_score, "agree": human_score == d["judge_score"],
            "note": note or None, "timestamp": datetime.now(UTC).isoformat(),
        })
        existing[decision_key(d)] = {**d, "human_score": human_score}
        reviewed_now += 1
        i += 1
        full = False

    success(f"Reviewed {reviewed_now} decision(s) this session. Saved to {review_path}")
    _print_stats(list(existing.values()), report_dir=report_dir)
