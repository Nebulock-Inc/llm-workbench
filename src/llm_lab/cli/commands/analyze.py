"""`llm-lab analyze` — statistical comparison of a run's results.

Heavy deps (pandas/scipy/statsmodels/matplotlib) are imported lazily so the rest of
the CLI works without the `analysis` extra installed.
"""

from __future__ import annotations

from pathlib import Path

import click

from llm_lab.cli.output import warn


@click.command("analyze")
@click.argument("results", type=click.Path(exists=True))
@click.option("--reference", default=None, help="Baseline model id for the mixed model / TOST (default: first).")
@click.option("--alpha", type=float, default=0.05, show_default=True, help="Significance level.")
@click.option("--margin", type=float, default=0.2, show_default=True, help="TOST equivalence margin (score points).")
@click.option("--eval", "eval_filter", multiple=True, help="Restrict to these eval names (repeatable).")
@click.option("--scorer", "scorer_filter", multiple=True, help="Restrict to these scorer names (repeatable).")
@click.option("--exclude-metric", "exclude_metrics", multiple=True,
              help="Keep this metric out of the aggregate/composite (by eval name or '<eval> / <scorer>' "
                   "label); still shown in the per-metric mean table. Repeatable. Use for metrics on a "
                   "different scale.")
@click.option("--out-dir", default=None, help="Write analysis.json/.md (+plots) here. Default: alongside RESULTS.")
@click.option("--plots/--no-plots", default=True, show_default=True, help="Generate matplotlib figures.")
@click.option("--verbose/--simple", default=False, help="Full stats vs. headline results only.")
def analyze(results, reference, alpha, margin, eval_filter, scorer_filter, exclude_metrics, out_dir, plots, verbose):
    """Run mixed-effects + nonparametric tests (where appropriate) on RESULTS.

    RESULTS is a run directory or its results.jsonl.
    """
    try:
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import statsmodels  # noqa: F401
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            f"The analysis tools need the 'analysis' extra ({exc.name} missing). "
            "Install it with: uv sync --extra analysis"
        ) from exc

    from llm_lab.analysis.report import run_analysis

    results_path = Path(results)
    if out_dir is None:
        out_dir = results_path if results_path.is_dir() else results_path.parent

    try:
        run_analysis(
            results_path,
            reference=reference,
            alpha=alpha,
            margin=margin,
            out_dir=out_dir,
            make_plots=plots,
            eval_filter=list(eval_filter) or None,
            scorer_filter=list(scorer_filter) or None,
            exclude_metrics=list(exclude_metrics) or None,
            verbose=verbose,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
