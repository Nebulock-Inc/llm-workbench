"""`llm-workbench power` — how many prompts do you need? (paired-design sample size)."""

from __future__ import annotations

import click

from llm_workbench.analysis.util import short_model_id
from llm_workbench.cli.output import console, teach


@click.command("power")
@click.option("--effect-size", type=float, default=None, help="Paired effect size dz to detect.")
@click.option("--mean-diff", type=float, default=None, help="Expected mean paired difference (with --sd-diff).")
@click.option("--sd-diff", type=float, default=None, help="Expected SD of the paired differences.")
@click.option("--from-run", "from_run_path", type=click.Path(exists=True), default=None,
              help="Estimate N from a pilot run directory / results.jsonl.")
@click.option("--reference", default=None, help="Baseline model (default: first) when using --from-run.")
@click.option("--eval", "eval_filter", multiple=True,
              help="With --from-run: size N on these eval names only (repeatable).")
@click.option("--scorer", "scorer_filter", multiple=True,
              help="With --from-run: size N on these scorers only, e.g. 'ground_truth' for the "
                   "AI/Maliciousness accuracy track (repeatable). Binary metrics are pooled by default.")
@click.option("--standardize/--raw", default=True, show_default=True,
              help="With --from-run: --standardize (default) pools all metrics (incl. binary accuracy "
                   "metrics like ground_truth) by z-scoring each to a common SD scale, so dz / N are in "
                   "SD units. --raw averages only the graded metrics on their raw rubric scale.")
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--power", "target_power", type=float, default=0.8, show_default=True)
@click.option("--alternative", type=click.Choice(["two-sided", "larger", "smaller"]),
              default="two-sided", show_default=True)
def power(effect_size, mean_diff, sd_diff, from_run_path, reference, eval_filter, scorer_filter,
          standardize, alpha, target_power, alternative):
    """Estimate the number of prompts needed to detect a difference between models.

    Provide a target --effect-size (or --mean-diff + --sd-diff), or --from-run to size
    the full study from a small pilot run's observed effects.

    --from-run pools all metrics into a standardized composite by default (z-scored to SD
    units). To size on a single track instead (e.g. the ground-truth accuracy when the
    judged metrics are saturated), add --scorer ground_truth. Pass --raw for the graded-only
    composite on the raw rubric scale.
    """
    try:
        import statsmodels  # noqa: F401
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            f"Power analysis needs the 'analysis' extra ({exc.name} missing). "
            "Install it with: uv sync --extra analysis"
        ) from exc

    from llm_workbench.analysis import power as P

    if from_run_path:
        from rich.table import Table

        try:
            result = P.from_run(from_run_path, reference=reference, alpha=alpha,
                                power=target_power, alternative=alternative,
                                standardize=standardize, evals=set(eval_filter),
                                scorers=set(scorer_filter))
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc

        if eval_filter or scorer_filter:
            scope = "selected metrics: " + ", ".join(result["metrics"])
        elif standardize:
            scope = "standardized composite (all metrics pooled)"
        else:
            scope = "quality composite"
        teach(f"Pilot effects on the {scope} vs {short_model_id(result['reference'])}. "
              f"'N needed' = prompts to reach {int(target_power * 100)}% power at alpha={alpha}.")
        table = Table(title="Required sample size (paired)", title_justify="left", header_style="bold")
        for col in ("model", "observed dz", "mean diff", "pilot N", "power now", "N needed"):
            table.add_column(col)
        for pair in result["pairs"]:
            n_needed = "—" if pair.n_needed is None else str(pair.n_needed)
            table.add_row(short_model_id(pair.model), f"{pair.dz:+.3f}", f"{pair.mean_diff:+.3f}",
                          str(pair.n_pilot), f"{pair.power_at_pilot * 100:.0f}%", n_needed)
        console.print(table)
        detectable = [p for p in result["pairs"] if p.n_needed is not None]
        if detectable:
            worst = max(detectable, key=lambda p: p.n_needed)
            console.print(f"\n[bold]Plan for the smallest effect:[/bold] ~{worst.n_needed} prompts "
                          f"to detect {short_model_id(worst.model)} vs reference (dz={worst.dz:+.3f}).")
        return

    dz = effect_size
    if dz is None and mean_diff is not None and sd_diff:
        dz = mean_diff / sd_diff
    if dz is None:
        raise click.ClickException("Provide --effect-size, or --mean-diff with --sd-diff, or --from-run.")

    n = P.n_for_effect(dz, alpha=alpha, power=target_power, alternative=alternative)
    console.print(f"[bold]{n} prompts[/bold] to detect a paired effect of dz={dz:.3f} "
                  f"at alpha={alpha}, power={target_power} ({alternative}).")
    teach("Paired/repeated-measures design (each prompt run on every model). The Wilcoxon test "
          "in `analyze` is ~95% as efficient, so add a small margin if you rely on it.")
