"""Render the analysis: reliability, an overall means table, per-metric mean tables,
and the full statistical battery on the quality composite, then render/save.

The inferential battery (`_analyze_matrix`) runs on the quality composite only —
individual metrics get mean tables (and only under `--verbose`). Running the full
battery once on the aggregate keeps the multiple-comparisons story clean.

Tests inside `_analyze_matrix` (the caller guarantees >=2 models and >=2 prompts):
- Nonparametric: Friedman + Kendall's W when >=3 models; the single pairwise Wilcoxon
  when exactly 2. Skipped when fewer than 2 complete-case prompts remain after dropping
  NaNs (prompts not scored by every model).
- Mixed-effects model, paired Cohen's d, TOST: the composite is graded, so these always
  apply.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from llm_lab.analysis.util import short_model_id as _short

_console = Console()


def _means_table(matrix, reference: str, title: str) -> Table:
    order = matrix.mean().sort_values(ascending=False)  # skipna: each model over its own prompts
    table = Table(title=title, title_justify="left", header_style="bold")
    for col in ("model", "mean", "median", "n"):
        table.add_column(col)
    for model, mean in order.items():
        col = matrix[model].dropna()
        label = _short(model) + ("  (ref)" if model == reference else "")
        table.add_row(label, f"{mean:.3f}", f"{col.median():.2f}", str(len(col)))
    return table


def _analyze_matrix(matrix, *, reference, alpha, margin, mixed_long, verbose=False):
    """Run the full statistical battery on the graded quality-composite matrix.

    Means, Cohen's d, and TOST use each model's / each pair's available prompts (skipna);
    the Friedman/Wilcoxon omnibus uses the complete-case subset (prompts scored by every
    model), matching the source analysis. The caller guarantees >=2 models and >=2 prompts.
    """
    from llm_lab.analysis import stats as S

    metric = "quality_composite"
    k = matrix.shape[1]
    complete = matrix.dropna()  # prompts every model scored — for the nonparametric omnibus
    report: dict[str, Any] = {"metric": metric, "n_prompts": int(complete.shape[0]), "n_models": int(k),
                            "means": {m: float(matrix[m].mean()) for m in matrix.columns},
                            "notes": []}

    if verbose:
        _console.print(_means_table(matrix, reference, f"{metric}  —  mean score"))

        # --- Nonparametric omnibus (complete-case: prompts scored by every model) ----
        if complete.shape[0] < 2:
            report["notes"].append("no complete-case prompts across all models; nonparametric omnibus skipped")
            _console.print(f"[dim]{report['notes'][-1]}[/dim]")
        elif k >= 3:
            fr = S.friedman(complete, alpha=alpha)
            report["friedman"] = asdict(fr)
            sig = "SIGNIFICANT" if fr.significant else "not significant"
            _console.print(f"Friedman: chi2({fr.k - 1})={fr.chi2:.2f}, p={fr.p:.4g}  [{sig} @ alpha={alpha}]"
                        f"   Kendall's W={fr.w:.3f} ({S.interpret_w(fr.w)})  (n={fr.n} complete cases)")
            if fr.significant:
                pw = S.pairwise_wilcoxon(complete, alpha=alpha)
                report["pairwise_wilcoxon"] = pw
                for row in pw:
                    mark = "*" if row["significant"] else " "
                    rel = (f"{_short(row['higher'])} > {_short(row['lower'])}" if row["significant"]
                        else f"{_short(row['a'])} vs {_short(row['b'])}")
                    _console.print(f"  {mark} {rel:40}  raw p={row['raw_p']:.4g}  Holm p={row['holm_p']:.4g}")
        else:  # exactly 2 models
            pw = S.pairwise_wilcoxon(complete, alpha=alpha)
            report["pairwise_wilcoxon"] = pw
            row = pw[0]
            verdict = "significant" if row["significant"] else "not significant"
            _console.print(f"Wilcoxon signed-rank: p={row['raw_p']:.4g} [{verdict}]  "
                        f"(higher: {_short(row['higher'])})")

    # --- Mixed-effects model ----------------------------------------------
    mm = S.mixed_model(mixed_long, reference=reference, alpha=alpha, with_metric=True)
    report["mixed_model"] = {"reference": mm.reference, "formula": mm.formula,
                             "error": mm.error, "effects": mm.effects,
                             "summary_text": mm.summary_text}
    if mm.error:
        _console.print(f"[yellow]Mixed model skipped: {mm.error}[/yellow]")
    else:
        mtable = Table(title=f"Mixed model ({mm.formula}) — diff vs {_short(reference)}",
                       title_justify="left", header_style="bold")
        for col in ("model", "coef", "95% CI", "p", "real?"):
            mtable.add_column(col)
        for e in mm.effects:
            mtable.add_row(_short(e["model"]), f"{e['coef']:+.3f}",
                           f"[{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]", f"{e['p']:.4g}",
                           "yes" if e["real"] else "no")
        _console.print(mtable)
        if mm.summary_text and verbose:
            _console.rule("[bold]Mixed effects hierarchical model — full summary[/bold]")
            _console.print(mm.summary_text, highlight=False)
            _console.rule()

    # --- Effect sizes + equivalence ---------------------------------------
    if verbose:
        report["cohens_d"] = S.cohens_d_pairwise(matrix)
        report["tost"] = S.tost_vs_reference(matrix, reference=reference, margin=margin, alpha=alpha)
        dtable = Table(title="Paired Cohen's d", title_justify="left", header_style="bold")
        for col in ("pair", "mean diff", "d", "magnitude"):
            dtable.add_column(col)
        for row in report["cohens_d"]:
            dtable.add_row(f"{_short(row['a'])} vs {_short(row['b'])}", f"{row['mean_diff']:+.3f}",
                        f"{row['cohens_d']:+.2f}", row["magnitude"])
        _console.print(dtable)

        ttable = Table(title=f"TOST equivalence vs {_short(reference)} (margin ±{margin})",
                    title_justify="left", header_style="bold")
        for col in ("model", "mean diff", "90% CI", "TOST p", "equivalent?"):
            ttable.add_column(col)
        for row in report["tost"]:
            ttable.add_row(_short(row["model"]), f"{row['mean_diff']:+.3f}",
                        f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]", f"{row['tost_p']:.4g}",
                        "YES" if row["equivalent"] else "no")
        _console.print(ttable)
    return report


def _reliability_table(rows: list[dict]) -> Table:
    table = Table(title="Reliability — task completion", title_justify="left", header_style="bold")
    for col in ("model", "runs", "errors", "success rate"):
        table.add_column(col)
    for r in rows:
        table.add_row(_short(r["model"]), str(r["n"]), str(r["errors"]), f"{r['success_rate'] * 100:.0f}%")
    return table


def run_analysis(results_path, *, reference=None, alpha=0.05, margin=0.2, out_dir=None,
                 make_plots=True, eval_filter=None, scorer_filter=None, exclude_metrics=None,
                 verbose: bool = False) -> dict:
    from llm_lab.analysis import frame

    excluded = set(exclude_metrics or ())

    # Reliability first, from the raw results (counts failed rows too), so it shows
    # even when the quality frame is empty because a model failed every prompt.
    reliability = frame.reliability(results_path)
    _console.print(_reliability_table(reliability))
    _console.print("[dim]Reliability = fraction of prompts that produced a usable answer; "
                   "quality stats below are computed only over successful prompts.[/dim]\n")

    long_df = frame.load_long(results_path)
    if eval_filter:
        long_df = long_df[long_df["eval"].isin(set(eval_filter))]
    if scorer_filter:
        long_df = long_df[long_df["scorer"].isin(set(scorer_filter))]
    if long_df.empty:
        raise ValueError("No rows left after --eval/--scorer filters.")

    models = sorted(long_df["model"].unique())
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models to compare; found: {', '.join(models) or 'none'}.")
    reference = reference or models[0]
    if reference not in models:
        raise ValueError(f"--reference {reference!r} is not among the models: {', '.join(models)}")

    _console.print(f"[bold]Analysis[/bold]  models={len(models)}  "
                   f"metrics={long_df['metric'].nunique()}  reference={_short(reference)}  "
                   f"alpha={alpha}  margin=±{margin}")
    _console.print("[dim]Design: repeated measures, paired on record. "
                   "The full statistical battery runs on the quality composite; "
                   "individual metrics get mean tables.[/dim]")

    report: dict[str, Any] = {"models": models, "reference": reference, "alpha": alpha,
                              "margin": margin, "reliability": reliability,
                              "excluded_from_aggregate": sorted(excluded), "metrics": {}, "composite": None}
    if excluded:
        _console.print(f"[dim]Excluded from the aggregate/composite: {', '.join(sorted(excluded))} "
                       "(still analyzed individually below).[/dim]")

    # Overall performance per model over quality metrics (matches calculate_overall_means).
    quality_rows = frame.aggregate_rows(long_df, excluded)
    if not quality_rows.empty:
        om = quality_rows.groupby("model")["score"].agg(["mean", "std", "count"]).sort_values(
            "mean", ascending=False)
        report["overall_means"] = {m: {"mean": float(r["mean"]), "std": float(r["std"]),
                                       "count": int(r["count"])} for m, r in om.iterrows()}
        otable = Table(title="Overall performance by model (quality metrics)", title_justify="left",
                       header_style="bold")
        for col in ("model", "mean", "std", "n"):
            otable.add_column(col)
        for m, r in om.iterrows():
            otable.add_row(_short(m) + ("  (ref)" if m == reference else ""),
                           f"{r['mean']:.3f}", f"{r['std']:.3f}", str(int(r["count"])))
        _console.print(otable)


    if verbose:
        # Per-metric means as ONE table: rows = metric, columns = model (skipna per cell).
        _console.print("\n[bold]Per-metric mean scores[/bold] [dim](full tests run on the aggregate below)[/dim]")
        pivot = long_df.groupby(["metric", "model"])["score"].mean().unstack("model")
        model_order = list(pivot.mean(axis=0).sort_values(ascending=False).index)
        pmt = Table(title="Mean score by metric × model", title_justify="left", header_style="bold")
        pmt.add_column("metric")
        for model in model_order:
            pmt.add_column(_short(model) + ("*" if model == reference else ""))
        for metric in sorted(pivot.index):
            means = {model: (float(v) if pd.notna(v := pivot.loc[metric, model]) else None) for model in model_order}
            pmt.add_row(metric, *["—" if means[model] is None else f"{means[model]:.2f}" for model in model_order])
            report["metrics"][metric] = {"metric": metric, "means": means}
        _console.print(pmt)
        _console.print("[dim]* = reference model[/dim]")

    composite = frame.composite_quality(long_df, excluded)
    if composite is not None and composite.shape[1] >= 2 and len(composite) >= 2:
        _console.print("\n[bold cyan]Mixed Effects Model Results[/bold cyan]")
        report["composite"] = _analyze_matrix(
            composite, reference=reference, alpha=alpha, margin=margin,
            mixed_long=frame.aggregate_rows(long_df, excluded), verbose=verbose)


        if verbose:
            # Distribution-free robustness check (matches run_nonparametric_check): skewness + bootstrap CI.
            from llm_lab.analysis import stats as S

            report["composite"]["skewness"] = S.skewness(composite)
            report["composite"]["bootstrap"] = S.bootstrap_ci_vs_reference(composite, reference)
            sk = Table(title="Score skewness (negative = ceiling skew)", title_justify="left", header_style="bold")
            sk.add_column("model")
            sk.add_column("skew")
            for row in sorted(report["composite"]["skewness"], key=lambda r: r["skew"]):
                sk.add_row(_short(row["model"]), f"{row['skew']:+.2f}")
            _console.print(sk)
            bt = Table(title=f"Bootstrap 95% CI of mean diff vs {_short(reference)}", title_justify="left",
                    header_style="bold")
            for col in ("model", "mean diff", "bootstrap 95% CI"):
                bt.add_column(col)
            for row in report["composite"]["bootstrap"]:
                bt.add_row(_short(row["model"]), f"{row['mean_diff']:+.3f}",
                        f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
            _console.print(bt)

            _console.print("\n[dim]Reminders: a significant p with tiny Kendall's W is real but practically "
                        "trivial; TOST concludes equivalence, non-significance alone does not.[/dim]")

    if out_dir:
        _write_and_plot(Path(out_dir), report, long_df, make_plots)
    return report


def _write_and_plot(out_dir: Path, report: dict, long_df, make_plots: bool) -> None:
    from llm_lab.analysis import frame

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "analysis.md").write_text(_markdown(report))
    _console.print(f"[green]✓[/green] Wrote {out_dir}/analysis.json and analysis.md")

    if not make_plots:
        return
    try:
        from llm_lab.analysis import plots
    except ModuleNotFoundError as exc:
        _console.print(f"[yellow]Plots skipped ({exc}); install llm-lab[analysis].[/yellow]")
        return

    made = []
    try:
        composite = report["composite"]
        if composite and composite.get("mixed_model", {}).get("effects"):
            p = plots.forest_plot(composite["mixed_model"]["effects"], report["reference"],
                                  out_dir / "composite_effects.png", "Quality difference vs reference")
            if p:
                made.append(p)
        comp_matrix = frame.composite_quality(long_df, set(report.get("excluded_from_aggregate", [])))
        if comp_matrix is not None:
            p2 = plots.violin_plot(comp_matrix, out_dir / "composite_violin.png",
                                   "Eval Score Distribution")
            if p2:
                made.append(p2)
    except Exception as exc:  # a plotting failure must not lose the (already-written) report
        _console.print(f"[yellow]Plot generation failed: {exc}[/yellow]")
    if made:
        _console.print(f"[green]✓[/green] Wrote {len(made)} plot(s): " + ", ".join(p.name for p in made))


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    """Render a GitHub markdown table as a list of lines (trailing blank included)."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def _markdown(report: dict) -> str:
    ref = report["reference"]

    def label(model: str) -> str:
        return _short(model) + (" (ref)" if model == ref else "")

    # Display models best-first when we have an overall ranking.
    if report.get("overall_means"):
        model_order = [m for m, _ in sorted(report["overall_means"].items(), key=lambda kv: -kv[1]["mean"])]
    else:
        model_order = list(report["models"])

    lines = ["# LLM Comparison Experiment Report", ""]
    lines += _md_table(["setting", "value"], [
        ["models", ", ".join(_short(m) for m in report["models"])],
        ["reference", _short(ref)],
        ["alpha", report["alpha"]],
        ["equivalence margin", f"±{report['margin']}"],
    ])

    if report.get("reliability"):
        lines += ["## Reliability (task completion)",
                  "_Fraction of prompts that produced a usable answer; quality stats below cover only "
                  "successful prompts._", ""]
        lines += _md_table(["model", "success rate", "successful", "errors"],
                           [[_short(r["model"]), f"{r['success_rate'] * 100:.0f}%",
                             f"{r['n'] - r['errors']}/{r['n']}", r["errors"]] for r in report["reliability"]])

    if report.get("overall_means"):
        lines += ["## Summary Performance (quality overview)", ""]
        lines += _md_table(["model", "mean", "sd", "n"],
                           [[label(m), f"{s['mean']:.3f}", f"{s['std']:.3f}", s["count"]]
                            for m, s in sorted(report["overall_means"].items(), key=lambda kv: -kv[1]["mean"])])
        lines += ["![Quality distribution](composite_violin.png)"]

    if report.get("metrics"):
        lines += ["## Mean score by metric × model", ""]
        rows = []
        for metric in sorted(report["metrics"]):
            means = report["metrics"][metric].get("means", {})
            rows.append([metric] + ["—" if means.get(m) is None else f"{means[m]:.2f}" for m in model_order])
        lines += _md_table(["metric"] + [label(m) for m in model_order], rows)

    comp = report.get("composite")
    if comp:
        lines += ["## Mixed Effects Model", ""]
        if comp.get("means"):
            lines += ["**Composite mean score:**", ""]
            lines += _md_table(["model", "mean"],
                               [[label(m), f"{v:.3f}"] for m, v in sorted(comp["means"].items(),
                                                                          key=lambda kv: -kv[1])])
        if comp.get("friedman"):
            fr = comp["friedman"]
            sig = "significant" if fr["significant"] else "not significant"
            lines += [f"**Friedman:** χ²={fr['chi2']:.2f}, p={fr['p']:.4g}, Kendall's W={fr['w']:.3f} "
                      f"({sig}, n={fr['n']} complete cases)", ""]
        mm = comp.get("mixed_model")
        if mm and mm.get("error"):
            lines += [f"_Mixed model skipped: {mm['error']}_", ""]
        elif mm and mm.get("effects"):
            lines += ["![Effect Sizes and Significance](composite_effects.png)"]

            lines += [f"**Difference vs {_short(ref)}** (random intercept per prompt):", ""]
            lines += _md_table(["model", "coef", "95% CI", "p", "real?"],
                               [[_short(e["model"]), f"{e['coef']:+.3f}",
                                 f"[{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]", f"{e['p']:.4g}",
                                 "yes" if e["real"] else "no"] for e in mm["effects"]])
            if mm.get("summary_text"):
                lines += ["<details><summary>Full mixed-effects model summary</summary>", "",
                          "```", mm["summary_text"], "```", "</details>", ""]

        if comp.get("cohens_d"):
            lines += ["### Paired Cohen's d", ""]
            lines += _md_table(["pair", "mean diff", "d", "magnitude"],
                               [[f"{_short(r['a'])} vs {_short(r['b'])}", f"{r['mean_diff']:+.3f}",
                                 f"{r['cohens_d']:+.2f}", r["magnitude"]] for r in comp["cohens_d"]])
        if comp.get("tost"):
            lines += [f"### TOST equivalence vs {_short(ref)} (margin ±{report['margin']})", ""]
            lines += _md_table(["model", "mean diff", "90% CI", "TOST p", "equivalent?"],
                               [[_short(r["model"]), f"{r['mean_diff']:+.3f}",
                                 f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", f"{r['tost_p']:.4g}",
                                 "yes" if r["equivalent"] else "no"] for r in comp["tost"]])
        if comp.get("skewness"):
            lines += ["### Score skewness (negative = ceiling skew)", ""]
            lines += _md_table(["model", "skew"],
                               [[_short(r["model"]), f"{r['skew']:+.2f}"]
                                for r in sorted(comp["skewness"], key=lambda r: r["skew"])])
        if comp.get("bootstrap"):
            lines += [f"### Bootstrap 95% CI of mean diff vs {_short(ref)}", ""]
            lines += _md_table(["model", "mean diff", "bootstrap 95% CI"],
                               [[_short(r["model"]), f"{r['mean_diff']:+.3f}",
                                 f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"] for r in comp["bootstrap"]])
        for note in comp.get("notes", []):
            lines.append(f"> note: {note}")

    return "\n".join(lines)
