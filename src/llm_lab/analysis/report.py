"""Run the appropriate analyses per metric + quality composite, then render/save.

Appropriateness gating:
- Need >=2 models overall and >=2 complete-case prompts for a metric to be analyzed.
- Nonparametric: Friedman + Kendall's W when >=3 models; the single pairwise Wilcoxon
  when exactly 2. Always applicable to any paired numeric metric.
- Mixed-effects model, paired Cohen's d, TOST: only for graded metrics (quality/operational,
  i.e. >2 distinct values) and the quality composite; skipped for binary pass/fail metrics
  where a linear model / mean-difference effect size is ill-suited.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_console = Console()


def _short(model: str) -> str:
    return model.split("/")[-1]


def _means_table(matrix, reference: str, title: str) -> Table:
    order = matrix.mean().sort_values(ascending=False)
    table = Table(title=title, title_justify="left", header_style="bold")
    for col in ("model", "mean", "median", "n"):
        table.add_column(col)
    for model, mean in order.items():
        label = _short(model) + ("  (ref)" if model == reference else "")
        table.add_row(label, f"{mean:.3f}", f"{matrix[model].median():.2f}", str(len(matrix)))
    return table


def _analyze_matrix(metric, matrix, *, graded, reference, alpha, margin, mixed_long, with_metric):
    """Run the appropriate tests on one complete-case (prompt x model) matrix."""
    from llm_lab.analysis import stats as S

    n_prompts, k = matrix.shape
    report: dict[str, Any] = {"metric": metric, "n_prompts": int(n_prompts), "n_models": int(k),
                              "graded": graded, "means": {m: float(matrix[m].mean()) for m in matrix.columns},
                              "notes": []}
    if n_prompts < 2 or k < 2:
        report["notes"].append("insufficient complete-case data (need >=2 prompts and >=2 models)")
        _console.print(f"[yellow]{metric}: {report['notes'][-1]}[/yellow]")
        return report

    _console.print(_means_table(matrix, reference, f"{metric}  —  mean score (complete cases: {n_prompts})"))

    # --- Nonparametric -----------------------------------------------------
    if k >= 3:
        fr = S.friedman(matrix, alpha=alpha)
        report["friedman"] = asdict(fr)
        sig = "SIGNIFICANT" if fr.significant else "not significant"
        _console.print(f"Friedman: chi2({fr.k - 1})={fr.chi2:.2f}, p={fr.p:.4g}  [{sig} @ alpha={alpha}]"
                       f"   Kendall's W={fr.w:.3f} ({S.interpret_w(fr.w)})")
        if fr.significant:
            pw = S.pairwise_wilcoxon(matrix, alpha=alpha)
            report["pairwise_wilcoxon"] = pw
            for row in pw:
                mark = "*" if row["significant"] else " "
                rel = (f"{_short(row['higher'])} > {_short(row['lower'])}" if row["significant"]
                       else f"{_short(row['a'])} vs {_short(row['b'])}")
                _console.print(f"  {mark} {rel:40}  raw p={row['raw_p']:.4g}  Holm p={row['holm_p']:.4g}")
    else:  # exactly 2 models
        pw = S.pairwise_wilcoxon(matrix, alpha=alpha)
        report["pairwise_wilcoxon"] = pw
        row = pw[0]
        verdict = "significant" if row["significant"] else "not significant"
        _console.print(f"Wilcoxon signed-rank: p={row['raw_p']:.4g} [{verdict}]  "
                       f"(higher: {_short(row['higher'])})")

    if not graded:
        report["notes"].append("binary/pass metric — mixed model, Cohen's d, and TOST skipped")
        _console.print(f"[dim]{report['notes'][-1]}[/dim]")
        return report

    # --- Mixed-effects model ----------------------------------------------
    if mixed_long is not None:
        mm = S.mixed_model(mixed_long, reference=reference, alpha=alpha, with_metric=with_metric)
        report["mixed_model"] = {"reference": mm.reference, "formula": mm.formula,
                                 "error": mm.error, "effects": mm.effects}
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

    # --- Effect sizes + equivalence ---------------------------------------
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
                 make_plots=True, eval_filter=None, scorer_filter=None) -> dict:
    from llm_lab.analysis import frame

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
                   "Tests run only where appropriate for each metric.[/dim]")

    report: dict[str, Any] = {"models": models, "reference": reference, "alpha": alpha,
                              "margin": margin, "reliability": reliability, "metrics": {}, "composite": None}

    for metric in sorted(long_df["metric"].unique()):
        matrix = frame.metric_pivot(long_df, metric)
        kind = long_df[long_df["metric"] == metric]["kind"].iloc[0]
        graded = kind in ("quality", "operational")
        mixed_long = long_df[long_df["metric"] == metric] if graded else None
        report["metrics"][metric] = _analyze_matrix(
            metric, matrix, graded=graded, reference=reference, alpha=alpha, margin=margin,
            mixed_long=mixed_long, with_metric=False)

    composite = frame.composite_quality(long_df)
    if composite is not None and composite.shape[1] >= 2 and len(composite) >= 2:
        _console.print("\n[bold cyan]QUALITY COMPOSITE (mean over quality metrics)[/bold cyan]")
        report["composite"] = _analyze_matrix(
            "quality_composite", composite, graded=True, reference=reference, alpha=alpha, margin=margin,
            mixed_long=long_df[long_df["kind"] == "quality"], with_metric=True)

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
        comp_matrix = frame.composite_quality(long_df)
        if comp_matrix is not None:
            p2 = plots.mean_bar_plot(comp_matrix, out_dir / "composite_means.png", "Quality composite — mean score")
            if p2:
                made.append(p2)
    except Exception as exc:  # a plotting failure must not lose the (already-written) report
        _console.print(f"[yellow]Plot generation failed: {exc}[/yellow]")
    if made:
        _console.print(f"[green]✓[/green] Wrote {len(made)} plot(s): " + ", ".join(p.name for p in made))


def _markdown(report: dict) -> str:
    lines = ["# Analysis report", "",
             f"- Models: {', '.join(report['models'])}",
             f"- Reference: {report['reference']}",
             f"- alpha={report['alpha']}, equivalence margin=±{report['margin']}", ""]
    if report.get("reliability"):
        lines.append("## Reliability (task completion)")
        lines.append("- success rate = fraction of prompts that produced a usable answer "
                     "(quality metrics below cover only successful prompts)")
        for r in report["reliability"]:
            lines.append(f"  - {_short(r['model'])}: {r['success_rate'] * 100:.0f}% "
                         f"({r['n'] - r['errors']}/{r['n']}, {r['errors']} error(s))")
        lines.append("")
    sections = list(report["metrics"].items())
    if report["composite"]:
        sections.append(("quality_composite", report["composite"]))
    for name, m in sections:
        lines.append(f"## {name}")
        lines.append(f"- complete-case prompts: {m.get('n_prompts')}, models: {m.get('n_models')}")
        if m.get("means"):
            ranked = sorted(m["means"].items(), key=lambda kv: -kv[1])
            lines.append("- mean score: " + ", ".join(f"{_short(k)}={v:.3f}" for k, v in ranked))
        if "friedman" in m:
            fr = m["friedman"]
            lines.append(f"- Friedman: chi2={fr['chi2']:.2f}, p={fr['p']:.4g}, "
                         f"W={fr['w']:.3f} ({'sig' if fr['significant'] else 'ns'})")
        if m.get("mixed_model") and not m["mixed_model"].get("error"):
            for e in m["mixed_model"]["effects"]:
                lines.append(f"  - {_short(e['model'])}: {e['coef']:+.3f} "
                             f"[{e['ci_low']:+.2f},{e['ci_high']:+.2f}] p={e['p']:.4g} "
                             f"{'(real)' if e['real'] else '(ns)'}")
        for note in m.get("notes", []):
            lines.append(f"- note: {note}")
        lines.append("")
    return "\n".join(lines)
