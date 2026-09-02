"""Matplotlib figures for the analysis report (imported only when plots are requested)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _short(model: str) -> str:
    return model.split("/")[-1]


def forest_plot(effects: list[dict], reference: str, out_path: Path, title: str) -> Path | None:
    """Forest plot of each model's mixed-model coefficient (diff vs reference) with 95% CI."""
    if not effects:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_short(e["model"]) for e in effects]

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(effects) + 1.5))
    for i, e in enumerate(effects):
        color = "#3b7dd8" if e["real"] else "#9aa0a6"  # errorbar ecolor is per-call, not a list
        ax.errorbar(e["coef"], i, xerr=[[e["coef"] - e["ci_low"]], [e["ci_high"] - e["coef"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, zorder=3)
    ax.axvline(0, color="#444", linestyle="--", linewidth=1)
    ax.set_yticks(list(range(len(effects))))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Score difference vs {_short(reference)} (0 = same)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def mean_bar_plot(matrix: pd.DataFrame, out_path: Path, title: str) -> Path | None:
    """Bar chart of mean score per model for one metric/composite (complete cases)."""
    if matrix is None or matrix.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = matrix.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(means) + 1.5))
    ax.barh([_short(m) for m in means.index], means.values, color="#3b7dd8")
    ax.invert_yaxis()
    ax.set_xlabel("mean score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
