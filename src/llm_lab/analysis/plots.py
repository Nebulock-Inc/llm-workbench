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


def violin_plot(matrix: pd.DataFrame, out_path: Path, title: str) -> Path | None:
    """Violin plot of the per-prompt score distribution per model, ordered by mean.

    Shows the full shape (spread, skew, ceiling effects) rather than just the mean, with a
    white marker at each model's mean. Each model uses its own available prompts (skipna).
    """
    if matrix is None or matrix.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = matrix.mean().sort_values(ascending=False).index
    data = [matrix[m].dropna().to_numpy() for m in order]
    means = [matrix[m].mean() for m in order]
    positions = range(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(1.4 * len(order) + 2, 5))
    parts = ax.violinplot(data, positions=list(positions), showmeans=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#3b7dd8")
        body.set_alpha(0.55)
    ax.scatter(list(positions), means, color="white", edgecolor="#22324a", zorder=3, label="mean")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([_short(m) for m in order], rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
