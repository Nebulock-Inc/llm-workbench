"""Turn a run's results.jsonl into the long-format frames the tests consume.

Design mirrors the source analysis: the same records (prompts) are scored by every
model, so this is a repeated-measures / paired-on-record comparison. Each
(eval, scorer) pair is one "metric".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Scorer -> metric kind. `quality` = graded rubric (goes into the quality composite
# and gets the mixed model / Cohen's d / TOST). `operational` = latency/cost (graded
# but not "quality"). Everything else is treated as binary pass/fail.
QUALITY_SCORERS = {"judge"}
OPERATIONAL_SCORERS = {"latency", "cost"}


def metric_kind(scorer: str, n_distinct: int) -> str:
    if scorer in OPERATIONAL_SCORERS:
        return "operational"
    if scorer in QUALITY_SCORERS or n_distinct > 2:
        return "quality"
    return "binary"


def load_long(results_path: str | Path) -> pd.DataFrame:
    """Explode results.jsonl into rows of (model, prompt_id, eval, scorer, metric, score, kind).

    Rows with a missing score or a scorer error are dropped (they can't enter the stats).
    """
    path = Path(results_path)
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for score in record.get("scores") or []:
            if score.get("score") is None or score.get("error"):
                continue
            rows.append({
                "model": record["model_id"],
                "prompt_id": record["record_id"],
                "eval": score["eval"],
                "scorer": score["scorer"],
                "metric": f"{score['eval']} / {score['scorer']}",
                "score": float(score["score"]),
            })
    if not rows:
        raise ValueError(f"No numeric scores found in {path}. Nothing to analyze.")

    df = pd.DataFrame(rows)
    # Classify each metric by how many distinct values it takes across the run.
    distinct = df.groupby("metric")["score"].nunique()
    df["kind"] = df.apply(lambda r: metric_kind(r["scorer"], int(distinct[r["metric"]])), axis=1)
    return df


def metric_pivot(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Complete-case (prompt x model) matrix for one metric: prompts scored by ALL models."""
    sub = long_df[long_df["metric"] == metric]
    return sub.pivot_table(index="prompt_id", columns="model", values="score").dropna()


def composite_quality(long_df: pd.DataFrame) -> pd.DataFrame | None:
    """Prompt x model matrix of the quality composite: mean over `quality` metrics.

    Returns None when the run has no quality metrics. Complete-case (prompts scored
    by all models on every quality metric).
    """
    quality = long_df[long_df["kind"] == "quality"]
    if quality.empty:
        return None
    return quality.groupby(["prompt_id", "model"])["score"].mean().unstack().dropna()
