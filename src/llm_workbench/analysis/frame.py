"""Turn a run's results.jsonl into the long-format frames the tests consume.

Design mirrors the source analysis: the same records (prompts) are scored by every
model, so this is a repeated-measures / paired-on-record comparison. Each
(eval, scorer) pair is one "metric".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_workbench.runner.results import dedupe_latest

# Scorer -> metric kind. `quality` = graded rubric (goes into the quality composite
# and gets the mixed model / Cohen's d / TOST). Everything else is treated as binary
# pass/fail. `OPERATIONAL_SCORERS` are run-time threshold metrics (latency): they are
# dropped from the statistical analysis entirely by `load_long` — kept only as run-time
# threshold checks (max_s -> passed) surfaced in results.csv and the run summary.
QUALITY_SCORERS = {"judge"}
OPERATIONAL_SCORERS = {"latency"}


def metric_kind(scorer: str, n_distinct: int) -> str:
    if scorer in QUALITY_SCORERS or n_distinct > 2:
        return "quality"
    return "binary"


def _read_records(results_path: str | Path) -> list[dict]:
    """Read every row of results.jsonl as a dict."""
    path = Path(results_path)
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_long(results_path: str | Path) -> pd.DataFrame:
    """Explode results.jsonl into rows of (model, prompt_id, eval, scorer, metric, score, kind).

    Deduped to the most recent run per (model, prompt); rows with a missing score or a
    scorer error are dropped (they can't enter the stats). Operational scorers (latency)
    are dropped too — they're run-time threshold metrics, not part of the analysis.
    """
    rows = []
    for record in dedupe_latest(_read_records(results_path), successful_only=True):
        for score in record.get("scores") or []:
            if score.get("score") is None or score.get("error"):
                continue
            if score["scorer"] in OPERATIONAL_SCORERS:  # run-time threshold metric, not analyzed
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
        raise ValueError(f"No numeric scores found in {results_path}. Nothing to analyze.")

    df = pd.DataFrame(rows)
    # Classify each metric by how many distinct values it takes across the run.
    distinct = df.groupby("metric")["score"].nunique()
    df["kind"] = df.apply(lambda r: metric_kind(r["scorer"], int(distinct[r["metric"]])), axis=1)
    return df


def reliability(results_path: str | Path) -> list[dict]:
    """Per-model task-completion reliability, read straight from results.jsonl.

    A row is a *success* when its task-level `error` is null (it produced a usable
    output that the evals could score). Unlike `load_long`, this counts EVERY recorded
    row — including task failures, which have no `scores` and are otherwise invisible to
    the analysis — so a model that fails prompts is reflected here. Returns
    [{model, n, errors, success_rate}] sorted by success_rate descending.
    """
    counts: dict[str, dict[str, int]] = {}
    for record in dedupe_latest(_read_records(results_path)):
        c = counts.setdefault(record["model_id"], {"n": 0, "errors": 0})
        c["n"] += 1
        if record.get("error"):
            c["errors"] += 1
    rows = [
        {"model": m, "n": c["n"], "errors": c["errors"],
         "success_rate": (c["n"] - c["errors"]) / c["n"] if c["n"] else 0.0}
        for m, c in counts.items()
    ]
    rows.sort(key=lambda r: (-r["success_rate"], r["model"]))
    return rows


def aggregate_rows(long_df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """The `quality` rows that feed the aggregate, minus any excluded metrics.

    A metric is excluded if its full label ('<eval> / <scorer>') or its eval name is in
    `exclude` — so a judge-scored binary metric on a different scale (e.g. an org-refusal
    guardrail) can be kept out of the composite while still shown in the per-metric mean table.
    """
    exclude = exclude or set()
    quality = long_df[long_df["kind"] == "quality"]
    return quality[~quality["metric"].isin(exclude) & ~quality["eval"].isin(exclude)]


def _zscore_per_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with `score` replaced by a within-metric z-score (population SD).

    Standardizing each metric to mean 0 / SD 1 makes metrics on different scales (e.g. a
    1-5 rubric and a 0/1 accuracy) comparable and equally weighted, so they can be pooled
    into one composite / mixed model.

    Zero-variance metrics (every row equal across the run — e.g. all models pass a
    guardrail) are DROPPED, not zeroed. They carry no signal to distinguish models, so
    keeping them as 0 rows would dilute the composite toward 0, skew per-model/per-prompt
    weighting when metric coverage differs, and add dead levels to the mixed model. Absence
    of signal is not evidence of a zero model effect.
    """
    sd = df.groupby("metric")["score"].transform(lambda s: s.std(ddof=0))
    informative = df[sd > 0].copy()
    if informative.empty:
        return informative
    informative["score"] = informative.groupby("metric")["score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    return informative


def pooled_rows(long_df: pd.DataFrame, exclude: set[str] | None = None, *,
                standardize: bool = True) -> pd.DataFrame:
    """The rows that feed the composite + mixed model.

    Default (`standardize=True`): ALL metrics (graded quality AND binary accuracy metrics
    like the ground-truth checks), each z-scored within metric, so different scales are
    pooled on a common unit-variance scale — the composite and the mixed-model coefficients
    are then in SD units. Zero-variance metrics are dropped (see `_zscore_per_metric`).
    With `standardize=False`: average the `quality`-kind metrics on their raw scale, minus
    excludes. This assumes those metrics share one rubric scale, but that is NOT enforced:
    a judge scorer is always `quality` (see `metric_kind`), so one emitting 0/1 guardrail
    scores would be pooled with graded 1-5 rubric metrics on a different scale. Exclude such
    differently-scaled judge metrics (or use `standardize=True`) to keep the raw composite
    meaningful. Operational metrics (latency) are already dropped upstream by `load_long`.
    """
    exclude = exclude or set()
    if not standardize:
        return aggregate_rows(long_df, exclude)
    rows = long_df[~long_df["metric"].isin(exclude) & ~long_df["eval"].isin(exclude)]
    return _zscore_per_metric(rows)


def composite_quality(long_df: pd.DataFrame, exclude: set[str] | None = None, *,
                      standardize: bool = True) -> pd.DataFrame | None:
    """Prompt x model matrix of the quality composite: mean over the pooled metrics.

    Returns None when no metrics remain. Not complete-cased (NaN where a model has no
    composite for a prompt); callers skipna per model/pair, complete-casing only for the
    Friedman omnibus. See `pooled_rows` for which metrics are included and whether they are
    z-scored (`standardize`).
    """
    rows = pooled_rows(long_df, exclude, standardize=standardize)
    if rows.empty:
        return None
    return rows.groupby(["prompt_id", "model"])["score"].mean().unstack()
