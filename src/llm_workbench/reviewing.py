"""Human review + judge calibration: pure helpers behind `llm-workbench calibrate`.

Turns a run's `results.jsonl` judge scores into reviewable decisions, persists the
reviewer's own score per decision, and computes calibration stats (judge vs. human
bias and agreement). Kept free of I/O prompts so it is unit-testable; the interactive
loop lives in `cli/commands/calibrate.py`.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_workbench.runner.results import _read_results, dedupe_latest

# A review is keyed by this tuple so re-runs resume and overwrite cleanly.
ReviewKey = tuple[str, str, str, str]  # (model_id, record_id, eval, scorer)


def decision_key(d: dict[str, Any]) -> ReviewKey:
    return (d.get("model_id"), d.get("record_id"), d.get("eval"), d.get("scorer"))


def judge_decisions(
    out_dir: str | Path,
    *,
    scorers: tuple[str, ...] = ("judge",),
    eval_filter: tuple[str, ...] = (),
    model_filter: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Flatten a run's latest results into one decision per (model, record, eval, scorer).

    Only scored rows are included (task-error rows have no scores). By default only the
    LLM `judge` scorer is returned — that is what calibration is about.
    """
    rows = dedupe_latest(_read_results(Path(out_dir) / "results.jsonl"), successful_only=True)
    decisions: list[dict[str, Any]] = []
    for r in rows:
        if model_filter and r.get("model_id") not in model_filter:
            continue
        for s in r.get("scores") or []:
            if scorers and s.get("scorer") not in scorers:
                continue
            if s.get("score") is None or s.get("error"):
                continue
            if eval_filter and s.get("eval") not in eval_filter:
                continue
            decisions.append({
                "model_id": r.get("model_id"),
                "record_id": r.get("record_id"),
                "eval": s.get("eval"),
                "scorer": s.get("scorer"),
                "judge_score": s.get("score"),
                "judge_label": s.get("label"),
                "judge_reasoning": s.get("reasoning"),
                "input": r.get("input"),
                "output": r.get("output"),
                "prompt": r.get("prompt"),
            })
    return decisions


def load_reviews(path: str | Path) -> dict[ReviewKey, dict[str, Any]]:
    """Load existing reviews keyed by (model, record, eval, scorer); latest line wins."""
    path = Path(path)
    reviews: dict[ReviewKey, dict[str, Any]] = {}
    if not path.is_file():
        return reviews
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        reviews[(r.get("model_id"), r.get("record_id"), r.get("eval"), r.get("scorer"))] = r
    return reviews


def append_review(path: str | Path, review: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(review, default=str) + "\n")


def calibration_stats(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-eval (plus an ALL row) judge-vs-human calibration.

    For each eval: n, mean judge score, mean human score, bias (judge − human),
    exact-agreement rate, and within-1 rate. Only reviews with numeric judge & human
    scores are counted.
    """
    by_eval: dict[str, list[tuple[float, float]]] = {}
    for r in reviews:
        j, h = r.get("judge_score"), r.get("human_score")
        if not isinstance(j, (int, float)) or not isinstance(h, (int, float)):
            continue
        # NaN/±inf are floats and slip past the isinstance check; drop them so they
        # can't poison the means or reach the JSON report as non-standard tokens.
        if not math.isfinite(j) or not math.isfinite(h):
            continue
        by_eval.setdefault(r.get("eval", "?"), []).append((float(j), float(h)))

    def _row(name: str, pairs: list[tuple[float, float]]) -> dict[str, Any]:
        n = len(pairs)
        mj = sum(j for j, _ in pairs) / n
        mh = sum(h for _, h in pairs) / n
        exact = sum(1 for j, h in pairs if j == h) / n
        within1 = sum(1 for j, h in pairs if abs(j - h) <= 1) / n
        return {
            "eval": name, "n": n,
            "mean_judge": round(mj, 3), "mean_human": round(mh, 3),
            "bias": round(mj - mh, 3), "exact_agree": round(exact, 3), "within_1": round(within1, 3),
        }

    rows = [_row(ev, pairs) for ev, pairs in sorted(by_eval.items()) if pairs]
    all_pairs = [p for pairs in by_eval.values() for p in pairs]
    if all_pairs:
        rows.append(_row("ALL", all_pairs))
    return rows


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a GitHub markdown table as a list of lines (trailing blank included)."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def calibration_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the calibration_stats() rows as a standalone markdown report."""
    lines = [
        "# Judge Calibration Report", "",
        ("_Judge-vs-human agreement. **Bias** = mean judge − mean human (positive = the judge "
         "scores higher than the reviewer). Read bias alongside the exact- and within-1-agreement "
         "rates: a large bias or low agreement is a signal to revise the rubric or judge prompt "
         "before trusting the run's scores._"), "",
    ]
    table_rows = [
        [f"**{r['eval']}**" if r["eval"] == "ALL" else r["eval"], r["n"],
         f"{r['mean_judge']:g}", f"{r['mean_human']:g}", f"{r['bias']:+g}",
         f"{r['exact_agree'] * 100:.0f}%", f"{r['within_1'] * 100:.0f}%"]
        for r in rows
    ]
    lines += _md_table(
        ["eval", "n", "mean judge", "mean human", "bias (J−H)", "exact agree", "within 1"],
        table_rows,
    )
    return "\n".join(lines)


def write_calibration_report(reviews: list[dict[str, Any]], out_dir: str | Path) -> Path | None:
    """Write `calibration.json` + `calibration.md` for these reviews into out_dir.

    Returns out_dir, or None if there are no scored reviews to report (nothing written).
    """
    rows = calibration_stats(reviews)
    if not rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_scored = sum(
        1 for r in reviews
        if isinstance(r.get("judge_score"), (int, float)) and isinstance(r.get("human_score"), (int, float))
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_reviews": n_scored,
        "calibration": rows,
    }
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    (out_dir / "calibration.md").write_text(calibration_markdown(rows))
    return out_dir
