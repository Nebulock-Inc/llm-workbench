"""Human review + judge calibration: pure helpers behind `llm-lab review`.

Turns a run's `results.jsonl` judge scores into reviewable decisions, persists the
reviewer's own score per decision, and computes calibration stats (judge vs. human
bias and agreement). Kept free of I/O prompts so it is unit-testable; the interactive
loop lives in `cli/commands/review.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_lab.runner.results import _read_results, dedupe_latest

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
