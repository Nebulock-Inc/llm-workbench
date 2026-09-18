"""Result records: shape, incremental save, CSV flattening, and summary."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_console = Console()


def record_id(record: dict, index: int) -> str:
    """Stable per-record id: the record's 'id' field, else a deterministic hash."""
    if "id" in record:
        return str(record["id"])
    blob = json.dumps(record, sort_keys=True, default=str)
    return f"rec-{index}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


@dataclass
class ScoreRow:
    eval: str
    scorer: str
    score: float
    label: str | None = None
    passed: bool | None = None
    reasoning: str | None = None
    error: str | None = None


@dataclass
class RunResult:
    model_id: str
    record_id: str
    input: dict[str, Any]
    output: str
    prompt: str | None
    latency_s: float
    usage: dict[str, Any]
    error: str | None
    scores: list[ScoreRow]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_results(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def dedupe_latest(records: list[dict[str, Any]], *, successful_only: bool = False) -> list[dict[str, Any]]:
    """Keep only the most recent row per (model_id, record_id), by timestamp.

    Re-runs, retries, and grown pilots append new rows for a (model, prompt) pair, so
    duplicates must collapse to one — the latest attempt — rather than being counted
    twice or averaged together.

    `successful_only=True` first drops rows with a task-level error, so the QUALITY path
    keeps the latest *successful* run per pair (matching the source pipeline, which
    excludes failures then keeps last). Reliability leaves it False to reflect the final
    outcome — including a failure that superseded an earlier success.
    """
    if successful_only:
        records = [r for r in records if not r.get("error")]
    latest: dict[tuple, dict[str, Any]] = {}
    for record in records:
        key = (record.get("model_id"), record.get("record_id"))
        current = latest.get(key)
        if current is None or str(record.get("timestamp", "")) >= str(current.get("timestamp", "")):
            latest[key] = record
    return list(latest.values())


def load_completed(out_dir: str | Path) -> set[tuple[str, str]]:
    """The (model_id, record_id) pairs already present in results.jsonl."""
    return {(r.get("model_id"), r.get("record_id")) for r in _read_results(Path(out_dir) / "results.jsonl")}


def write_csv(out_dir: str | Path) -> Path:
    """Flatten results.jsonl to one CSV row per (model, record, scorer)."""
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for r in dedupe_latest(_read_results(out_dir / "results.jsonl"), successful_only=True):
        base = {"model_id": r["model_id"], "record_id": r["record_id"],
                "latency_s": r["latency_s"]}
        scores = r.get("scores") or []
        if scores:
            for s in scores:
                rows.append({**base, "eval": s["eval"], "scorer": s["scorer"], "score": s["score"],
                             "label": s.get("label"), "passed": s.get("passed"), "score_error": s.get("error")})
        else:
            rows.append({**base, "eval": None, "scorer": None, "score": None,
                         "label": None, "passed": None, "score_error": None})
    cols = ["model_id", "record_id", "eval", "scorer", "score", "label", "passed",
            "latency_s", "score_error"]
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in cols})
    return csv_path


def summarize(out_dir: str | Path) -> list[dict[str, Any]]:
    """Aggregate per (model, eval, scorer): mean score, pass rate, n, error count."""
    agg: dict[tuple, list] = defaultdict(list)
    for r in dedupe_latest(_read_results(Path(out_dir) / "results.jsonl"), successful_only=True):
        for s in r.get("scores") or []:
            agg[(r["model_id"], s["eval"], s["scorer"])].append((s.get("score"), s.get("passed"), s.get("error")))
    rows = []
    for (model, ev, scorer), vals in sorted(agg.items()):
        scores = [v[0] for v in vals if v[0] is not None and v[2] is None]
        passes = [v[1] for v in vals if v[1] is not None]
        rows.append({
            "model": model, "eval": ev, "scorer": scorer, "n": len(vals),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "pass_rate": round(sum(1 for p in passes if p) / len(passes), 3) if passes else None,
            "errors": sum(1 for v in vals if v[2]),
        })
    return rows


def print_summary(out_dir: str | Path) -> None:
    results = dedupe_latest(_read_results(Path(out_dir) / "results.jsonl"))
    rows = summarize(out_dir)
    if not rows:
        _console.print("No scores to summarize.")
        return

    table = Table(title="Comparison summary", title_justify="left", header_style="bold")
    for col in ("model", "eval", "scorer", "n", "mean score", "pass rate", "errors"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["model"], r["eval"], r["scorer"], str(r["n"]),
            "" if r["mean_score"] is None else f'{r["mean_score"]:g}',
            "" if r["pass_rate"] is None else f'{r["pass_rate"] * 100:.0f}%',
            str(r["errors"]),
        )
    _console.print(table)

    per_model: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "lat": 0.0, "err": 0})
    for r in results:
        m = per_model[r["model_id"]]
        m["n"] += 1
        m["lat"] += r["latency_s"] or 0.0
        m["err"] += 1 if r["error"] else 0
    mtable = Table(title="Per-model reliability", title_justify="left", header_style="bold")
    for col in ("model", "runs", "task errors", "mean latency (s)"):
        mtable.add_column(col)
    for model, m in sorted(per_model.items()):
        mtable.add_row(model, str(int(m["n"])), str(int(m["err"])),
                       f'{m["lat"] / m["n"]:.2f}' if m["n"] else "-")
    _console.print(mtable)
