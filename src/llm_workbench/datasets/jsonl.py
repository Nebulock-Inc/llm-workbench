"""JSONL dataset I/O: one JSON object per line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of records (dicts). Blank lines are skipped."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: each line must be a JSON object, got {type(obj).__name__}.")
        records.append(obj)
    if not records:
        raise ValueError(f"{path}: no records found.")
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record as a JSON line (used for incremental, crash-safe results)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file (overwriting), one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")
