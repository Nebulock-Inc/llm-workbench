"""Build BYO datasets: convert a CSV to JSONL, or generate prompts with an LLM."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from llm_workbench.datasets.jsonl import write_jsonl
from llm_workbench.object_types import GenerationRequest


def from_csv(csv_path: str | Path, out_path: str | Path, *, id_col: str | None = None) -> int:
    """Convert a CSV to JSONL — one record per row, every column becomes a field."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path}: no rows found.")

    records = []
    for i, row in enumerate(rows):
        record = dict(row)
        if id_col:
            if id_col not in record:
                raise ValueError(f"--id-col {id_col!r} is not a column: {', '.join(record)}")
            record["id"] = record[id_col]
        record.setdefault("id", f"row-{i}")
        records.append(record)
    write_jsonl(out_path, records)
    return len(records)


def _parse_json_array(text: str) -> list:
    """Extract the first [...] JSON array from model output (tolerates prose around it)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("model did not return a JSON array")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of objects")
    return data


def generate(client, *, instructions: str, n: int, out_path: str | Path,
             with_expected: bool = False, max_tokens: int = 4096) -> int:
    """Ask a model to generate `n` diverse test prompts for a task, write them as JSONL.

    `client` is any ModelClient (from make_client). Records are {id, input[, expected]}.
    """
    shape = '{"input": "...", "expected": "..."}' if with_expected else '{"input": "..."}'
    prompt = (
        f"Generate {n} diverse, realistic test prompts for evaluating an AI system on this task:\n\n"
        f"{instructions}\n\n"
        f"Return ONLY a JSON array of exactly {n} objects, each shaped like {shape}. "
        "Vary the difficulty and phrasing. Output nothing but the JSON array."
    )
    resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=max_tokens, temperature=0.7))
    if resp.error:
        raise ValueError(f"generation failed: {resp.error}")

    records = []
    for i, item in enumerate(_parse_json_array(resp.text or "")):
        if not isinstance(item, dict) or "input" not in item:
            continue
        record = {"id": f"gen-{i}", "input": str(item["input"])}
        if with_expected and item.get("expected") is not None:
            record["expected"] = item["expected"]
        records.append(record)
    if not records:
        raise ValueError("no usable records parsed from the model output")
    write_jsonl(out_path, records)
    return len(records)
