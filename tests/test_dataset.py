"""Dataset builders: CSV -> JSONL and LLM-assisted generation (offline via mock)."""

from __future__ import annotations

import json

import pytest

from llm_lab.datasets.build import from_csv, generate
from llm_lab.datasets.jsonl import read_jsonl
from llm_lab.models.mock import MockClient


def test_from_csv_writes_records(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("input,expected\n2+2?,4\ncap?,Paris\n")
    out = tmp_path / "out.jsonl"
    n = from_csv(csv_path, out)
    assert n == 2
    records = read_jsonl(out)
    assert records[0] == {"input": "2+2?", "expected": "4", "id": "row-0"}


def test_from_csv_id_col(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("qid,input\nA,2+2?\nB,cap?\n")
    out = tmp_path / "out.jsonl"
    from_csv(csv_path, out, id_col="qid")
    assert [r["id"] for r in read_jsonl(out)] == ["A", "B"]


def test_from_csv_bad_id_col(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("input\n2+2?\n")
    with pytest.raises(ValueError, match="id-col"):
        from_csv(csv_path, tmp_path / "out.jsonl", id_col="nope")


def test_generate_parses_json_array(tmp_path):
    canned = json.dumps([{"input": "What is 2+2?"}, {"input": "Name a prime."}])
    client = MockClient("mock/canned", native=f"canned:{canned}")
    out = tmp_path / "gen.jsonl"
    n = generate(client, instructions="math", n=2, out_path=out)
    assert n == 2
    records = read_jsonl(out)
    assert records[0]["input"] == "What is 2+2?" and records[0]["id"] == "gen-0"


def test_generate_tolerates_prose_around_array(tmp_path):
    canned = 'Sure! Here you go:\n[{"input": "Q1"}]\nHope that helps.'
    client = MockClient("mock/canned", native=f"canned:{canned}")
    n = generate(client, instructions="x", n=1, out_path=tmp_path / "g.jsonl")
    assert n == 1


def test_generate_rejects_non_array(tmp_path):
    client = MockClient("mock/canned", native="canned:I cannot do that")
    with pytest.raises(ValueError, match="JSON array"):
        generate(client, instructions="x", n=1, out_path=tmp_path / "g.jsonl")
