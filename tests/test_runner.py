"""Datasets, the default task, and the comparison runner (results + resume)."""

from __future__ import annotations

import json

import pytest

from llm_workbench.datasets.jsonl import read_jsonl
from llm_workbench.evals.spec import write_spec
from llm_workbench.models.mock import MockClient
from llm_workbench.runner.comparison import run_comparison
from llm_workbench.runner.results import _read_results, load_completed, summarize
from llm_workbench.tasks.base import PromptTemplateTask


def _dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id": "q1", "input": "2+2?", "expected": "4"}\n'
        '{"id": "q2", "input": "capital of France?", "expected": "Paris"}\n'
    )
    return path


def _spec(tmp_path, weight=1.0):
    return write_spec(dest_dir=tmp_path / "e", name="Help", description="",
                      labels={"good": 5, "bad": 1}, prompt="Q {input} A {output}", weight=weight)


def test_read_jsonl(tmp_path):
    records = read_jsonl(_dataset(tmp_path))
    assert len(records) == 2 and records[0]["id"] == "q1"


def test_read_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl")


def test_prompt_template_task():
    task = PromptTemplateTask("Q: {input}")
    out = task.run({"input": "hi"}, MockClient("mock/echo"))
    assert out.output == "Q: hi"
    assert out.response is not None and out.response.latency_s >= 0.0


def test_run_comparison_writes_results_and_summarizes(tmp_path):
    out = tmp_path / "runs"
    events: list[str] = []
    run_comparison(
        task=PromptTemplateTask("Q: {input}"),
        dataset_path=_dataset(tmp_path),
        model_ids=["mock/echo"],
        specs=[_spec(tmp_path, weight=2.0)],
        judge_model="mock/canned:SCORE: 5",
        out_dir=out,
        on_event=lambda kind, model, rid: events.append(kind),
    )
    assert events.count("done") == 2
    records = _read_results(out / "results.jsonl")
    assert len(records) == 2
    # The eval's weight is recorded onto each score row so the analysis is self-contained.
    assert all(s["weight"] == 2.0 for r in records for s in r["scores"])
    assert (out / "results.csv").is_file()
    assert load_completed(out) == {("mock/echo", "q1"), ("mock/echo", "q2")}

    judge_rows = [r for r in summarize(out) if r["scorer"] == "judge"]
    assert judge_rows and judge_rows[0]["n"] == 2 and judge_rows[0]["mean_score"] == 5.0


def test_run_comparison_resume_skips_completed(tmp_path):
    out = tmp_path / "runs"
    kwargs = dict(
        task=PromptTemplateTask("Q: {input}"),
        dataset_path=_dataset(tmp_path),
        model_ids=["mock/echo"],
        specs=[_spec(tmp_path)],
        judge_model="mock/canned:SCORE: 5",
        out_dir=out,
    )
    run_comparison(**kwargs)  # first pass
    events: list[str] = []
    run_comparison(on_event=lambda kind, m, r: events.append(kind), **kwargs)  # rerun
    assert events.count("skip") == 2 and "done" not in events
    assert len(_read_results(out / "results.jsonl")) == 2  # no new rows appended


def test_run_comparison_records_task_errors(tmp_path):
    class Boom:
        name = "boom"

        def run(self, record, client):
            raise RuntimeError("bad task")

    out = tmp_path / "runs"
    run_comparison(task=Boom(), dataset_path=_dataset(tmp_path), model_ids=["mock/echo"],
                   specs=[_spec(tmp_path)], out_dir=out, limit=1)
    results = _read_results(out / "results.jsonl")
    assert len(results) == 1 and "task error" in results[0]["error"]


def test_custom_task_with_aux_end_to_end(tmp_path):
    from llm_workbench.tasks.loader import load_task

    ds = tmp_path / "d.jsonl"
    ds.write_text('{"id": "q1", "input": "hello", "org_name": "Acme"}\n')
    spec = write_spec(dest_dir=tmp_path / "e", name="Org", description="",
                      labels={"ok": 1, "bad": 0}, prompt="Org {org_name} | Q {input} | A {output}")
    task = load_task("examples.org_scoped_task:OrgScopedTask")  # importable from repo root
    out = tmp_path / "runs"
    run_comparison(task=task, dataset_path=ds, model_ids=["mock/echo"], specs=[spec],
                   judge_model="mock/canned:SCORE: 1", out_dir=out)
    results = _read_results(out / "results.jsonl")
    assert len(results) == 1 and results[0]["error"] is None
    assert results[0]["scores"][0]["score"] == 1.0


def test_requires_aux_gates_every_scorer(tmp_path):
    # A spec scoped with `requires_aux` must skip ALL its scorers (judge + deterministic)
    # on rows missing the required task-aux value, not just the judge.
    from llm_workbench.evals.spec import load_spec
    from llm_workbench.tasks.loader import load_task

    ds = tmp_path / "d.jsonl"
    ds.write_text('{"id": "q1", "output": "ans one", "verdict": "AI"}\n'
                  '{"id": "q2", "output": "ans two"}\n')  # q2 has no verdict
    edir = tmp_path / "e"
    edir.mkdir()
    (edir / "eval.yaml").write_text(
        "name: V\nrequires_aux: verdict\nlabels:\n  good: 1\n  bad: 0\n"
        "prompt: 'Grade {output}'\nscorers:\n  - type: contains\n    params:\n      substring: ans\n"
    )
    out = tmp_path / "runs"
    run_comparison(
        task=load_task("llm_workbench.tasks:ReplayTask"),  # copies record fields into aux
        dataset_path=ds, model_ids=["mock/echo"], specs=[load_spec(edir)],
        judge_model="mock/canned:SCORE: 1", out_dir=out,
    )
    by_id = {r["record_id"]: r for r in _read_results(out / "results.jsonl")}
    scorers = {"judge", "contains"}
    # q1 carries the verdict -> every scorer runs, no not-applicable error.
    assert {s["scorer"] for s in by_id["q1"]["scores"]} == scorers
    assert all(s["error"] is None for s in by_id["q1"]["scores"])
    # q2 lacks it -> every scorer is skipped as not-applicable (flagged, not just an error string).
    assert {s["scorer"] for s in by_id["q2"]["scores"]} == scorers
    assert all("not applicable" in (s["error"] or "") for s in by_id["q2"]["scores"])
    assert all(s["not_applicable"] for s in by_id["q2"]["scores"])
    assert not any(s["not_applicable"] for s in by_id["q1"]["scores"])


def test_summary_and_csv_dedupe_to_latest(tmp_path):
    from llm_workbench.runner.results import summarize, write_csv

    def row(ts, score):
        return json.dumps({"model_id": "mock/m1", "record_id": "q0", "input": {}, "output": "",
                           "prompt": None, "latency_s": 0.1, "usage": {},
                           "error": None, "timestamp": ts,
                           "scores": [{"eval": "E", "scorer": "judge", "score": score, "error": None}]})

    (tmp_path / "results.jsonl").write_text(row("2026-01-01T00:00:00", 2.0) + "\n" +
                                            row("2026-02-01T00:00:00", 5.0) + "\n")
    judge = next(r for r in summarize(tmp_path) if r["scorer"] == "judge")
    assert judge["n"] == 1 and judge["mean_score"] == 5.0   # latest, not n=2 or mean 3.5

    write_csv(tmp_path)
    lines = (tmp_path / "results.csv").read_text().strip().splitlines()
    assert len(lines) == 2  # header + one deduped row


def test_csv_status_column(tmp_path):
    import csv

    from llm_workbench.runner.results import write_csv

    def result(scores):
        return json.dumps({"model_id": "mock/m1", "record_id": "q0", "input": {}, "output": "",
                           "prompt": None, "latency_s": 0.1, "usage": {}, "error": None,
                           "timestamp": "2026-01-01T00:00:00", "scores": scores})

    (tmp_path / "results.jsonl").write_text(result([
        {"eval": "E", "scorer": "ok", "score": 1.0, "error": None},                       # scored
        {"eval": "E", "scorer": "flagged", "score": 0.0, "error": "n/a: benign row",
         "not_applicable": True},                                                         # flag wins
        {"eval": "E", "scorer": "legacy", "score": 0.0, "error": "n/a: no evidence cited"},  # old file: prefix fallback
        {"eval": "E", "scorer": "boom", "score": 0.0, "error": "judge error: throttled"},  # real failure
    ]) + "\n")

    write_csv(tmp_path)
    with (tmp_path / "results.csv").open() as fh:
        status = {r["scorer"]: r["status"] for r in csv.DictReader(fh)}
    assert status == {"ok": "scored", "flagged": "not_applicable",
                      "legacy": "not_applicable", "boom": "error"}


def test_limit(tmp_path):
    out = tmp_path / "runs"
    run_comparison(task=PromptTemplateTask("Q {input}"), dataset_path=_dataset(tmp_path),
                   model_ids=["mock/echo"], specs=[_spec(tmp_path)], out_dir=out, limit=1)
    assert len(_read_results(out / "results.jsonl")) == 1
