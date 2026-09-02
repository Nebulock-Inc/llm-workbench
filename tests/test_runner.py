"""Datasets, the default task, and the comparison runner (results + resume)."""

from __future__ import annotations

import pytest

from llm_lab.datasets.jsonl import read_jsonl
from llm_lab.evals.spec import write_spec
from llm_lab.models.mock import MockClient
from llm_lab.runner.comparison import run_comparison
from llm_lab.runner.results import _read_results, load_completed, summarize
from llm_lab.tasks.base import PromptTemplateTask


def _dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id": "q1", "input": "2+2?", "expected": "4"}\n'
        '{"id": "q2", "input": "capital of France?", "expected": "Paris"}\n'
    )
    return path


def _spec(tmp_path):
    return write_spec(dest_dir=tmp_path / "e", name="Help", description="",
                      labels={"good": 5, "bad": 1}, prompt="Q {input} A {output}")


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
        specs=[_spec(tmp_path)],
        judge_model="mock/canned:SCORE: 5",
        out_dir=out,
        on_event=lambda kind, model, rid: events.append(kind),
    )
    assert events.count("done") == 2
    assert len(_read_results(out / "results.jsonl")) == 2
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
    from llm_lab.tasks.loader import load_task

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


def test_limit(tmp_path):
    out = tmp_path / "runs"
    run_comparison(task=PromptTemplateTask("Q {input}"), dataset_path=_dataset(tmp_path),
                   model_ids=["mock/echo"], specs=[_spec(tmp_path)], out_dir=out, limit=1)
    assert len(_read_results(out / "results.jsonl")) == 1
