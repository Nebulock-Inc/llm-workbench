"""Review/calibration helpers: decision flattening, persistence, and stats."""

from __future__ import annotations

import json

from llm_workbench.reviewing import (
    append_review,
    calibration_stats,
    decision_key,
    judge_decisions,
    load_reviews,
    write_calibration_report,
)


def _write_results(tmp_path, rows):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _result(model, rid, scores, *, output="ans", ts="2026-01-01T00:00:00", error=None):
    return {"model_id": model, "record_id": rid, "input": {"event_id": rid}, "output": output,
            "prompt": "SELECT 1", "latency_s": 0.0, "usage": {},
            "error": error, "scores": scores, "timestamp": ts}


def test_judge_decisions_filters_to_judge_and_skips_errors(tmp_path):
    rows = [
        _result("m1", "r1", [
            {"eval": "E1", "scorer": "judge", "score": 5, "label": "good", "reasoning": "why"},
            {"eval": "E1", "scorer": "exact_match", "score": 0.0, "label": None},          # not a judge
            {"eval": "E2", "scorer": "judge", "score": None, "error": "parse fail"},        # errored judge
        ]),
    ]
    out = _write_results(tmp_path, rows)
    decisions = judge_decisions(out)
    assert len(decisions) == 1
    d = decisions[0]
    assert d["eval"] == "E1" and d["judge_score"] == 5 and d["judge_reasoning"] == "why"
    assert decision_key(d) == ("m1", "r1", "E1", "judge")


def test_judge_decisions_dedupes_latest_and_excludes_task_errors(tmp_path):
    rows = [
        _result("m1", "r1", [{"eval": "E1", "scorer": "judge", "score": 3}], ts="2026-01-01T00:00:00"),
        _result("m1", "r1", [{"eval": "E1", "scorer": "judge", "score": 5}], ts="2026-01-02T00:00:00"),  # newer wins
        _result("m1", "r2", [], error="task error: boom"),  # excluded by successful_only
    ]
    out = _write_results(tmp_path, rows)
    decisions = judge_decisions(out)
    assert len(decisions) == 1
    assert decisions[0]["judge_score"] == 5


def test_eval_and_model_filters(tmp_path):
    rows = [
        _result("m1", "r1", [{"eval": "E1", "scorer": "judge", "score": 4}]),
        _result("m2", "r1", [{"eval": "E2", "scorer": "judge", "score": 2}]),
    ]
    out = _write_results(tmp_path, rows)
    assert len(judge_decisions(out, eval_filter=("E1",))) == 1
    assert judge_decisions(out, model_filter=("m2",))[0]["eval"] == "E2"


def test_review_roundtrip_and_resume(tmp_path):
    path = tmp_path / "review.jsonl"
    append_review(path, {"model_id": "m1", "record_id": "r1", "eval": "E1", "scorer": "judge",
                         "judge_score": 5, "human_score": 4, "agree": False})
    # a later line for the same key overwrites on load
    append_review(path, {"model_id": "m1", "record_id": "r1", "eval": "E1", "scorer": "judge",
                         "judge_score": 5, "human_score": 5, "agree": True})
    reviews = load_reviews(path)
    assert len(reviews) == 1
    assert reviews[("m1", "r1", "E1", "judge")]["human_score"] == 5


def test_calibration_stats():
    reviews = [
        {"eval": "E1", "judge_score": 5, "human_score": 4},
        {"eval": "E1", "judge_score": 5, "human_score": 5},
        {"eval": "E2", "judge_score": 1, "human_score": 3},
        {"eval": "E2", "judge_score": 2, "human_score": 2, "note": None},
        {"eval": "E2", "judge_score": None, "human_score": 2},  # ignored (non-numeric judge)
    ]
    rows = {r["eval"]: r for r in calibration_stats(reviews)}
    assert rows["E1"]["n"] == 2
    assert rows["E1"]["mean_judge"] == 5 and rows["E1"]["mean_human"] == 4.5
    assert rows["E1"]["bias"] == 0.5
    assert rows["E1"]["exact_agree"] == 0.5 and rows["E1"]["within_1"] == 1.0
    # E2: pairs (1,3) and (2,2); within_1 = only (2,2) => 0.5
    assert rows["E2"]["n"] == 2 and rows["E2"]["within_1"] == 0.5
    assert rows["ALL"]["n"] == 4


def test_write_calibration_report(tmp_path):
    reviews = [
        {"eval": "E1", "judge_score": 5, "human_score": 4},
        {"eval": "E1", "judge_score": 5, "human_score": 5},
        {"eval": "E2", "judge_score": None, "human_score": 2},  # not counted
    ]
    out = write_calibration_report(reviews, tmp_path)
    assert out == tmp_path

    report = json.loads((tmp_path / "calibration.json").read_text())
    assert report["n_reviews"] == 2  # the non-numeric judge row is excluded
    evals = {r["eval"]: r for r in report["calibration"]}
    assert evals["E1"]["bias"] == 0.5 and "ALL" in evals

    md = (tmp_path / "calibration.md").read_text()
    assert "Judge Calibration Report" in md and "bias (J−H)" in md and "**ALL**" in md


def test_write_calibration_report_no_scored_reviews(tmp_path):
    # Nothing numeric to calibrate -> no files written, returns None.
    assert write_calibration_report([{"eval": "E1", "judge_score": None, "human_score": 2}], tmp_path) is None
    assert not (tmp_path / "calibration.json").exists()
