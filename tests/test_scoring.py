"""Judge, programmatic, and operational scorers, plus build_scorers."""

from __future__ import annotations

import pytest

from llm_lab.evals.spec import EvalSpec, ScorerSpec
from llm_lab.models.mock import MockClient
from llm_lab.object_types import Response
from llm_lab.scoring import operational as op
from llm_lab.scoring import programmatic as pr
from llm_lab.scoring.base import ScoreContext
from llm_lab.scoring.judge import LLMJudgeScorer
from llm_lab.scoring.registry import build_scorers
from llm_lab.tasks.base import TaskOutput


def _spec(**over):
    kw = dict(name="E", labels={"good": 5, "ok": 3, "bad": 1},
              prompt="Judge {input} {output}", variables=("input", "output"))
    kw.update(over)
    return EvalSpec(**kw)


def _ctx(output="4", record=None, response=None, judge=None):
    record = record if record is not None else {"input": "2+2?", "expected": "4"}
    task_output = TaskOutput(output=output, response=response)
    return ScoreContext(record=record, output=output, task_output=task_output, judge_client=judge)


def _canned(text):
    return MockClient("mock/canned", native=f"canned:{text}")


# --- LLM judge -----------------------------------------------------------

def test_judge_parses_score_and_label():
    result = LLMJudgeScorer(_spec()).score(_ctx(judge=_canned("SCORE: 5\nREASONING: great answer")))
    assert result.score == 5.0
    assert result.label == "good"
    assert "great" in result.reasoning


def test_judge_snaps_to_nearest_allowed_score():
    # 4 is not a valid rubric score (allowed: 5, 3, 1) -> snaps to nearest.
    result = LLMJudgeScorer(_spec()).score(_ctx(judge=_canned("SCORE: 4")))
    assert result.error is None
    assert result.score in (3.0, 5.0)


def test_judge_parse_failure_sets_error_and_keeps_raw():
    result = LLMJudgeScorer(_spec()).score(_ctx(judge=_canned("I really cannot decide")))
    assert result.error and "parse" in result.error
    assert result.raw_response


def test_judge_without_client_errors():
    result = LLMJudgeScorer(_spec()).score(_ctx(judge=None))
    assert result.error and "judge_client" in result.error


def test_judge_prompt_fills_from_record_aux_and_output():
    spec = _spec(prompt="Org: {org_name}\nQ: {input}\nA: {output}",
                 variables=("org_name", "input", "output"))
    task_output = TaskOutput(output="the answer", aux={"org_name": "Acme Corp"})
    ctx = ScoreContext(record={"input": "a question"}, output="the answer", task_output=task_output)
    prompt = LLMJudgeScorer(spec)._judge_prompt(ctx)
    assert "Acme Corp" in prompt      # from aux
    assert "a question" in prompt     # from record
    assert "the answer" in prompt     # from output


# --- programmatic --------------------------------------------------------

def test_exact_match():
    sp = ScorerSpec(type="exact_match", expected_field="expected")
    assert pr.ExactMatch(sp).score(_ctx(output="4")).passed is True
    assert pr.ExactMatch(sp).score(_ctx(output="5")).passed is False


def test_exact_match_missing_field_errors():
    sp = ScorerSpec(type="exact_match", expected_field="nope")
    result = pr.ExactMatch(sp).score(_ctx(output="4"))
    assert result.error and "nope" in result.error


def test_contains_case_sensitivity():
    ci = ScorerSpec(type="contains", params={"substring": "Paris"})
    assert pr.Contains(ci).score(_ctx(output="the capital is paris")).passed is True
    cs = ScorerSpec(type="contains", params={"substring": "Paris", "case_sensitive": True})
    assert pr.Contains(cs).score(_ctx(output="the capital is paris")).passed is False


def test_regex():
    sp = ScorerSpec(type="regex", params={"pattern": r"^\d+$"})
    assert pr.Regex(sp).score(_ctx(output="42")).passed is True
    assert pr.Regex(sp).score(_ctx(output="4a2")).passed is False


def test_json_schema():
    sp = ScorerSpec(type="json_schema", params={"schema": {"type": "object", "required": ["a"]}})
    assert pr.JSONSchemaValid(sp).score(_ctx(output='{"a": 1}')).passed is True
    assert pr.JSONSchemaValid(sp).score(_ctx(output='{"b": 1}')).passed is False
    assert pr.JSONSchemaValid(sp).score(_ctx(output="not json")).passed is False


def test_numeric_tolerance():
    sp = ScorerSpec(type="numeric_tolerance", expected_field="expected", params={"atol": 0.5})
    assert pr.NumericTolerance(sp).score(_ctx(output="about 4.2", record={"expected": 4})).passed is True
    assert pr.NumericTolerance(sp).score(_ctx(output="about 9", record={"expected": 4})).passed is False


# --- operational ---------------------------------------------------------

def test_latency_scorer():
    resp = Response(text="x", model_id="m", latency_s=1.2)
    assert op.LatencyScorer(ScorerSpec(type="latency", params={"max_s": 2.0})).score(_ctx(response=resp)).passed is True
    assert op.LatencyScorer(ScorerSpec(type="latency", params={"max_s": 1.0})).score(_ctx(response=resp)).passed is False


def test_latency_scorer_missing_metadata_errors():
    result = op.LatencyScorer(ScorerSpec(type="latency")).score(_ctx(response=None))
    assert result.error


# --- build_scorers -------------------------------------------------------

def test_build_scorers_judge_toggle():
    spec = _spec(scorers=[ScorerSpec(type="exact_match", expected_field="expected")])
    with_judge = build_scorers(spec, include_judge=True)
    assert with_judge[0].name == "judge" and len(with_judge) == 2
    without = build_scorers(spec, include_judge=False)
    assert all(s.name != "judge" for s in without) and len(without) == 1


def test_build_scorers_unknown_type():
    spec = _spec(scorers=[ScorerSpec(type="bogus")])
    with pytest.raises(ValueError, match="Unknown scorer type"):
        build_scorers(spec, include_judge=False)
