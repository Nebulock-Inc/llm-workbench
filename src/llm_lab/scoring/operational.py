"""Operational scorers: latency, read from Response metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_lab.scoring.base import ScoreContext, ScoreResult

if TYPE_CHECKING:
    from llm_lab.evals.spec import ScorerSpec


class LatencyScorer:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = "latency"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        resp = ctx.task_output.response
        if resp is None:
            return ScoreResult(scorer=self.name, score=0.0, error="no response metadata")
        max_s = self.spec.params.get("max_s")
        passed = None if max_s is None else resp.latency_s <= float(max_s)
        return ScoreResult(scorer=self.name, score=resp.latency_s, passed=passed)
