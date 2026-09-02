"""Operational scorers: latency and cost, read from Response metadata."""

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


class CostScorer:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = "cost"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        resp = ctx.task_output.response
        if resp is None or resp.cost_usd is None:
            return ScoreResult(scorer=self.name, score=0.0, error="no cost metadata")
        max_usd = self.spec.params.get("max_usd")
        passed = None if max_usd is None else resp.cost_usd <= float(max_usd)
        return ScoreResult(scorer=self.name, score=resp.cost_usd, passed=passed)
