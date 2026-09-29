"""Scorer registry: map a scorer type to a factory, and build scorers from an eval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from llm_workbench.scoring import operational as op
from llm_workbench.scoring import programmatic as pr
from llm_workbench.scoring.base import Scorer
from llm_workbench.scoring.judge import LLMJudgeScorer

if TYPE_CHECKING:
    from llm_workbench.evals.spec import EvalSpec, ScorerSpec

SCORER_FACTORIES: dict[str, Callable[["ScorerSpec"], Scorer]] = {
    "exact_match": pr.ExactMatch,
    "contains": pr.Contains,
    "regex": pr.Regex,
    "json_schema": pr.JSONSchemaValid,
    "numeric_tolerance": pr.NumericTolerance,
    "latency": op.LatencyScorer,
}


def register_scorer(type_name: str, factory: Callable[["ScorerSpec"], Scorer]) -> None:
    SCORER_FACTORIES[type_name] = factory


def scorer(type_name: str):
    """Decorator sugar to register a scorer factory (class/callable) under `type_name`.

        @scorer("my_metric")
        class MyMetric:
            name = "my_metric"
            def __init__(self, spec): ...
            def score(self, ctx): ...
    """
    def _decorate(factory: Callable[["ScorerSpec"], Scorer]):
        register_scorer(type_name, factory)
        return factory

    return _decorate


def build_scorers(spec: "EvalSpec", *, include_judge: bool = True) -> list[Scorer]:
    """Instantiate the scorers an eval declares. The judge is included only when
    the eval has one AND include_judge is True (the runner sets this by whether a
    judge model was supplied)."""
    scorers: list[Scorer] = []
    if include_judge and spec.has_judge:
        scorers.append(LLMJudgeScorer(spec))
    for scorer_spec in spec.scorers:
        factory = SCORER_FACTORIES.get(scorer_spec.type)
        if factory is None:
            known = ", ".join(sorted(SCORER_FACTORIES))
            raise ValueError(f"Unknown scorer type {scorer_spec.type!r}. Known: {known}")
        scorers.append(factory(scorer_spec))
    return scorers
