"""The Scorer contract, shared by LLM-judge, programmatic, and operational scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llm_lab.models.base import ModelClient
from llm_lab.tasks.base import TaskOutput

if TYPE_CHECKING:
    from llm_lab.evals.spec import EvalSpec


@dataclass
class ScoreContext:
    record: dict[str, Any]           # the input row
    output: str                      # task output text
    task_output: TaskOutput          # full output incl. Response metadata
    eval_spec: "EvalSpec | None" = None
    judge_client: ModelClient | None = None


@dataclass
class ScoreResult:
    scorer: str
    score: float                     # normalized numeric score
    label: str | None = None         # mapped label, when applicable
    passed: bool | None = None       # for binary/threshold checks
    reasoning: str | None = None     # judge explanation
    raw_response: str | None = None
    error: str | None = None


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, ctx: ScoreContext) -> ScoreResult: ...
