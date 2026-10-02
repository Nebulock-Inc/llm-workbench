"""The Scorer contract, shared by LLM-judge, programmatic, and operational scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_workbench.models.base import ModelClient
from llm_workbench.tasks.base import TaskOutput


@dataclass
class ScoreContext:
    record: dict[str, Any]           # the input row
    output: str                      # task output text
    task_output: TaskOutput          # full output incl. Response metadata
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
    # True when the scorer deliberately did not apply to this row (e.g. a per-field metric
    # on a benign row, or a field the scenario has no ground truth for) rather than failing.
    # Distinct from `error`, which also carries real failures (judge/API errors). Lets the
    # row be filtered out of analysis without being mistaken for a 0 score. See
    # runner/results.write_csv (the `status` column) and analysis/frame.load_long.
    not_applicable: bool = False


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, ctx: ScoreContext) -> ScoreResult: ...
