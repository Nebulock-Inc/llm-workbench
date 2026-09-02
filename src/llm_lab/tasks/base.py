"""The Task contract: turn an input record into an output to be scored."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_lab.models.base import ModelClient
from llm_lab.object_types import Response


@dataclass
class TaskOutput:
    output: str                                          # primary text the evals score
    prompt: str | None = None                            # actual prompt sent, for logging
    aux: dict[str, Any] = field(default_factory=dict)    # extra vars for eval templates
    response: Response | None = None                     # so latency/cost scorers see metadata


@runtime_checkable
class Task(Protocol):
    name: str

    def run(self, record: dict[str, Any], client: ModelClient) -> TaskOutput: ...
