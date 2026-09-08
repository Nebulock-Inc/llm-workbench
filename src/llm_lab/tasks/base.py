"""The Task contract: turn an input record into an output to be scored."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_lab.evals.template import fill_template
from llm_lab.models.base import ModelClient
from llm_lab.object_types import GenerationRequest, Response


@dataclass
class TaskOutput:
    output: str                                          # primary text the evals score
    prompt: str | None = None                            # actual prompt sent, for logging
    aux: dict[str, Any] = field(default_factory=dict)    # extra vars for eval templates
    response: Response | None = None                     # so the latency scorer sees metadata


@runtime_checkable
class Task(Protocol):
    name: str

    def run(self, record: dict[str, Any], client: ModelClient) -> TaskOutput: ...

    # Optional hook (duck-typed; not required by the Protocol): if a task defines
    #   def review_context(self, decision: dict[str, Any]) -> str | None
    # then `llm-lab review --task module:Class` calls it per decision and renders the
    # returned text as an extra panel. Use it to surface context that is deliberately
    # NOT persisted to results.jsonl — e.g. re-running a query live so a human reviewer
    # can see its results without customer data ever touching disk. Return None to show
    # nothing. `decision` carries at least: model_id, record_id, eval, input (the record),
    # output (model answer), prompt (the prompt/SQL sent), and the judge_* fields.


class PromptTemplateTask:
    """Default task: fill a {placeholder} template from each record, call the model,
    return its answer. Covers the common 'one prompt per input row' case with no code.
    """

    def __init__(self, template: str, *, name: str = "prompt-template",
                 system: str | None = None, max_tokens: int | None = None) -> None:
        self.template = template
        self.name = name
        self.system = system
        self.max_tokens = max_tokens

    def run(self, record: dict, client) -> TaskOutput:
        prompt = fill_template(self.template, record)
        resp = client.generate(GenerationRequest(prompt=prompt, system=self.system, max_tokens=self.max_tokens))
        return TaskOutput(output=resp.text, prompt=prompt, response=resp)
