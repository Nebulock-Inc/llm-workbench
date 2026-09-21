"""The Task contract: turn an input record into an output to be scored."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_workbench.evals.template import fill_template
from llm_workbench.models.base import ModelClient
from llm_workbench.object_types import GenerationRequest, Response


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
    # then `llm-workbench calibrate --task module:Class` calls it per decision and renders the
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


class ReplayTask:
    """Score an answer already stored in the record — WITHOUT calling any model under test.

    `run` ignores `client` and returns the frozen answer from the record (field `output`
    by default), so the full judge / scoring / reporting pipeline runs over fixed inputs.
    Uses:

    - **Judge calibration:** feed hand-authored good/bad answers whose expected judge
      scores you know, and check the judge separates them (and that each dimension moves
      independently). Curated failures with known-low scores turn the judges themselves
      into something you can regression-test.
    - **Rubric / judge-model regression:** re-run the same fixtures after editing a rubric
      or swapping `--judge-model`, and assert the scores still land where they should.
    - **Re-scoring a past run:** point it at a previous run's rows to grade them under a
      changed rubric without paying to regenerate the answers.

    What is and isn't exercised: the model-under-test call is skipped entirely (no
    generation cost or latency), but the JUDGE model still runs — it is the thing under
    test. Because nothing is generated, `response` is None, so the `latency` scorer has no
    metadata and is effectively not-applicable for replay runs.

    Reaching your eval `{variables}`: record fields are already visible to eval prompts
    directly, so a fixture row carrying `input` / `query_results` / etc. needs no wiring.
    By default every record field is ALSO copied into `aux`, so a deterministic
    `ground_truth.aux_field` (e.g. a parsed `verdict`) resolves too — put that value in the
    fixture row. Pass `aux_fields=[...]` to copy only a subset instead of the whole record.

    Args:
        field: record key holding the answer to score (default ``"output"``).
        aux_fields: record keys to copy into ``aux``; ``None`` (default) copies all fields.
    """

    name = "replay"

    def __init__(self, field: str = "output",
                 aux_fields: list[str] | tuple[str, ...] | None = None) -> None:
        self.field = field
        self.aux_fields = None if aux_fields is None else tuple(aux_fields)

    def run(self, record: dict, client) -> TaskOutput:  # client intentionally unused
        try:
            output = record[self.field]
        except KeyError as exc:
            raise KeyError(
                f"ReplayTask: record has no field {self.field!r} to replay; "
                f"available keys: {sorted(record)}"
            ) from exc
        if self.aux_fields is None:
            aux = dict(record)
        else:
            aux = {k: record[k] for k in self.aux_fields if k in record}
        return TaskOutput(output=str(output), prompt=None, aux=aux)
