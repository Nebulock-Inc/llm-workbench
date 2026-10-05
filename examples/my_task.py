"""Example bring-your-own task — the minimal shape of a custom `--task`.

Runs on the shipped dataset (each row has an `input` field):
    uv run llm-workbench run compare --task examples.my_task:UppercaseEchoTask \
        --dataset examples/dataset.jsonl --model mock/echo --eval example-helpfulness

A task is any class with a `name` and `run(record, client) -> TaskOutput`. This one just
calls the model and uppercases the answer, to show the mechanism; see org_scoped_task.py
for a task that injects context and exposes derived values to the eval via `aux`.
"""

from llm_workbench.object_types import GenerationRequest
from llm_workbench.tasks.base import TaskOutput


class UppercaseEchoTask:
    name = "uppercase-echo"

    def run(self, record, client):
        resp = client.generate(GenerationRequest(prompt=record["input"], max_tokens=50))
        return TaskOutput(output=resp.text.upper(), prompt=record["input"], response=resp)
