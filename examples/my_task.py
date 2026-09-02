"""Example bring-your-own task. Point --task at it: examples.my_task:UppercaseEchoTask"""

from llm_lab.object_types import GenerationRequest
from llm_lab.tasks.base import TaskOutput


class UppercaseEchoTask:
    name = "uppercase-echo"

    def run(self, record, client):
        resp = client.generate(GenerationRequest(prompt=record["input"], max_tokens=50))
        return TaskOutput(output=resp.text.upper(), prompt=record["input"], response=resp)
