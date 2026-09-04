"""Example custom task for prompts that need extra steps before/after the model call.

This mirrors a tenant-scoped workflow: each record carries an org, the task injects
org scoping into the prompt, calls the model under test, and exposes the resolved org
(and any other derived values) to the eval prompts via `aux`.

Run it with:
    uv run llm-lab run compare --task examples.org_scoped_task:OrgScopedTask \
        --dataset examples/org_scoped_dataset.jsonl --model ollama/llama3.1 \
        --eval example-helpfulness --judge-model ollama/llama3.1

Dataset rows carry an `input` and the `org_name` this task scopes to (see
examples/org_scoped_dataset.jsonl):
    {"id": "q1", "input": "Show failed logins in the last day", "org_name": "Acme Corp"}
"""

from __future__ import annotations

from llm_lab.object_types import GenerationRequest
from llm_lab.tasks.base import TaskOutput


class OrgScopedTask:
    name = "org-scoped"

    def run(self, record, client):
        org = record["org_name"]

        # 1. Preprocess / inject scoping. In a real pipeline this is where you'd
        #    resolve identifiers, look up tenant IDs, fetch retrieval context, etc.
        prompt = (
            f"You are a security assistant for {org}. Use ONLY {org}'s data; never "
            f"reference any other organization.\n\nQuestion: {record['input']}"
        )

        # 2. Call the model under test. Call it as many times as your workflow needs
        #    (e.g. generate a query, run it against your system, then summarize).
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=512))

        # 3. Return the final answer, and expose any derived values your eval prompts
        #    reference as {placeholders} via `aux` (here: {org_name}).
        return TaskOutput(
            output=resp.text,
            prompt=prompt,
            response=resp,            # lets the latency scorer read timing metadata
            aux={"org_name": org},    # a rubric prompt can now use {org_name}
        )
