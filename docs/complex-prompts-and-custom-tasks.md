# Complex Prompts and Custom Tasks

Most experiments need nothing more than a prompt template. This guide covers the two cases beyond that: prompts with several inputs, and prompts whose inputs require extra steps to produce (scoping, retrieval, multi-step pipelines).

## Multi-variable prompts (no code)

The default task (`--prompt-template`) fills a single template from each record's fields, so **multi-variable prompts work with no code** — put any number of `{placeholders}` in the template and add matching fields to your JSONL. Tenant/org scoping, for example, is just another field:

`my_prompt.txt`:
```
You are assisting {org_name}. Use only {org_name}'s data.

Question: {input}
```

`pilot.jsonl`:
```json
{"id": "q1", "input": "Show failed logins in the last day", "org_name": "Acme Corp"}
```

`fill_template` also JSON-encodes list/dict values, so a field like `{"context": [...]}` renders as readable JSON in the prompt.

## Custom tasks (extra steps)

When producing the input takes **extra steps** — resolving identifiers, injecting scoping computed from an id, retrieving context, or a multi-call *generate → run → summarize* pipeline — write a **custom task**. A task is any class with a `name` and a `run(record, client) -> TaskOutput`, where `client` is the model under test. Point `--task module:Class` at it; your module just needs to be importable from the working directory.

```python
# examples/org_scoped_task.py
from llm_lab.object_types import GenerationRequest
from llm_lab.tasks.base import TaskOutput

class OrgScopedTask:
    name = "org-scoped"

    def run(self, record, client):
        org = record["org_name"]
        # 1. preprocess: resolve ids, inject scoping, fetch retrieval context, ...
        prompt = f"You are assisting {org}. Use ONLY {org}'s data.\n\nQuestion: {record['input']}"
        # 2. call the model under test — as many times as your workflow needs
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=512))
        # 3. return the answer, exposing derived values to eval prompts via `aux`
        return TaskOutput(output=resp.text, prompt=prompt, response=resp,
                          aux={"org_name": org})
```

Run it:

```bash
uv run llm-lab run compare --task examples.org_scoped_task:OrgScopedTask \
    --dataset pilot.jsonl --model ollama/llama3.1 --eval answer-quality \
    --judge-model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
```

A runnable version lives at [`examples/org_scoped_task.py`](../examples/org_scoped_task.py).

## What `run(record, client)` receives and returns

- **`record`** — one row of your JSONL dataset, as a dict (all fields, including your own like `org_id`, `org_name`).
- **`client`** — the model under test (a `ModelClient`). Call `client.generate(GenerationRequest(...))` as many times as your workflow needs. Need a *different* model for a sub-step (e.g. a cheap model to draft, an expensive one to finish)? Build it inside the task with `from llm_lab.models.factory import make_client; make_client("provider/model")`.
- **Returns `TaskOutput`** with:
  - `output` (str) — the final answer that the evals score.
  - `prompt` (optional) — the actual prompt sent, saved for debugging.
  - `response` (optional) — the `Response` from the model call, so the `latency` scorer has timing metadata.
  - `aux` (dict) — any derived values you want your eval prompts to reference.

## How task output reaches your evals

A judge prompt's `{variables}` are filled from three sources, merged in this order:

1. the **record** fields,
2. the task's **`aux`** dict,
3. **`output`** (the model's answer).

So a custom task can surface derived values — retrieved context, generated queries, the resolved org — as `aux` entries, and your rubric prompt references them with `{...}` exactly like `{input}` and `{output}`.

For a multi-step pipeline (e.g. natural language → generated query → executed against your system → summarized answer), do the extra model calls and external lookups inside `run`, return the final summary as `output`, and put the intermediate artifacts in `aux`:

```python
return TaskOutput(
    output=summary,
    aux={"queries": generated_query, "query_results": rows, "org_name": org},
)
```

Then an eval prompt can judge faithfulness against what actually ran:

```
Question: {input}
Queries the agent ran: {queries}
Data returned: {query_results}
Answer: {output}
```

## Tips

- Keep failures soft: model errors already surface as `Response.error`; if your task itself raises, the runner records the row as a task error and continues the sweep rather than aborting.
- Any variable your eval prompt references but the task/record doesn't provide is left as the literal `{name}` in the judge prompt — check `evals validate` and the `variables` shown by `evals list` against what your task emits.
- The task is plain Python: import your own packages, hit your own services, cache, batch — whatever your production pipeline does.
