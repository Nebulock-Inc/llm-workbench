---
name: define-task
description: Use to define the task being tested in llm-workbench — decide between the default prompt-template task (no code) and a custom Python task, and write whichever fits. Triggers on "define my task", "write a prompt template", "I need a custom task", "multi-step / retrieval / RAG / org-scoped / tenant-scoped task", "expose values to the eval", "re-score a past run", "ReplayTask".
---

# Define the task

The **task** is what you're testing: it takes one dataset record, does whatever the model
needs to do, and returns the answer the evals score. Everything downstream measures how well
models perform *this* task, so pin it down first. There are two ways; most projects only need
the first.

## Default task: a prompt template (no code)

If the task is "fill each record into a prompt and ask the model," write a plain-text template
with `{placeholders}` filled from record fields, and pass `--prompt-template`:

```
You are a helpful security analyst. Answer the question using clear, specific language.

Question: {input}
```

- `{input}` comes from the record's `input` field. **Any** record field can be a placeholder
  (`{org_name}`, `{context}`, …), so multi-variable prompts also need no code — just add the
  fields to your JSONL. List/dict fields render as readable JSON.
- That's the whole task. Recommend this path unless the user clearly needs extra steps.

## Custom task: when reaching the answer takes extra steps

Write a small Python class **only** when getting to the answer needs more than filling one
template — resolving ids, injecting tenant/org scoping, retrieving context, or a multi-call
generate → run → summarize pipeline. A task is any class with `name` and
`run(record, client) -> TaskOutput` (`client` is the model under test):

```python
# my_task.py
from llm_workbench.object_types import GenerationRequest
from llm_workbench.tasks.base import TaskOutput

class MyTask:
    name = "my-task"

    def run(self, record, client):
        prompt = f"Assisting {record['org_name']}.\n\nQuestion: {record['input']}"
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=512))
        return TaskOutput(output=resp.text, prompt=prompt, response=resp,
                          aux={"org_name": record["org_name"]})
```

Point at it with `--task my_task:MyTask` (importable from the working directory; mutually
exclusive with `--prompt-template`).

- **`aux` is how a task surfaces derived values** — retrieved context, generated queries, a
  resolved org — so eval rubric prompts can reference them as `{placeholders}` exactly like
  `{input}` and `{output}`.
- Call `client.generate(...)` as many times as the workflow needs for multi-step pipelines.
- Two runnable templates ship with the tool:
  https://github.com/Nebulock-Inc/llm-workbench/tree/main/examples —
  `my_task.py` (minimal) and `org_scoped_task.py` (scoping + `aux`).

Full contract (everything `run` receives, all `TaskOutput` fields, how `aux` reaches evals,
the optional `review_context()` hook for calibration): the Complex Prompts and Custom Tasks doc
(https://github.com/Nebulock-Inc/llm-workbench/blob/main/docs/complex-prompts-and-custom-tasks.md).

## Built-in `ReplayTask` — score a stored answer

`--task llm_workbench.tasks:ReplayTask` scores an answer already in the record instead of
calling the model under test (the judge still runs). Use it to calibrate judges, regression-test
a rubric or judge-model change, or re-score a past run.

**Gate for the guide:** a `--prompt-template` file exists, or a `--task module:Class` imports
cleanly from the working directory.
