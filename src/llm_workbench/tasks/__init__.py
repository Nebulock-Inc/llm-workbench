"""Task contract and built-in tasks.

`Task` is the protocol (`run(record, client) -> TaskOutput`). Built-ins:
- `PromptTemplateTask` — fill one template per record and call the model (the default task).
- `ReplayTask` — score an answer already in the record, skipping the model under test
  (judge calibration, rubric/judge regression, re-scoring a past run).
"""

from llm_workbench.tasks.base import (
    PromptTemplateTask,
    ReplayTask,
    Task,
    TaskOutput,
)

__all__ = ["Task", "TaskOutput", "PromptTemplateTask", "ReplayTask"]
