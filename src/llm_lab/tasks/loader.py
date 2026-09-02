"""Resolve a Task from an import-path string ('module:attribute') or a prompt file."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from llm_lab.tasks.base import PromptTemplateTask, Task


def load_task(spec: str, **kwargs: Any) -> Task:
    """Load a Task from 'module.path:attribute'.

    The attribute may be a Task instance, a Task class (instantiated with kwargs),
    or a factory callable returning a Task. The current working directory is added
    to sys.path so a user's local task module (e.g. 'my_task:MyTask') is importable.
    """
    if ":" not in spec:
        raise ValueError(f"Task spec must be 'module:attribute', got {spec!r}.")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_name, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(module_name), attr)
    if isinstance(obj, type):                      # a class -> instantiate
        return obj(**kwargs) if kwargs else obj()
    if callable(obj) and not hasattr(obj, "run"):  # a factory -> call
        return obj(**kwargs) if kwargs else obj()
    return obj                                     # already a Task instance


def task_from_prompt_file(path: str | Path, **kwargs: Any) -> PromptTemplateTask:
    """Build the default PromptTemplateTask from a template file."""
    return PromptTemplateTask(Path(path).read_text(), **kwargs)
