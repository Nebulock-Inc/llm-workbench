"""Load user-supplied provider plugins so custom backends register at startup.

A provider plugin is any importable module whose import side effect is a
`register_provider(prefix, ClientClass)` call (see models/registry.py). Naming
the module once in `LLM_WORKBENCH_PROVIDERS` (or via `--provider-module`) makes
its prefix usable by every command, exactly like a built-in backend — after
that, `prefix/any-model` resolves through the same factory as `ollama/...`.

The current working directory is added to sys.path before importing, mirroring
`tasks.loader.load_task`, so a module dropped in the project root (next to your
dataset and evals) is importable by its bare name.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from collections.abc import Iterable

from llm_workbench.models.registry import REGISTRY


def load_provider_plugins(modules: Iterable[str], *, add_cwd: bool = True) -> list[str]:
    """Import each module so its `register_provider(...)` calls run.

    Returns the module names imported (blanks ignored). Raises ImportError with
    an actionable message if a named module cannot be imported — a configured
    provider that will not load is a misconfiguration, so we fail fast rather
    than silently drop it. Warns, but does not fail, when a plugin overrides a
    prefix that was already registered (e.g. a built-in): deliberate replacement
    is allowed.
    """
    names = [m.strip() for m in modules if m and m.strip()]
    if not names:
        return []

    if add_cwd:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

    before = dict(REGISTRY)  # prefix -> class, snapshot to detect overrides
    loaded: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"Could not import provider module {name!r} "
                f"(from LLM_WORKBENCH_PROVIDERS / --provider-module): {exc}. "
                "Make sure it is importable from the current directory or installed."
            ) from exc
        loaded.append(name)

    for prefix, cls in REGISTRY.items():
        if prefix in before and before[prefix] is not cls:
            warnings.warn(
                f"Provider plugin overrides the built-in {prefix!r} backend "
                f"with {cls.__module__}.{cls.__qualname__}.",
                stacklevel=2,
            )
    return loaded
