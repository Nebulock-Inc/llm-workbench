"""Load user-supplied scorer plugins so custom scorer `type:`s register at startup.

A scorer plugin is any importable module whose import side effect calls
`register_scorer(type_name, factory)` (see scoring/registry.py). Naming the module once
via `--scorer-module` (repeatable) makes its scorer types usable in any `eval.yaml`, exactly
like a built-in scorer — resolved by `build_scorers` for both `run compare` and
`evals validate`.

The current working directory is added to sys.path before importing, mirroring
`tasks.loader.load_task` and `models.plugins`, so a module in the project root is importable
by its bare name.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable


def load_scorer_plugins(modules: Iterable[str], *, add_cwd: bool = True) -> list[str]:
    """Import each module so its `register_scorer(...)` calls run. Returns the names
    imported (blanks ignored). Fails fast with an actionable message if a named module
    cannot be imported — a configured scorer that won't load is a misconfiguration."""
    names = [m.strip() for m in modules if m and m.strip()]
    if not names:
        return []
    if add_cwd:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"Could not import scorer module {name!r}: {exc}. "
                "Pass an importable module (cwd is on sys.path), e.g. --scorer-module my_pkg.scorers."
            ) from exc
    return names
