"""Discover evals on disk."""

from __future__ import annotations

from pathlib import Path

from llm_workbench.evals.spec import EvalSpec, load_spec


def discover(eval_dir: str | Path, glob: str = "*") -> list[EvalSpec]:
    """Load every eval under `eval_dir` (each a subdir containing eval.yaml).

    `glob` filters subdirectory names (e.g. 'query-*'); default loads all.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
    specs: list[EvalSpec] = []
    for sub in sorted(eval_dir.glob(glob)):
        if sub.is_dir() and (sub / "eval.yaml").is_file():
            specs.append(load_spec(sub / "eval.yaml"))
    return specs
