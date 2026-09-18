"""Small display helpers shared by the analysis report, plots, and power CLI.

Dependency-free (no pandas/scipy), so it is safe to import without the `analysis` extra.
"""

from __future__ import annotations


def short_model_id(model_id: str) -> str:
    """The display form of a model id: drop the `provider/` prefix (e.g. ollama/llama3.1 -> llama3.1)."""
    return model_id.split("/")[-1]
