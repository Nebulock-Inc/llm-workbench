"""Provider registry: maps a model-id prefix to a ModelClient class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_lab.models.base import BaseClient

REGISTRY: dict[str, type["BaseClient"]] = {}


def register_provider(prefix: str, cls: type["BaseClient"]) -> None:
    """Register a client class under a model-id prefix (e.g. 'ollama')."""
    REGISTRY[prefix] = cls
