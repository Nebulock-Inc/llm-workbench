"""Core value types shared across the model, task, and scoring layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """One generation call's inputs, provider-agnostic."""

    prompt: str
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific passthrough


@dataclass(slots=True, frozen=True)
class Usage:
    """Token accounting; any field may be None when a provider doesn't report it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class Response:
    """Result of a generation call, with timing/usage metadata for comparison."""

    text: str
    model_id: str                       # canonical "prefix/native"
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    cost_usd: float | None = None
    raw: dict[str, Any] | None = None    # raw provider payload, for debugging
    error: str | None = None             # set on soft failure instead of raising
