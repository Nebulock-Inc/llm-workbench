"""The ModelClient contract and a BaseClient that adds timing/usage/cost."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from llm_lab.object_types import GenerationRequest, Response, Usage


@runtime_checkable
class ModelClient(Protocol):
    """Anything that turns a prompt into text. `.call` is the convenience shim."""

    model_id: str

    def generate(self, request: GenerationRequest) -> Response: ...
    def call(self, prompt: str) -> str: ...


class BaseClient(ABC):
    """Concrete base: wraps `_raw_generate` with timing and usage.

    Subclasses implement `_raw_generate`, returning the text, a Usage, and the
    raw provider payload. Timing is added here so every
    provider reports latency the same way, and exceptions become Response.error.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def _raw_generate(
        self, request: GenerationRequest
    ) -> tuple[str, Usage, dict[str, Any] | None]: ...

    def generate(self, request: GenerationRequest) -> Response:
        start = time.perf_counter()
        try:
            text, usage, raw = self._raw_generate(request)
            response = Response(
                text=text,
                model_id=self.model_id,
                usage=usage,
                latency_s=time.perf_counter() - start,
                raw=raw,
            )
        except Exception as exc:  # soft-fail: one bad call shouldn't kill the sweep
            response = Response(
                text="",
                model_id=self.model_id,
                latency_s=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._annotate(response)

    def _annotate(self, response: Response) -> Response:
        """Provider hook to flag soft problems (e.g. truncation) after a Response
        is built. Default: unchanged. Overrides must not raise."""
        return response

    def call(self, prompt: str) -> str:
        return self.generate(GenerationRequest(prompt=prompt)).text
