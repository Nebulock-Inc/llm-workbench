"""The ModelClient contract and a BaseClient that adds timing/usage."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from llm_workbench.object_types import GenerationRequest, Response, Usage


@contextmanager
def importing(package: str, *, extra: str, used_for: str) -> Iterator[None]:
    """Wrap a lazy provider-SDK import, turning a missing package into an actionable error.

    Usage:
        with importing("openai", extra="openai", used_for="openai/ and bedrock-mantle/ models"):
            from openai import OpenAI
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"The '{package}' package is required for {used_for}. "
            f"Install it with: uv sync --extra {extra}"
        ) from exc


@runtime_checkable
class ModelClient(Protocol):
    """Anything that turns a prompt into text. `.call` is the convenience shim."""

    model_id: str

    def generate(self, request: GenerationRequest) -> Response: ...
    def call(self, prompt: str) -> str: ...
    def availability_warning(self) -> str | None: ...


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

    def availability_warning(self) -> str | None:
        """Optional pre-flight check, run once per client before the sweep starts.

        Return a human-readable warning if the model looks unavailable (e.g. not
        pulled, endpoint unreachable), else None. Providers that can cheaply list
        what's available override this; the default assumes availability. Must not
        raise — a check that can't run should return None (or a warning), never blow up."""
        return None

    def call(self, prompt: str) -> str:
        return self.generate(GenerationRequest(prompt=prompt)).text
