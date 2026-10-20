"""Ollama local models via the /api/generate HTTP endpoint."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import requests

from llm_workbench.models.base import BaseClient
from llm_workbench.object_types import GenerationRequest, Response, Usage

if TYPE_CHECKING:
    from llm_workbench.config import LabConfig, OllamaConfig


class OllamaClient(BaseClient):
    def __init__(self, model_id: str, native: str, cfg: "OllamaConfig") -> None:
        super().__init__(model_id)
        self.native = native
        self.cfg = cfg

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "OllamaClient":
        return cls(model_id, native, config.ollama)

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        body: dict[str, Any] = {"model": self.native, "prompt": request.prompt, "stream": False}
        if request.system:
            body["system"] = request.system
        if request.stop:
            body["stop"] = request.stop

        options: dict[str, Any] = {}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if options:
            body["options"] = options

        if self.cfg.keep_alive is not None:
            body["keep_alive"] = self.cfg.keep_alive
        if self.cfg.think is False:              # only send when explicitly disabled
            body["think"] = False

        timeout = (self.cfg.connect_timeout, self.cfg.read_timeout)
        try:
            resp = requests.post(f"{self.cfg.base_url}/api/generate", json=body, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama call timed out after {timeout}s (model={self.native}); "
                "the model may be cold-loading or the generation stalled."
            ) from exc
        resp.raise_for_status()


        data = resp.json()

        n_in, n_out = data.get("prompt_eval_count"), data.get("eval_count")
        usage = Usage.from_in_out(n_in, n_out)
        return data["response"], usage, data

    def _annotate(self, response: Response) -> Response:
        if not response.error and not response.text and (response.raw or {}).get("done_reason") == "length":
            return replace(response, error="truncated before final answer (raise max_tokens or set OLLAMA_THINK=false)")
        return response

    def availability_warning(self) -> str | None:
        """Ask Ollama what it has pulled (/api/tags) and flag if `native` is missing.

        A missing model otherwise surfaces only as a 404 on every row mid-sweep."""
        try:
            resp = requests.get(f"{self.cfg.base_url}/api/tags", timeout=self.cfg.connect_timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as exc:
            return (f"Could not reach Ollama at {self.cfg.base_url} to check availability "
                    f"({type(exc).__name__}); is it running? Set OLLAMA_BASE_URL to change the host.")
        # Guard against a malformed payload: this hook must never raise, so bail with a
        # warning rather than letting a bad shape blow up the .get() iteration below.
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not all(
            isinstance(m, dict) and isinstance(m.get("name"), str) for m in models
        ):
            return (f"Ollama at {self.cfg.base_url} returned an unexpected /api/tags response; "
                    f"could not check whether {self.native!r} is pulled.")
        names = {m["name"] for m in models}
        # Ollama defaults a tag-less name to ':latest'.
        if self.native in names or (":" not in self.native and f"{self.native}:latest" in names):
            return None
        available = ", ".join(sorted(n for n in names if n)) or "(none)"
        return (f"Model {self.native!r} is not pulled in Ollama at {self.cfg.base_url}; "
                f"every call will 404. Run `ollama pull {self.native}`. Available: {available}")

