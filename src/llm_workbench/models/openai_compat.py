"""Any OpenAI-compatible chat endpoint: OpenAI, OpenRouter, Together, vLLM,
LM Studio, Azure, or a local server — selected by swapping base_url/api_key.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from llm_workbench.models.base import BaseClient, importing
from llm_workbench.object_types import GenerationRequest, Response, Usage

if TYPE_CHECKING:
    from llm_workbench.config import LabConfig, OpenAICompatConfig


class OpenAICompatClient(BaseClient):
    def __init__(self, model_id: str, native: str, cfg: "OpenAICompatConfig") -> None:
        super().__init__(model_id)
        self.native = native
        self.cfg = cfg
        self._client_obj: Any | None = None  # lazily built so `models list` needs no SDK

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "OpenAICompatClient":
        return cls(model_id, native, config.openai)

    def _client(self) -> Any:
        if self._client_obj is None:
            with importing("openai", extra="openai", used_for="openai/ and bedrock-mantle/ models"):
                from openai import OpenAI
            # api_key is required by the SDK even for keyless local servers.
            self._client_obj = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key or "not-needed")
        return self._client_obj

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {"model": self.native, "messages": messages}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop"] = request.stop

        resp = self._client().chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = (
            Usage(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens, total_tokens=u.total_tokens)
            if u
            else Usage()
        )
        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
        return text, usage, raw

    def _annotate(self, response: Response) -> Response:
        if not response.error and not response.text:
            ch = ((response.raw or {}).get("choices") or [{}])[0]
            if ch.get("finish_reason") == "length":
                return replace(response, error="truncated before final answer (raise max_tokens or disable thinking)")
        return response
