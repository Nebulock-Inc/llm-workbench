"""Native Anthropic API (the `anthropic` SDK), for users with an Anthropic API key
rather than Bedrock access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_lab.models.base import BaseClient
from llm_lab.object_types import GenerationRequest, Usage

if TYPE_CHECKING:
    from llm_lab.config import AnthropicConfig, LabConfig


class AnthropicClient(BaseClient):
    def __init__(self, model_id: str, native: str, cfg: "AnthropicConfig", default_max_tokens: int) -> None:
        super().__init__(model_id)
        self.model = native
        self.cfg = cfg
        self.max_tokens = default_max_tokens
        self._client_obj: Any | None = None

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "AnthropicClient":
        return cls(model_id, native, config.anthropic, config.default_max_tokens)

    def _client(self) -> Any:
        if self._client_obj is None:
            try:
                from anthropic import Anthropic
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "The 'anthropic' package is required for anthropic/ models. "
                    "Install it with: uv sync --extra anthropic"
                ) from exc
            # api_key=None lets the SDK fall back to ANTHROPIC_API_KEY in the env.
            self._client_obj = Anthropic(api_key=self.cfg.api_key)
        return self._client_obj

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        msg = self._client().messages.create(**kwargs)
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        usage = Usage(
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            total_tokens=msg.usage.input_tokens + msg.usage.output_tokens,
        )
        raw = msg.model_dump() if hasattr(msg, "model_dump") else None
        return text, usage, raw
