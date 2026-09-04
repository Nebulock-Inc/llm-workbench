"""AWS Bedrock via boto3. Anthropic uses the Messages API (invoke_model); Meta Llama
and Mistral use the Converse API, which normalizes system prompts, stop sequences, and
the stop reason across models (and handles each model's chat templating internally).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import boto3

from llm_lab.models.base import BaseClient
from llm_lab.object_types import GenerationRequest, Response, Usage

if TYPE_CHECKING:
    from llm_lab.config import AWSConfig, LabConfig


class BedrockClient(BaseClient):
    def __init__(self, model_id: str, native: str, aws: "AWSConfig", default_max_tokens: int) -> None:
        super().__init__(model_id)
        self.native = native
        self.max_tokens = default_max_tokens
        self.kind = self._provider_kind(native)
        session = boto3.Session(profile_name=aws.profile) if aws.profile else boto3.Session()
        self.client = session.client("bedrock-runtime", region_name=aws.region)

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "BedrockClient":
        return cls(model_id, native, config.aws, config.default_max_tokens)

    @staticmethod
    def _provider_kind(native: str) -> str:
        n = native.lower()
        if "anthropic" in n or "claude" in n:
            return "anthropic"
        if "meta" in n or "llama" in n:
            return "meta"
        if "mistral" in n:
            return "mistral"
        raise ValueError(
            f"Unsupported Bedrock model family: {native!r}. "
            "Supported families: anthropic/claude, meta/llama, mistral."
        )

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        if self.kind == "anthropic":
            return self._anthropic(request)
        return self._converse(request)  # meta + mistral

    def _anthropic(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            body["system"] = request.system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop:
            body["stop_sequences"] = request.stop

        params: dict[str, Any] = {"modelId": self.native, "body": json.dumps(body)}
        try:
            resp = self.client.invoke_model(**params)
        except Exception as exc:  # some models require an explicit trace setting
            if "data retention" in str(exc):
                params["trace"] = "DISABLED"
                resp = self.client.invoke_model(**params)
            else:
                raise
        data = json.loads(resp["body"].read())
        text = data["content"][0]["text"] if data.get("content") else ""
        u = data.get("usage", {})
        usage = Usage.from_in_out(u.get("input_tokens"), u.get("output_tokens"))
        return text, usage, data

    def _converse(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        """Meta Llama and Mistral via the Converse API — one shape for system, stop, and
        stop reason. Converse applies each model's chat template, so we pass the raw text."""
        inference: dict[str, Any] = {"maxTokens": request.max_tokens or self.max_tokens}
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.stop:
            inference["stopSequences"] = request.stop
        kwargs: dict[str, Any] = {
            "modelId": self.native,
            "messages": [{"role": "user", "content": [{"text": request.prompt}]}],
            "inferenceConfig": inference,
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]

        resp = self.client.converse(**kwargs)
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = blocks[0].get("text", "") if blocks else ""
        u = resp.get("usage", {})
        usage = Usage(
            input_tokens=u.get("inputTokens"),
            output_tokens=u.get("outputTokens"),
            total_tokens=u.get("totalTokens"),
        )
        return text, usage, resp  # keep the converse dict so _annotate can read stopReason

    def _annotate(self, response: Response) -> Response:
        # Flag output cut off at the token cap. invoke_model (anthropic) reports
        # `stop_reason`, Converse (meta/mistral) reports `stopReason`; both use "max_tokens".
        if response.error or response.text:
            return response
        raw = response.raw or {}
        if (raw.get("stop_reason") or raw.get("stopReason")) in ("max_tokens", "length"):
            return replace(response, error="truncated before final answer (raise max_tokens)")
        return response
