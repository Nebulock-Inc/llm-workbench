"""AWS Bedrock via boto3. One client, three request/response shapes selected by
model family: Anthropic (Messages), Meta Llama (invoke_model), Mistral (Converse).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import boto3

from llm_lab.models.base import BaseClient
from llm_lab.object_types import GenerationRequest, Usage

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
        if self.kind == "meta":
            return self._meta(request)
        return self._mistral(request)

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
        usage = Usage(
            input_tokens=u.get("input_tokens"),
            output_tokens=u.get("output_tokens"),
            total_tokens=((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)) or None,
        )
        return text, usage, data

    def _meta(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        body: dict[str, Any] = {"prompt": prompt, "max_gen_len": request.max_tokens or self.max_tokens}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        resp = self.client.invoke_model(modelId=self.native, body=json.dumps(body))
        data = json.loads(resp["body"].read())
        n_in, n_out = data.get("prompt_token_count"), data.get("generation_token_count")
        usage = Usage(input_tokens=n_in, output_tokens=n_out, total_tokens=((n_in or 0) + (n_out or 0)) or None)
        return data.get("generation", ""), usage, data

    def _mistral(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        inference: dict[str, Any] = {"maxTokens": request.max_tokens or self.max_tokens}
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.stop:
            inference["stopSequences"] = request.stop
        resp = self.client.converse(
            modelId=self.native,
            messages=[{"role": "user", "content": [{"text": request.prompt}]}],
            inferenceConfig=inference,
        )
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = blocks[0].get("text", "") if blocks else ""
        u = resp.get("usage", {})
        usage = Usage(
            input_tokens=u.get("inputTokens"),
            output_tokens=u.get("outputTokens"),
            total_tokens=u.get("totalTokens"),
        )
        return text, usage, None  # converse response isn't JSON-serializable wholesale
