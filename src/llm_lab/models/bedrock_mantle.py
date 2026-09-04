"""Amazon Bedrock Mantle — the OpenAI-compatible Responses API for non-Anthropic
Bedrock models (e.g. google.gemma-4-31b). Driven with the openai SDK, SigV4-signed
with the AWS profile at the httpx layer (no Bedrock API key needed).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import boto3

from llm_lab.models.base import BaseClient, importing
from llm_lab.object_types import GenerationRequest, Response, Usage

if TYPE_CHECKING:
    from llm_lab.config import AWSConfig, LabConfig


def _sigv4_signer(profile: str | None, region: str):
    """Return an httpx request event-hook that SigV4-signs each request for Mantle.

    The openai SDK only sends a Bearer token; we overwrite it with SigV4 headers
    (service 'bedrock-mantle') so the same AWS profile the rest of the harness uses
    authenticates the call.
    """
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    credentials = session.get_credentials()

    def sign(request) -> None:  # request: httpx.Request
        body = request.content or b""
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        SigV4Auth(credentials.get_frozen_credentials(), "bedrock-mantle", region).add_auth(aws_req)
        for key, value in aws_req.headers.items():
            request.headers[key] = value

    return sign


class MantleClient(BaseClient):
    def __init__(self, model_id: str, native: str, aws: "AWSConfig", default_max_tokens: int) -> None:
        super().__init__(model_id)
        self.native = native
        self.region = aws.region
        self.profile = aws.profile
        self.max_tokens = default_max_tokens
        self._client_obj: Any | None = None

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "MantleClient":
        return cls(model_id, native, config.aws, config.default_max_tokens)

    def _client(self) -> Any:
        if self._client_obj is None:
            with importing("openai", extra="openai", used_for="bedrock-mantle/ models"):
                import httpx
                from openai import OpenAI
            # Mantle serves these models under /openai/v1 (NOT the bare /v1 route).
            base_url = f"https://bedrock-mantle.{self.region}.api.aws/openai/v1"
            http_client = httpx.Client(
                timeout=180.0,  # first call can cold-start well past the SDK default
                event_hooks={"request": [_sigv4_signer(self.profile, self.region)]},
            )
            # api_key is required by the SDK but unused — the signer overwrites Authorization.
            self._client_obj = OpenAI(base_url=base_url, api_key="sigv4", http_client=http_client)
        return self._client_obj

    def _request_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Map a GenerationRequest onto the OpenAI Responses API parameters.

        `system` maps to `instructions` (the Responses API's system-prompt slot) and
        `temperature` is passed through. The Responses API has no stop-sequence parameter,
        so `request.stop` cannot be honored here (unlike the chat-completions/Converse
        providers) — it is intentionally not forwarded.
        """
        kwargs: dict[str, Any] = {
            "model": self.native,
            "input": [{"role": "user", "content": request.prompt}],
            "max_output_tokens": request.max_tokens or self.max_tokens,
        }
        if request.system:
            kwargs["instructions"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            warnings.warn(
                f"bedrock-mantle ({self.native}) cannot honor `stop`: the OpenAI-Responses "
                "API has no stop-sequence parameter, so it is ignored.",
                stacklevel=2,
            )
        return kwargs

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        resp = self._client().responses.create(**self._request_kwargs(request))
        text = resp.output_text or ""
        u = getattr(resp, "usage", None)
        usage = (
            Usage(
                input_tokens=getattr(u, "input_tokens", None),
                output_tokens=getattr(u, "output_tokens", None),
                total_tokens=getattr(u, "total_tokens", None),
            )
            if u
            else Usage()
        )
        # Keep the provider payload like the other clients (for debugging + _annotate).
        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
        return text, usage, raw

    def _annotate(self, response: Response) -> Response:
        # The Responses API reports a hit token cap via status="incomplete" +
        # incomplete_details.reason="max_output_tokens".
        if response.error or response.text:
            return response
        raw = response.raw or {}
        reason = (raw.get("incomplete_details") or {}).get("reason")
        if raw.get("status") == "incomplete" and reason in ("max_output_tokens", "max_tokens"):
            return replace(response, error="truncated before final answer (raise max_tokens)")
        return response
