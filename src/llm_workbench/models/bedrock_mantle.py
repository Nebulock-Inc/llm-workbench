"""Amazon Bedrock Mantle — the OpenAI-compatible endpoint for non-Anthropic
Bedrock models (e.g. google.gemma-4-31b, zai.glm-4.7). Driven with the openai SDK,
SigV4-signed with the AWS profile at the httpx layer (no Bedrock API key needed).

Mantle serves models on different combinations of base path and API surface, and
exposes no discovery API:
  - base path: `/openai/v1` (gemma) OR the bare `/v1` (some models, e.g. zai.glm-4.7)
  - API surface: the Responses API OR Chat Completions
Not every model supports every combination (gemma speaks Responses under /openai/v1;
zai.glm-* only speaks Chat Completions under the bare /v1), and the split isn't
derivable from the model id. Rather than hardcode a table that rots as models are
added, this client probes candidate endpoints in preference order on the first call,
advancing only on the specific "wrong route / unsupported API" rejections, and
remembers the winning endpoint per model for the rest of the process.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import boto3

from llm_workbench.models.base import BaseClient, importing
from llm_workbench.object_types import GenerationRequest, Response, Usage

if TYPE_CHECKING:
    from llm_workbench.config import AWSConfig, LabConfig


# Candidate endpoints to probe, as (base-path suffix, API surface), in preference
# order. Mantle has no discovery API, so we try these until one answers and cache
# the winner. Preference: the richer /openai/v1 surfaces first, bare /v1 after.
_CANDIDATE_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("/openai/v1", "responses"),  # e.g. google.gemma-4-31b
    ("/openai/v1", "chat"),       # OpenAI-compat chat models on the /openai route
    ("/v1", "chat"),              # bare-route chat models, e.g. zai.glm-4.7
    ("/v1", "responses"),         # bare-route responses models
)

# The endpoint each native model was found to speak, learned on the first call and
# remembered for the rest of the process so later records — and any freshly built
# client for the same model — skip the probe.
_ENDPOINT_BY_MODEL: dict[str, tuple[str, str]] = {}


def _is_endpoint_mismatch(exc: Exception) -> bool:
    """True if `exc` is Mantle saying "wrong endpoint for this model" — i.e. we
    should try the next candidate rather than surface it as the real failure.

    Covers the two observed rejections (matched loosely, case-insensitive, so a
    vendor reword doesn't silently disable the probe):
      - Responses on an unsupported model: "... does not support the '/openai/v1/responses' API"
      - Chat on the wrong base path:       "model `x` isn't supported on this route"
    A genuine 400 (e.g. an unsupported parameter) mentions neither "route" nor an
    unsupported "API", so it correctly propagates instead of masking itself.
    """
    msg = str(exc).lower()
    if "supported on this route" in msg:
        return True
    if "does not support" in msg and "api" in msg:
        return True
    return False


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
        self._clients: dict[str, Any] = {}  # one OpenAI client per base-path suffix

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "MantleClient":
        return cls(model_id, native, config.aws, config.default_max_tokens)

    def _client(self, base_suffix: str) -> Any:
        """OpenAI SDK client for a given Mantle base path (e.g. '/openai/v1' or '/v1').

        Built lazily and cached per suffix so a model that settles on one endpoint
        reuses its client, and `models list` (which never calls) needs no SDK.
        """
        if base_suffix not in self._clients:
            with importing("openai", extra="openai", used_for="bedrock-mantle/ models"):
                import httpx
                from openai import OpenAI
            base_url = f"https://bedrock-mantle.{self.region}.api.aws{base_suffix}"
            http_client = httpx.Client(
                timeout=180.0,  # first call can cold-start well past the SDK default
                event_hooks={"request": [_sigv4_signer(self.profile, self.region)]},
            )
            # api_key is required by the SDK but unused — the signer overwrites Authorization.
            self._clients[base_suffix] = OpenAI(base_url=base_url, api_key="sigv4", http_client=http_client)
        return self._clients[base_suffix]

    def _raw_generate(self, request: GenerationRequest) -> tuple[str, Usage, dict[str, Any] | None]:
        known = _ENDPOINT_BY_MODEL.get(self.native)
        if known is not None:
            return self._generate_via(known, request)
        # Unknown model: probe candidates in order, advancing only on the specific
        # "wrong endpoint" rejections. Any other error is the real failure and stops us.
        last_mismatch: Exception | None = None
        for endpoint in _CANDIDATE_ENDPOINTS:
            try:
                result = self._generate_via(endpoint, request)
            except Exception as exc:
                if not _is_endpoint_mismatch(exc):
                    raise
                last_mismatch = exc
                continue
            _ENDPOINT_BY_MODEL[self.native] = endpoint
            return result
        raise RuntimeError(
            f"bedrock-mantle: no supported endpoint found for {self.native!r} "
            f"(tried {list(_CANDIDATE_ENDPOINTS)}); check the model card. "
            f"Last rejection: {last_mismatch}"
        ) from last_mismatch

    def _generate_via(
        self, endpoint: tuple[str, str], request: GenerationRequest
    ) -> tuple[str, Usage, dict[str, Any] | None]:
        base_suffix, api = endpoint
        client = self._client(base_suffix)
        if api == "responses":
            return self._generate_responses(client, request)
        return self._generate_chat(client, request)

    # --- Responses API -----------------------------------------------------

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

    def _generate_responses(
        self, client: Any, request: GenerationRequest
    ) -> tuple[str, Usage, dict[str, Any] | None]:
        resp = client.responses.create(**self._request_kwargs(request))
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

    # --- Chat Completions API ---------------------------------------------

    def _chat_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Map a GenerationRequest onto the Chat Completions parameters.

        `system` becomes a system-role message and — unlike the Responses API —
        `stop` is a first-class parameter here, so it is forwarded.
        """
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": self.native,
            "messages": messages,
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop"] = request.stop
        return kwargs

    def _generate_chat(
        self, client: Any, request: GenerationRequest
    ) -> tuple[str, Usage, dict[str, Any] | None]:
        resp = client.chat.completions.create(**self._chat_kwargs(request))
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = (
            Usage(
                input_tokens=getattr(u, "prompt_tokens", None),
                output_tokens=getattr(u, "completion_tokens", None),
                total_tokens=getattr(u, "total_tokens", None),
            )
            if u
            else Usage()
        )
        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
        return text, usage, raw

    def _annotate(self, response: Response) -> Response:
        # Flag an empty answer that hit the token cap. The two Mantle surfaces report
        # it differently: Responses via status="incomplete" + incomplete_details.reason,
        # Chat Completions via choices[].finish_reason == "length".
        if response.error or response.text:
            return response
        raw = response.raw or {}
        reason = (raw.get("incomplete_details") or {}).get("reason")
        if raw.get("status") == "incomplete" and reason in ("max_output_tokens", "max_tokens"):
            return replace(response, error="truncated before final answer (raise max_tokens)")
        choices = raw.get("choices") or []
        if choices and choices[0].get("finish_reason") == "length":
            return replace(response, error="truncated before final answer (raise max_tokens)")
        return response
