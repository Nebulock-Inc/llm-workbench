"""Model factory, registry, and BaseClient behavior (timing, soft-fail, _annotate)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm_workbench.models.base import BaseClient, ModelClient
from llm_workbench.models.factory import make_client
from llm_workbench.models.mock import MockClient
from llm_workbench.models.registry import REGISTRY
from llm_workbench.object_types import GenerationRequest, Usage


def test_registry_has_builtin_providers():
    for prefix in ("mock", "ollama", "openai", "bedrock", "bedrock-mantle", "anthropic"):
        assert prefix in REGISTRY


def test_make_client_mock_echo_satisfies_protocol():
    client = make_client("mock/echo")
    assert isinstance(client, MockClient)
    assert isinstance(client, ModelClient)  # runtime_checkable Protocol
    assert client.call("hello") == "hello"


def test_native_splits_on_first_slash_only():
    # "canned:hi/there" contains a slash; only the first slash separates prefix.
    client = make_client("mock/canned:hi/there")
    assert client.call("x") == "hi/there"


def test_missing_slash_is_rejected():
    with pytest.raises(ValueError, match="must be"):
        make_client("llama3.1")


def test_unknown_prefix_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        make_client("nope/whatever")


def test_generate_captures_usage_and_latency():
    resp = make_client("mock/echo").generate(GenerationRequest(prompt="a b c"))
    assert resp.text == "a b c"
    assert resp.usage.total_tokens == 6  # 3 prompt + 3 output words
    assert resp.latency_s >= 0.0
    assert resp.error is None


def test_baseclient_soft_fails_on_exception():
    class Boom(BaseClient):
        def _raw_generate(self, request):
            raise RuntimeError("kaboom")

    resp = Boom("mock/boom").generate(GenerationRequest(prompt="x"))
    assert resp.text == ""
    assert resp.error and "kaboom" in resp.error


def test_anthropic_flags_max_tokens_truncation():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.anthropic import AnthropicClient
    from llm_workbench.object_types import Response

    client = AnthropicClient.from_config(model_id="anthropic/claude", native="claude", config=LabConfig())
    # empty output that hit the token cap -> flagged (like ollama/openai)
    truncated = Response(text="", model_id="anthropic/claude", raw={"stop_reason": "max_tokens"})
    assert "truncated" in (client._annotate(truncated).error or "")
    # a partial-but-usable answer at the cap stays usable (not marked a failure)
    partial = Response(text="a partial answer", model_id="anthropic/claude", raw={"stop_reason": "max_tokens"})
    assert client._annotate(partial).error is None
    # normal completion is untouched
    normal = Response(text="done", model_id="anthropic/claude", raw={"stop_reason": "end_turn"})
    assert client._annotate(normal).error is None


def test_bedrock_flags_truncation_across_apis():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock import BedrockClient
    from llm_workbench.object_types import Response

    client = BedrockClient.from_config(model_id="bedrock/anthropic.claude-x", native="anthropic.claude-x",
                                       config=LabConfig())
    # invoke_model (anthropic) reports stop_reason; Converse (meta/mistral) reports stopReason
    for raw in ({"stop_reason": "max_tokens"}, {"stopReason": "max_tokens"}, {"stop_reason": "length"}):
        assert "truncated" in (client._annotate(Response(text="", model_id="x", raw=raw)).error or "")
    # a partial answer that hit the cap stays usable; a normal stop is untouched
    assert client._annotate(Response(text="partial", model_id="x", raw={"stopReason": "max_tokens"})).error is None
    assert client._annotate(Response(text="", model_id="x", raw={"stopReason": "end_turn"})).error is None


def test_mantle_forwards_system_and_temperature_but_not_stop():
    import pytest

    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock_mantle import MantleClient
    from llm_workbench.object_types import GenerationRequest

    client = MantleClient.from_config(model_id="bedrock-mantle/g", native="g", config=LabConfig())
    # `stop` is unsupported by the Responses API — it must be dropped AND warned about.
    with pytest.warns(UserWarning, match="cannot honor `stop`"):
        kw = client._request_kwargs(GenerationRequest(prompt="hi", system="be terse", temperature=0.2,
                                                      max_tokens=50, stop=["END"]))
    assert kw["instructions"] == "be terse"      # system -> instructions
    assert kw["temperature"] == 0.2
    assert kw["max_output_tokens"] == 50
    assert "stop" not in kw                       # Responses API has no stop parameter

    kw2 = client._request_kwargs(GenerationRequest(prompt="hi"))
    assert "instructions" not in kw2 and "temperature" not in kw2
    assert kw2["max_output_tokens"] == client.max_tokens


def test_mantle_chat_kwargs_forwards_system_temperature_and_stop():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock_mantle import MantleClient
    from llm_workbench.object_types import GenerationRequest

    client = MantleClient.from_config(model_id="bedrock-mantle/g", native="g", config=LabConfig())
    # The Chat Completions surface DOES support `stop` (unlike Responses): no warning, forwarded.
    kw = client._chat_kwargs(GenerationRequest(prompt="hi", system="be terse", temperature=0.2,
                                               max_tokens=50, stop=["END"]))
    assert kw["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    assert kw["temperature"] == 0.2
    assert kw["max_tokens"] == 50
    assert kw["stop"] == ["END"]

    kw2 = client._chat_kwargs(GenerationRequest(prompt="hi"))
    assert kw2["messages"] == [{"role": "user", "content": "hi"}]  # no system message
    assert "temperature" not in kw2 and "stop" not in kw2
    assert kw2["max_tokens"] == client.max_tokens


def test_mantle_annotate_flags_truncation():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock_mantle import MantleClient
    from llm_workbench.object_types import Response

    client = MantleClient.from_config(model_id="bedrock-mantle/g", native="g", config=LabConfig())
    # Empty output + incomplete-due-to-token-cap -> flagged like the other providers.
    truncated = Response(text="", model_id="x",
                         raw={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})
    assert client._annotate(truncated).error and "truncated" in client._annotate(truncated).error
    # The Chat Completions surface reports the same cap via finish_reason == "length".
    chat_truncated = Response(text="", model_id="x", raw={"choices": [{"finish_reason": "length"}]})
    assert "truncated" in (client._annotate(chat_truncated).error or "")
    # A normal completion is left untouched (either surface).
    ok = Response(text="the answer", model_id="x", raw={"status": "completed"})
    assert client._annotate(ok).error is None
    chat_ok = Response(text="the answer", model_id="x", raw={"choices": [{"finish_reason": "stop"}]})
    assert client._annotate(chat_ok).error is None
    # Missing raw (None) must not crash.
    assert client._annotate(Response(text="", model_id="x")).error is None


def test_mantle_endpoint_mismatch_detection():
    from llm_workbench.models.bedrock_mantle import _is_endpoint_mismatch

    # Both observed "wrong endpoint" rejections advance the probe...
    assert _is_endpoint_mismatch(Exception(
        "Error code: 400 - The model 'zai.glm-4.7' does not support the '/openai/v1/responses' API"))
    assert _is_endpoint_mismatch(Exception(
        "Error code: 400 - model `zai.glm-4.7` isn't supported on this route"))
    # ...but a genuine failure (unsupported param, timeout) must NOT be swallowed.
    assert not _is_endpoint_mismatch(Exception("model does not support temperature"))
    assert not _is_endpoint_mismatch(Exception("some unrelated timeout"))


def test_mantle_probes_endpoints_until_one_answers_then_caches():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock_mantle import _ENDPOINT_BY_MODEL, MantleClient
    from llm_workbench.object_types import GenerationRequest, Usage

    _ENDPOINT_BY_MODEL.pop("zai.glm-4.7", None)  # start from an undetected model
    client = MantleClient.from_config(model_id="bedrock-mantle/zai.glm-4.7", native="zai.glm-4.7",
                                      config=LabConfig())
    tried: list[tuple[str, str]] = []

    # zai.glm-4.7: rejected on both /openai/v1 endpoints, answers on bare /v1 chat.
    def fake_generate_via(endpoint, _request):
        tried.append(endpoint)
        base, api = endpoint
        if base == "/openai/v1" and api == "responses":
            raise Exception("does not support the '/openai/v1/responses' API")
        if base == "/openai/v1" and api == "chat":
            raise Exception("model `zai.glm-4.7` isn't supported on this route")
        return "glm answer", Usage(), {"choices": [{"finish_reason": "stop"}]}

    client._generate_via = fake_generate_via  # type: ignore[assignment]

    text, _usage, _raw = client._raw_generate(GenerationRequest(prompt="hi"))
    assert text == "glm answer"
    assert tried == [("/openai/v1", "responses"), ("/openai/v1", "chat"), ("/v1", "chat")]
    assert _ENDPOINT_BY_MODEL["zai.glm-4.7"] == ("/v1", "chat")  # winning endpoint cached

    # Second call for the same model goes straight to the cached endpoint.
    tried.clear()
    client._raw_generate(GenerationRequest(prompt="again"))
    assert tried == [("/v1", "chat")]


def test_mantle_real_400_is_not_swallowed_by_the_probe():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.bedrock_mantle import _ENDPOINT_BY_MODEL, MantleClient
    from llm_workbench.object_types import GenerationRequest

    _ENDPOINT_BY_MODEL.pop("some.model", None)
    client = MantleClient.from_config(model_id="bedrock-mantle/some.model", native="some.model",
                                      config=LabConfig())

    def real_error(_endpoint, _request):
        raise ValueError("context length exceeded")  # not an endpoint mismatch

    client._generate_via = real_error  # type: ignore[assignment]
    with pytest.raises(ValueError, match="context length exceeded"):
        client._raw_generate(GenerationRequest(prompt="hi"))
    assert "some.model" not in _ENDPOINT_BY_MODEL  # nothing cached on a real failure


def test_annotate_hook_runs_after_generate():
    class Flag(BaseClient):
        def _raw_generate(self, request):
            return "", Usage(), {"done_reason": "length"}

        def _annotate(self, response):
            return replace(response, error="empty") if not response.text else response

    resp = Flag("mock/flag").generate(GenerationRequest(prompt="x"))
    assert resp.error == "empty"


class _FakeTagsResponse:
    """Stand-in for requests.Response from Ollama's /api/tags."""

    def __init__(self, names):
        self._names = names

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def _ollama_client():
    from llm_workbench.config import LabConfig
    from llm_workbench.models.ollama import OllamaClient
    return OllamaClient.from_config(model_id="ollama/mistral:latest", native="mistral:latest",
                                    config=LabConfig())


def test_ollama_warns_when_model_not_pulled(monkeypatch):
    import llm_workbench.models.ollama as ollama_mod

    monkeypatch.setattr(ollama_mod.requests, "get",
                        lambda *a, **k: _FakeTagsResponse(["llama3.1:latest", "qwen3:latest"]))
    warning = _ollama_client().availability_warning()
    assert warning is not None
    assert "not pulled" in warning
    assert "ollama pull mistral:latest" in warning
    assert "llama3.1:latest" in warning  # lists what IS available


def test_ollama_quiet_when_model_present(monkeypatch):
    import llm_workbench.models.ollama as ollama_mod

    monkeypatch.setattr(ollama_mod.requests, "get",
                        lambda *a, **k: _FakeTagsResponse(["mistral:latest", "qwen3:latest"]))
    assert _ollama_client().availability_warning() is None


def test_ollama_defaults_tagless_name_to_latest(monkeypatch):
    import llm_workbench.models.ollama as ollama_mod
    from llm_workbench.config import LabConfig
    from llm_workbench.models.ollama import OllamaClient

    monkeypatch.setattr(ollama_mod.requests, "get",
                        lambda *a, **k: _FakeTagsResponse(["mistral:latest"]))
    client = OllamaClient.from_config(model_id="ollama/mistral", native="mistral", config=LabConfig())
    assert client.availability_warning() is None  # 'mistral' matches 'mistral:latest'


def test_ollama_warns_when_unreachable(monkeypatch):
    import llm_workbench.models.ollama as ollama_mod

    def boom(*a, **k):
        raise ollama_mod.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_mod.requests, "get", boom)
    warning = _ollama_client().availability_warning()
    assert warning is not None
    assert "Could not reach Ollama" in warning
