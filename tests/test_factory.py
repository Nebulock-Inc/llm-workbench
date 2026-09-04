"""Model factory, registry, and BaseClient behavior (timing, soft-fail, _annotate)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm_lab.models.base import BaseClient, ModelClient
from llm_lab.models.factory import make_client
from llm_lab.models.mock import MockClient
from llm_lab.models.registry import REGISTRY
from llm_lab.object_types import GenerationRequest, Usage


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
    from llm_lab.config import LabConfig
    from llm_lab.models.anthropic import AnthropicClient
    from llm_lab.object_types import Response

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
    from llm_lab.config import LabConfig
    from llm_lab.models.bedrock import BedrockClient
    from llm_lab.object_types import Response

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

    from llm_lab.config import LabConfig
    from llm_lab.models.bedrock_mantle import MantleClient
    from llm_lab.object_types import GenerationRequest

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


def test_mantle_annotate_flags_truncation():
    from llm_lab.config import LabConfig
    from llm_lab.models.bedrock_mantle import MantleClient
    from llm_lab.object_types import Response

    client = MantleClient.from_config(model_id="bedrock-mantle/g", native="g", config=LabConfig())
    # Empty output + incomplete-due-to-token-cap -> flagged like the other providers.
    truncated = Response(text="", model_id="x",
                         raw={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})
    assert client._annotate(truncated).error and "truncated" in client._annotate(truncated).error
    # A normal completion is left untouched.
    ok = Response(text="the answer", model_id="x", raw={"status": "completed"})
    assert client._annotate(ok).error is None
    # Missing raw (None) must not crash.
    assert client._annotate(Response(text="", model_id="x")).error is None


def test_annotate_hook_runs_after_generate():
    class Flag(BaseClient):
        def _raw_generate(self, request):
            return "", Usage(), {"done_reason": "length"}

        def _annotate(self, response):
            return replace(response, error="empty") if not response.text else response

    resp = Flag("mock/flag").generate(GenerationRequest(prompt="x"))
    assert resp.error == "empty"
