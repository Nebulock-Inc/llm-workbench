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


def test_annotate_hook_runs_after_generate():
    class Flag(BaseClient):
        def _raw_generate(self, request):
            return "", Usage(), {"done_reason": "length"}

        def _annotate(self, response):
            return replace(response, error="empty") if not response.text else response

    resp = Flag("mock/flag").generate(GenerationRequest(prompt="x"))
    assert resp.error == "empty"
