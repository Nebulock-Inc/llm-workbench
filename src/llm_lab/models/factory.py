"""Turn a model-id string like 'ollama/llama3.1' into a ready ModelClient."""

from __future__ import annotations

from llm_lab.config import LabConfig
from llm_lab.models.base import ModelClient
from llm_lab.models.registry import REGISTRY, register_provider

# Register built-in providers. Importing a provider module must NOT import its
# heavy SDK (openai/anthropic) at module top — those are lazy-imported inside
# each client — so this stays cheap even when an extra isn't installed.
from llm_lab.models.mock import MockClient
from llm_lab.models.ollama import OllamaClient
from llm_lab.models.openai_compat import OpenAICompatClient
from llm_lab.models.bedrock import BedrockClient
from llm_lab.models.bedrock_mantle import MantleClient
from llm_lab.models.anthropic import AnthropicClient

register_provider("mock", MockClient)
register_provider("ollama", OllamaClient)
register_provider("openai", OpenAICompatClient)
register_provider("bedrock", BedrockClient)
register_provider("bedrock-mantle", MantleClient)
register_provider("anthropic", AnthropicClient)


def make_client(model_id: str, config: LabConfig | None = None) -> ModelClient:
    """Instantiate the client for `model_id` ('<prefix>/<native>')."""
    config = config or LabConfig.from_env()
    prefix, sep, native = model_id.partition("/")
    if not sep:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Model id must be '<provider>/<model>', got {model_id!r}. Known providers: {known}")
    cls = REGISTRY.get(prefix)
    if cls is None:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown provider prefix {prefix!r} in {model_id!r}. Known providers: {known}")
    return cls.from_config(model_id=model_id, native=native, config=config)
