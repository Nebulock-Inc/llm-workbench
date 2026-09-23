"""Typed configuration.

Secrets and endpoints live here (read from the environment, with .env support);
an experiment only ever references models by id string.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    keep_alive: str | None = None
    think: bool = True
    connect_timeout: int = 10
    read_timeout: int = 300


class AWSConfig(BaseModel):
    profile: str | None = None
    region: str = "us-east-1"


class OpenAICompatConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None


class AnthropicConfig(BaseModel):
    api_key: str | None = None


class LabConfig(BaseModel):
    """Root config; per-provider secrets/endpoints plus run-wide defaults."""

    ollama: OllamaConfig = OllamaConfig()
    aws: AWSConfig = AWSConfig()
    openai: OpenAICompatConfig = OpenAICompatConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    default_max_tokens: int = 4096
    provider_modules: list[str] = Field(default_factory=list)  # custom backend plugin modules

    @classmethod
    def from_env(cls) -> "LabConfig":
        """Build config from env vars. A .env at/above cwd is loaded first;
        real environment variables always win over its values."""
        env_file = find_dotenv(usecwd=True)
        if env_file:
            load_dotenv(env_file)
        env = os.environ
        return cls(
            ollama=OllamaConfig(
                base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                keep_alive=env.get("OLLAMA_KEEP_ALIVE") or None,
                think=env.get("OLLAMA_THINK", "true").lower() != "false",
            ),
            aws=AWSConfig(
                profile=env.get("AWS_PROFILE") or None,
                region=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1",
            ),
            openai=OpenAICompatConfig(
                base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=env.get("OPENAI_API_KEY") or None,
            ),
            anthropic=AnthropicConfig(
                api_key=env.get("ANTHROPIC_API_KEY") or None,
            ),
            provider_modules=_split_modules(env.get("LLM_WORKBENCH_PROVIDERS", "")),
        )


def _split_modules(raw: str) -> list[str]:
    """Parse a comma- or whitespace-separated list of module names."""
    return [m.strip() for m in raw.replace(",", " ").split() if m.strip()]
