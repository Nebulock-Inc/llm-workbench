"""Provider plugin loading: cwd import, registration, fail-fast, override warning."""

from __future__ import annotations

import textwrap

import pytest

from llm_workbench.config import LabConfig, _split_modules
from llm_workbench.models.factory import make_client
from llm_workbench.models.plugins import load_provider_plugins
from llm_workbench.models.registry import REGISTRY

_ACME_PLUGIN = '''
from llm_workbench.models.base import BaseClient
from llm_workbench.models.registry import register_provider
from llm_workbench.object_types import Usage


class AcmeClient(BaseClient):
    def __init__(self, model_id, native):
        super().__init__(model_id)
        self.native = native

    def _raw_generate(self, request):
        return f"acme:{self.native}", Usage(input_tokens=1, output_tokens=1, total_tokens=2), None

    @classmethod
    def from_config(cls, *, model_id, native, config):
        return cls(model_id, native)


register_provider("acme", AcmeClient)
'''

# Registers a *different* class under an existing built-in prefix to trip the override warning.
_OVERRIDE_PLUGIN = '''
from llm_workbench.models.base import BaseClient
from llm_workbench.models.registry import register_provider
from llm_workbench.object_types import Usage


class FakeMock(BaseClient):
    def __init__(self, model_id, native):
        super().__init__(model_id)
        self.native = native

    def _raw_generate(self, request):
        return "fake", Usage(), None

    @classmethod
    def from_config(cls, *, model_id, native, config):
        return cls(model_id, native)


register_provider("mock", FakeMock)
'''


@pytest.fixture
def _clean_acme():
    REGISTRY.pop("acme", None)
    yield
    REGISTRY.pop("acme", None)


def test_loads_module_from_cwd_and_registers_provider(tmp_path, monkeypatch, _clean_acme):
    (tmp_path / "acme_provider.py").write_text(textwrap.dedent(_ACME_PLUGIN))
    monkeypatch.chdir(tmp_path)

    loaded = load_provider_plugins(["acme_provider"])

    assert loaded == ["acme_provider"]
    assert "acme" in REGISTRY
    assert make_client("acme/foo").call("hi") == "acme:foo"  # any model under the prefix resolves


def test_empty_or_blank_list_is_noop():
    assert load_provider_plugins([]) == []
    assert load_provider_plugins(["", "  "]) == []


def test_missing_module_raises_actionable_error():
    with pytest.raises(ImportError, match="Could not import provider module"):
        load_provider_plugins(["definitely_not_a_real_module_xyz"])


def test_override_of_builtin_warns(tmp_path, monkeypatch):
    (tmp_path / "override_mod.py").write_text(textwrap.dedent(_OVERRIDE_PLUGIN))
    monkeypatch.chdir(tmp_path)
    original = REGISTRY["mock"]
    try:
        with pytest.warns(UserWarning, match="overrides the built-in 'mock'"):
            load_provider_plugins(["override_mod"])
    finally:
        REGISTRY["mock"] = original  # keep the shared registry pristine for other tests


def test_config_parses_provider_modules(monkeypatch):
    monkeypatch.setenv("LLM_WORKBENCH_PROVIDERS", "acme_provider, other.mod")
    assert LabConfig.from_env().provider_modules == ["acme_provider", "other.mod"]


def test_split_modules_handles_commas_and_whitespace():
    assert _split_modules("a, b  c,,d") == ["a", "b", "c", "d"]
    assert _split_modules("") == []
