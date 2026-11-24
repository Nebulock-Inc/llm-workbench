"""Offline test provider.

`mock/echo` returns the prompt; `mock/canned:<text>` returns a fixed string
(handy for exercising the judge score-parser without a real model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_workbench.models.base import BaseClient
from llm_workbench.object_types import GenerationRequest, Usage

if TYPE_CHECKING:
    from llm_workbench.config import LabConfig



class MockClient(BaseClient):
    def __init__(self, model_id: str, native: str = "echo") -> None:
        super().__init__(model_id)
        self.native = native

    def _raw_generate(
        self, request: GenerationRequest
    ) -> tuple[str, Usage, dict[str, Any] | None]:
        if self.native.startswith("canned:"):
            text = self.native.split(":", 1)[1]
        else:  # echo
            text = request.prompt
        n_in = len(request.prompt.split())
        n_out = len(text.split())
        usage = Usage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out)
        return text, usage, {"mock": True, "native": self.native}

    @classmethod
    def from_config(cls, *, model_id: str, native: str, config: "LabConfig") -> "MockClient":
        return cls(model_id, native=native)
