"""Built-in tasks: ReplayTask (score a stored answer, skip the model under test)."""

from __future__ import annotations

import pytest

from llm_workbench.tasks import ReplayTask
from llm_workbench.tasks.loader import load_task


def _record():
    return {
        "id": "grounding-bad",
        "input": "some event",
        "query_results": "some rows",
        "output": "VERDICT: AI\n\nThis was clearly an AI tool.",
        "verdict": "AI",
        "expected_band": "low",
    }


def test_replay_returns_stored_output_without_calling_client():
    # client is None: a real model call would raise, proving the model under test is skipped.
    out = ReplayTask().run(_record(), client=None)
    assert out.output.startswith("VERDICT: AI")
    assert out.response is None  # no generation -> latency scorer is N/A


def test_replay_copies_all_record_fields_into_aux_by_default():
    # Full passthrough lets ground_truth.aux_field (e.g. `verdict`) resolve with no wiring.
    out = ReplayTask().run(_record(), client=None)
    assert out.aux["verdict"] == "AI"
    assert out.aux["expected_band"] == "low"


def test_replay_aux_fields_selects_subset():
    out = ReplayTask(aux_fields=["verdict"]).run(_record(), client=None)
    assert out.aux == {"verdict": "AI"}


def test_replay_custom_field():
    rec = {"stored": "the answer"}
    assert ReplayTask(field="stored").run(rec, client=None).output == "the answer"


def test_replay_missing_field_raises_with_helpful_message():
    with pytest.raises(KeyError, match="nope"):
        ReplayTask(field="nope").run(_record(), client=None)


def test_replay_loadable_by_task_spec_zero_arg():
    # Mirrors the CLI `--task llm_workbench.tasks:ReplayTask` path (no constructor kwargs).
    task = load_task("llm_workbench.tasks:ReplayTask")
    assert task.name == "replay"
    assert task.run(_record(), client=None).output.startswith("VERDICT: AI")
