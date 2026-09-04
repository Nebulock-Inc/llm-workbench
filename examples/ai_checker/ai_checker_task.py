"""Custom llm_lab task: the ai-checker ClickHouse protocol.

Per record (one flagged event/endpoint):
    event facts + question
      -> model-under-test generates a ClickHouse SELECT   (query_gen.md + CLICKHOUSE_SECTION)
      -> validate_sql; retry with feedback on failure
      -> inject_org_scope (tenant isolation via additional_table_filters)
      -> execute against ClickHouse
      -> model-under-test decides AI involvement           (answer_synthesis.md)
                                                            first line: `VERDICT: AI|NOT_AI`
    aux exposes {queries, query_results, org_name, verdict} to the evals.

Run it (from the llm_lab repo root so `examples.` resolves):
    AI_CHECKER_VERBOSE=1 llm-lab run compare \
        --task examples.ai_checker.ai_checker_task:AICheckerTask \
        --dataset examples/ai_checker/pilot.jsonl \
        --eval-dir evals_kit/ai-checker \
        --model bedrock/us.anthropic.claude-opus-4-8 --model ollama/qwen3 \
        --judge-model bedrock/us.meta.llama3-1-70b-instruct-v1:0 \
        --out-dir runs/ai-checker

Dataset rows (one JSON object per line):
    {"id": "ev-1", "input": "<event facts / question>", "org_id": "org-...", "endpoint_uid": "..."}
Evaluation is judge-only (no ground-truth verdict field); the evals under
evals_kit/ai-checker/ grade the model's determination and SQL.

Prereqs: `uv sync --extra clickhouse`, ClickHouse creds in .env (see connect_clickhouse.py),
and network reachability to the cluster.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from llm_lab.evals.template import fill_template
from llm_lab.object_types import GenerationRequest
from llm_lab.tasks.base import TaskOutput

# Make the sibling modules importable however this task is loaded (as
# `examples.ai_checker.ai_checker_task` from the repo root, or bare from within the dir).
sys.path.insert(0, str(Path(__file__).parent))

from ai_checker_pipeline import (  # noqa: E402
    CLICKHOUSE_SECTION,
    extract_sql,
    inject_org_scope,
    org_display_name,
    parse_risk,
    parse_verdict,
    run_query,
    validate_sql,
    _log,
)
from connect_clickhouse import get_client  # noqa: E402

_PROMPT_DIR = Path(__file__).parent / "ai_checker_prompts"
_QUERY_GEN_TEMPLATE = (_PROMPT_DIR / "query_gen.md").read_text()
_SYNTHESIS_TEMPLATE = (_PROMPT_DIR / "answer_synthesis.md").read_text()

# Standing question used when a record's `input` is blank. The event is identified by
# event_id/timestamp, so pilot rows can leave `input` empty and inherit this; set a
# row's `input` to override with a case-specific question.
_DEFAULT_QUESTION = (
    "Was this event created by, or does it involve, AI? Also assess whether the event is malicious or "
    "suspicious in any way."
)

_RETRY_TEMPLATE = """PREVIOUS ATTEMPT FAILED:
Query: {previous_attempt}
Error: {error_feedback}

Fix the error and try again. Emit a single read-only SELECT with a time bound and a LIMIT.
"""


class _WorkflowFailed(RuntimeError):
    """Raised when query gen / scoping / execution can't produce results, so llm_lab
    records a task-error row and continues the sweep (failed rows stay out of the
    score distribution)."""


class AICheckerTask:
    name = "ai-checker-query"

    def __init__(self, max_retries: int = 6, max_tokens: int = 2048):
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def _generate_query(self, client, event_info, event_id, timestamp, previous_attempt, error_feedback):
        retry_section = ""
        if previous_attempt and error_feedback:
            retry_section = fill_template(
                _RETRY_TEMPLATE,
                {"previous_attempt": previous_attempt, "error_feedback": error_feedback},
            )
        prompt = fill_template(
            _QUERY_GEN_TEMPLATE,
            {
                "clickhouse_section": CLICKHOUSE_SECTION,
                "retry_section": retry_section,
                "input": event_info,
                "event_id": event_id,
                "timestamp": timestamp,
            },
        )
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=self._max_tokens))
        if resp.error:
            raise _WorkflowFailed(f"query-gen model error: {resp.error}")
        return extract_sql(resp.text), resp

    def _synthesize(self, client, event_info, query, rows):
        results_text = json.dumps(rows[:10], indent=2, default=str) if rows else "No results found"
        if len(rows) > 10:
            results_text += f"\n\n... and {len(rows) - 10} more rows"
        prompt = fill_template(
            _SYNTHESIS_TEMPLATE,
            {
                "input": event_info,
                "queries": query,
                "result_count": len(rows),
                "query_results": results_text,
            },
        )
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=self._max_tokens))
        if resp.error:
            raise _WorkflowFailed(f"synthesis model error: {resp.error}")
        return resp.text, resp

    def run(self, record, client) -> TaskOutput:
        event_info = (record.get("input") or "").strip() or _DEFAULT_QUESTION
        org_id = record.get("org_id")
        if not org_id:
            raise _WorkflowFailed("record has no org_id; cannot scope the query to a tenant")
        event_id = record.get("event_id", "")
        timestamp = record.get("timestamp", "")
        org_name = org_display_name(org_id)
        _log(f"\n▶ {record.get('id', '?')}  org={org_name} event_id={event_id}\n  event: {str(event_info)[:140]}")

        clickhouse = get_client()

        previous_attempt = error_feedback = None
        sql = scoped_settings = params = rows = None
        last_error = None

        for attempt in range(self._max_retries):
            _log(f"  step1 generate SQL (attempt {attempt + 1}/{self._max_retries})")
            sql, _ = self._generate_query(client, event_info, event_id, timestamp, previous_attempt, error_feedback)
            _log(f"    generated SQL:\n{sql}\n    ────")

            ok, err = validate_sql(sql)
            if not ok:
                _log(f"    ✗ validation failed: {err}")
                previous_attempt, error_feedback = sql, err
                last_error = f"Validation: {err}"
                continue
            _log("    ✓ validation passed")

            try:
                sql, scoped_settings, params = inject_org_scope(sql, org_id)
            except PermissionError as pe:
                _log(f"    ✗ org-scoping refused: {pe}")
                previous_attempt = sql
                error_feedback = f"Org-scoping error: {pe}"
                last_error = f"OrgScope: {pe}"
                continue

            try:
                _log("  step2 execute against ClickHouse...")
                t0 = time.perf_counter()
                rows = run_query(clickhouse, sql, scoped_settings, params)
                _log(f"    ✓ {len(rows)} rows in {time.perf_counter() - t0:.1f}s")
                if not rows:
                    # No rows = the event wasn't found. Never let the model judge on empty
                    # data — retry (it may widen the window), and fail the record if still
                    # empty after all attempts (see the `rows is None` check below).
                    _log("    ✗ query returned no rows; will not synthesize a verdict on empty data")
                    previous_attempt = sql
                    error_feedback = (
                        "Query returned no rows — the event was not found. Widen the time bounds "
                        "or loosen the filters so the target event is returned. Do NOT fabricate data."
                    )
                    last_error = "No results returned"
                    rows = None
                    continue
                break
            except Exception as e:
                _log(f"    ✗ execution failed: {e}")
                _log(f"    failing query:\n{sql}")
                _log(f"    org-scope settings: {scoped_settings.get('additional_table_filters')}")
                previous_attempt = sql
                error_feedback = f"Query execution error: {e}"
                last_error = f"Execution: {e}"
                continue

        if rows is None:
            _log(f"  ✗ giving up after {self._max_retries} attempt(s): {last_error}")
            raise _WorkflowFailed(
                f"no results for org={org_id} after {self._max_retries} attempt(s): "
                f"{last_error!r} last_query={sql!r}"
            )

        _log("  step3 synthesize verdict")
        final_response, synth_resp = self._synthesize(client, event_info, sql, rows)
        verdict = parse_verdict(final_response)
        risk = parse_risk(final_response)
        _log(f"  ✓ verdict={verdict} risk={risk} answer: {len(final_response)} chars")

        return TaskOutput(
            output=final_response,
            prompt=sql,
            response=synth_resp,
            aux={
                # Surface the *effective* question (row's input, or the default when blank)
                # so judge prompts that reference {input} see it too — aux overrides the
                # (possibly empty) record `input` field in the judge's variable merge.
                "input": event_info,
                "queries": sql,
                "query_results": json.dumps(rows, default=str)[:5000],
                "org_name": org_name,
                "verdict": verdict or "",
                "risk": risk or "",
            },
        )
