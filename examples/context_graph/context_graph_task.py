"""Custom llm_lab task: the open_source_models Neptune query protocol, replicated.

Self-contained replication of the experiment in
``/Users/stephaniekirmer/repos/experiments/experiments/open_source_models`` — it does
NOT import from or modify that folder. The pipeline is owned here; the two LLM prompts
live as llm_lab template files under ``examples/context_graph_prompts/`` and are filled
with llm_lab's own ``fill_template``. Live production infra (schema, org-scoping, actor
resolution, Neptune) is imported from the shared context-api repo — see
``context_graph_pipeline``.

Protocol (per record):
    natural-language question
      -> model-under-test generates an openCypher query   (query_gen.md + GRAPH_SECTION)
      -> validate; retry with feedback on failure
      -> _inject_org_scope (tenant isolation) + resolve actor params
      -> execute against BLUE prod Neptune
      -> model-under-test summarizes the rows            (answer_synthesis.md)
    aux exposes {queries, query_results, org_name} to the ported evals.

Prerequisites: run from an env where context-api + boto3 are importable (the
experiment venv is easiest — ``uv pip install -e <llm_lab>`` into it), with live
Neptune reachability (VPC/VPN) and AWS creds for Bedrock + Neptune.

Run it (from the llm_lab repo root so `examples.` resolves):
    llm-lab run compare \
        --task examples.context_graph_task:ContextGraphQueryTask \
        --dataset examples/context_graph_pilot.jsonl \
        --eval-dir evals_kit/context-graph \
        --model bedrock/us.anthropic.claude-opus-4-8 --model ollama/qwen3 \
        --judge-model bedrock/us.meta.llama3-1-70b-instruct-v1:0 \
        --out-dir runs/context-graph

Dataset rows (one JSON object per line):
    {"id": "detections-1", "input": "<question>", "org_id": "organization-live-...", "category": "detections"}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from llm_lab.evals.template import fill_template
from llm_lab.object_types import GenerationRequest
from llm_lab.tasks.base import TaskOutput

# Make the sibling pipeline module importable however this task is loaded
# (as `examples.context_graph_task` from the repo root, or `context_graph_task`
# from within examples/).
sys.path.insert(0, str(Path(__file__).parent))

# _log is shared with the pipeline module and gated by CONTEXT_GRAPH_VERBOSE=1, which
# surfaces per-stage progress (query-gen / validation / org-scope / execution /
# synthesis) and the pipeline's benign actor-resolution notes:
#   CONTEXT_GRAPH_VERBOSE=1 llm-lab run compare ...
from examples.context_graph.context_graph_pipeline import (  # noqa: E402
    GRAPH_SECTION,
    NEPTUNE_BLUE_ENDPOINT,
    ORG_NAMES,
    clean_and_extract_query,
    extract_actor_identifiers,
    get_neptune_client,
    global_rule_org_ids,
    resolve_actor_params,
    run_query,
    validate_cypher,
    _inject_org_scope,
    _log,
)

_PROMPT_DIR = Path(__file__).parent / "context_graph_prompts"
_QUERY_GEN_TEMPLATE = (_PROMPT_DIR / "query_gen.md").read_text()
_SYNTHESIS_TEMPLATE = (_PROMPT_DIR / "answer_synthesis.md").read_text()

_RETRY_TEMPLATE = """PREVIOUS ATTEMPT FAILED:
Query: {previous_attempt}
Error: {error_feedback}

Fix the error and try again. Focus on:
- Starting with valid openCypher keywords (MATCH, RETURN, WITH, etc.)
- Including a RETURN clause
- Proper syntax for relationships: (a)-[r:TYPE]->(b)
"""


class _WorkflowFailed(RuntimeError):
    """Raised when query generation / scoping / execution can't produce results, so
    llm_lab records a task-error row and continues (failed queries stay out of the
    score distribution) — matching the experiment's save_failed_workflow behavior."""


class ContextGraphQueryTask:
    name = "context-graph-query"

    def __init__(self, max_retries: int = 10, max_tokens: int = 2048):
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def _generate_query(self, client, user_prompt, previous_attempt, error_feedback):
        retry_section = ""
        if previous_attempt and error_feedback:
            retry_section = fill_template(
                _RETRY_TEMPLATE,
                {"previous_attempt": previous_attempt, "error_feedback": error_feedback},
            )
        prompt = fill_template(
            _QUERY_GEN_TEMPLATE,
            {"graph_section": GRAPH_SECTION, "retry_section": retry_section, "input": user_prompt},
        )
        _t0 = time.perf_counter()
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=self._max_tokens))
        if resp.error:
            raise _WorkflowFailed(f"query-gen model error: {resp.error}")
        _log(f"    · query-gen model call: {time.perf_counter() - _t0:.1f}s")
        return clean_and_extract_query(resp.text), resp

    def _synthesize(self, client, user_prompt, query, raw_results):
        results_text = json.dumps(raw_results[:10], indent=2) if raw_results else "No results found"
        if len(raw_results) > 10:
            results_text += f"\n\n... and {len(raw_results) - 10} more results"
        prompt = fill_template(
            _SYNTHESIS_TEMPLATE,
            {
                "input": user_prompt,
                "queries": query,
                "result_count": len(raw_results),
                "query_results": results_text,
            },
        )
        _t0 = time.perf_counter()
        resp = client.generate(GenerationRequest(prompt=prompt, max_tokens=self._max_tokens))
        if resp.error:
            raise _WorkflowFailed(f"synthesis model error: {resp.error}")
        _log(f"  · synthesis model call: {time.perf_counter() - _t0:.1f}s")
        return resp.text, resp

    def run(self, record, client) -> TaskOutput:
        user_prompt = record["input"]
        org_id = record.get("org_id")
        if not org_id:
            raise _WorkflowFailed("record has no org_id; cannot scope the query to a tenant")
        org_name = ORG_NAMES.get(org_id) or org_id
        _log(f"\n▶ {record.get('id', '?')}  org={org_name}\n  Q: {user_prompt[:140]}")

        # Pin to BLUE prod cluster; resolve actor identifiers once (reused across retries).
        neptune = get_neptune_client(endpoint=NEPTUNE_BLUE_ENDPOINT)
        actors = extract_actor_identifiers(user_prompt)
        if actors:
            _log(f"  actor identifiers: {actors}")
        actor_params = resolve_actor_params(org_id, actors)
        if actor_params["user_identifiers"] or actor_params["endpoint_uids"]:
            _log(f"  resolved: {len(actor_params['user_identifiers'])} user id(s), "
                 f"{len(actor_params['endpoint_uids'])} endpoint uid(s)")

        previous_attempt = error_feedback = None
        cypher_query = scoped_query = raw_results = None
        last_error = None

        for attempt in range(self._max_retries):
            _log(f"  step1 generate query (attempt {attempt + 1}/{self._max_retries})")
            # STEP 1: generate + validate
            cypher_query, _ = self._generate_query(client, user_prompt, previous_attempt, error_feedback)
            _log(f"    query: {' '.join(cypher_query.split())[:200]}")
            is_valid, error_msg = validate_cypher(cypher_query)
            if not is_valid:
                _log(f"    ✗ validation failed: {error_msg}")
                previous_attempt, error_feedback = cypher_query, error_msg
                last_error = f"Validation: {error_msg}"
                continue
            _log("    ✓ validation passed")

            # STEP 2: org-scope (tenant isolation) then execute against BLUE.
            try:
                scoped_query, scope_params = _inject_org_scope(cypher_query, org_id, global_rule_org_ids())
            except PermissionError as pe:
                # Fail-closed path (no org anchor / UNION / CALL): feed back to generator.
                _log(f"    ✗ org-scoping refused: {pe}")
                previous_attempt = cypher_query
                error_feedback = (
                    f"Org-scoping error: {pe} Anchor the query on an org-scoped node "
                    "(Rule, Finding, Entity, Hunt, HuntReport, ...) and avoid UNION/CALL."
                )
                last_error = f"OrgScope: {pe}"
                continue
            try:
                _log("  step2 execute against Neptune (BLUE)...")
                _t0 = time.perf_counter()
                raw_results = run_query(neptune, scoped_query, {**scope_params, **actor_params})
                _log(f"    ✓ {len(raw_results)} rows in {time.perf_counter() - _t0:.1f}s")
                break  # success
            except Exception as e:
                _log(f"    ✗ execution failed: {str(e)[:200]}")
                previous_attempt = cypher_query
                error_feedback = f"Query execution error: {e}"
                last_error = f"Execution: {e}"
                continue

        if raw_results is None:
            _log(f"  ✗ giving up after {self._max_retries} attempt(s): {last_error}")
            raise _WorkflowFailed(
                f"no results for org={org_id} after {self._max_retries} attempt(s): "
                f"{last_error!r} last_query={cypher_query!r}"
            )

        # STEP 3: synthesize the natural-language answer (unscoped query goes in the
        # prompt, matching the experiment; the scoped query is surfaced to the evals).
        _log("  step3 synthesize answer")
        final_response, synth_resp = self._synthesize(client, user_prompt, cypher_query, raw_results)
        _log(f"  ✓ answer: {len(final_response)} chars")

        return TaskOutput(
            output=final_response,
            prompt=scoped_query,
            response=synth_resp,  # final call, for the latency scorer
            aux={
                "queries": scoped_query,
                "query_results": json.dumps(raw_results, default=str)[:5000],
                "org_name": org_name,
            },
        )
