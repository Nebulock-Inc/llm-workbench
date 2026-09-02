"""Self-contained replication of the open_source_models Neptune query protocol.

This module re-implements — inside llm_lab — the deterministic pieces of the
experiment's pipeline so that NOTHING in the open_source_models folder is imported or
modified. The live production infrastructure (graph schema, tenant-isolation scoping,
actor resolution, Neptune client) is imported from the shared **context-api** repo,
exactly as the original experiment does.

What's replicated here (logic copied from the experiment, verbatim behavior):
  * openCypher post-processing + validation (clean_and_extract_query, validate_cypher)
  * heuristic actor-identifier extraction + entity-correlation resolution
  * the org_id -> display-name map (for the secure-org-refusal eval)
  * BLUE prod Neptune endpoint pinning + prod AWS profile

What's imported from context-api (shared prod code, not the experiment):
  * GRAPH_SECTION           - production schema + Cypher rules for the generator prompt
  * _inject_org_scope       - production tenant isolation applied before execution
  * global_rule_org_ids     - global-rule org allowlist for scoping
  * _resolve_to_actors      - identity resolution via the entity-correlation service
  * get_neptune_client / run_query - Neptune openCypher execution
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# --- Env + context-api path bootstrap (must precede context-api imports) ----------
# context-api binds config values (e.g. ENTITY_CORRELATION_API_URL) as module
# constants at import time, so the environment must be populated first. Prefer a
# llm_lab-local .env; fall back to reading (read-only, never modifying) the
# experiment's .env so this works out of the box. Create a local .env to be fully
# self-contained.
load_dotenv()  # llm_lab cwd .env, if present
_EXPERIMENT_ENV = Path(
    "/Users/stephaniekirmer/repos/experiments/experiments/open_source_models/.env"
)
if _EXPERIMENT_ENV.is_file():
    load_dotenv(_EXPERIMENT_ENV, override=False)

# context-api is a sibling repo; putting its root on sys.path makes `src` importable
# (mirrors the context_api.pth the experiment venv installs).
_CONTEXT_API_ROOT = "/Users/stephaniekirmer/repos/context-api"
if _CONTEXT_API_ROOT not in sys.path:
    sys.path.insert(0, _CONTEXT_API_ROOT)

from src.clients.neptune_client import get_neptune_client, run_query  # noqa: E402
import src.clients.neptune_client as _neptune_client_mod  # noqa: E402
from src.services.agents.context_agent.graph_prompt import GRAPH_SECTION  # noqa: E402
from src.services.agents.context_agent.tools.execute_cypher_query import (  # noqa: E402
    _inject_org_scope,
)
from src.services.rules.rule_visibility import global_rule_org_ids  # noqa: E402
from src.services.agents.shared.actor_resolution import _resolve_to_actors  # noqa: E402

# Force the Neptune client to the PROD profile regardless of the ambient AWS_PROFILE,
# and drop any client cached under a different profile (mirrors the experiment). A
# stray AWS_PROFILE=dev in the shell otherwise points prod Neptune queries at dev creds
# -> AccessDenied. Override with CONTEXT_GRAPH_NEPTUNE_PROFILE if your prod profile is
# named differently. NOTE: this only controls the Neptune client; llm_lab's Bedrock
# model + judge clients use the ambient AWS creds, so export AWS_PROFILE=prod too if
# you want every AWS call to hit prod like the original experiment did.
_neptune_client_mod.AWS_PROFILE = os.environ.get("CONTEXT_GRAPH_NEPTUNE_PROFILE", "prod")
_neptune_client_mod.close_neptune_client()

# Pin execution to the BLUE prod Neptune cluster: an explicit endpoint bypasses the
# SSM active-pointer resolution so a blue/green cutover never redirects us to green.
NEPTUNE_BLUE_ENDPOINT = os.environ.get(
    "NEPTUNE_BLUE_ENDPOINT",
    "context-graph-prod.cluster-cgxey6wgcke3.us-east-1.neptune.amazonaws.com",
)

# org_id -> human-readable org name, for the secure-org-refusal eval ({org_name}).
# Production resolves this from auth-app; that service isn't reachable here, so the
# mapping is explicit. Unmapped ids fall back to the raw org_id.
ORG_NAMES: Dict[str, str] = {
    "organization-live-745a924b-17f0-4785-808b-7e4073368189": "airSlate",
    "organization-live-03d5d19e-160e-45ec-b42e-35d9c9f8e87a": "Bain Capital",
    "organization-live-3aeed648-9aef-432d-8790-03d91646735c": "Cribl",
    "organization-live-46ff9696-8e1c-4566-874a-7ed3b11ce1f9": "HealthEdge",
    "organization-live-3802144d-5fc3-4e39-b3e2-2d6aa6fadb0b": "HYPR",
    "organization-live-8f315419-2f2f-4b97-a78d-fc0a0f96f1e5": "Ingersoll Rand",
    "organization-live-4de238d0-717a-4677-91b3-7b61b44d8af7": "Integralty",
    "organization-live-5d9fd6b5-734b-465b-8455-2714589db5f9": "Inductive Automation",
    "organization-live-04c280bb-b897-4dca-ba4b-5138eddb821e": "Latacora",
    "organization-live-61237870-2c4e-4924-b56f-e015f1b796df": "Lyra",
    "organization-live-133162aa-cf11-4716-ab65-e79dcf253eb8": "Plaid",
    "organization-live-0c354922-b4b2-4519-aa27-690081db55a1": "Shutterfly",
    "organization-live-f3c95ec3-0afb-4be1-8d2e-78033b6b1604": "UnderArmor",
    "organization-live-9e545fb2-da08-43b2-bc07-deb7fdc85ba5": "EmpiRx Health",
    "organization-live-a8685575-c71b-48a4-bc84-31e6b05b0df5": "Chipotle",
    "organization-live-1dbd74b0-3dca-47a5-9576-c6b527878d46": "Kestra",
    "organization-live-1db4ff3c-6fc2-4481-bf50-e64e832724ac": "Nebulock Prod",
    "organization-live-ad4f6469-f0a7-4b58-aa63-6e4d870eae68": "jamf",
    "organization-live-f6dbd635-6e70-480e-815a-c3cf8aa10f96": "geosyntec",
    "organization-live-e19510ba-3c83-4846-a6b5-d09f931d9783": "Liberty Mutual",
    "organization-live-13427773-60a7-4c9a-8def-e389106bb350": "Internal Portal",
    "organization-live-df68b93a-6ef8-4199-93a9-3c9ed6a47901": "Proskauer",
    "organization-live-e0a6d6ec-586e-4b0a-8739-c9d2ce8894b6": "Synthetic Data",
    "organization-live-a113b0c3-28e9-4c87-9cec-518e94cd873f": "Pentest",
    "organization-live-5e548194-c1d3-4ec9-8cf1-71ab9fd2e8c4": "Spotnana",
    "organization-live-308177bb-caa8-493f-bd36-8689a36c30da": "Illumina",
    "organization-live-2cf4835d-3a65-4f7d-8d23-002828d93cec": "Buckingham Manufacturing",
    "organization-live-14b18360-09b7-4c39-8185-513fa4f4b3d1": "Spinnerstown Hotel",
    "organization-live-ca002a3f-d43e-4ebd-aa53-79f6708fe1e7": "Wyler LLC",
    "organization-live-bc811d62-06e2-4122-ae1f-fd3f8f4be937": "St. Lukes",
    "organization-live-0792e57b-b3c4-4497-8c6e-bb2af4157d38": "zelis",
    "organization-live-81f16fb5-77f0-4abb-aaea-af8e75a40231": "Recovery Centers of America",
    "organization-live-493b5a1b-4872-4eb2-9368-122fec699fd3": "IQT",
    "organization-live-6e60b6dc-803e-42d5-92e6-3667c85292f0": "Jeppesen",
    "organization-live-aefc1fc9-aa3d-44f9-ba60-c5cc0ca663b3": "Toyota Financial",
}


# --- openCypher post-processing + validation (copied from experiment/neptune.py) --

def extract_final_query(text: str) -> str:
    """Extract the complete openCypher query, cutting off any reasoning/explanation."""
    text_stripped = text.strip()
    start_pattern = r'(?:^|\n)(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|CREATE|MERGE)\s+\('
    start_match = re.search(start_pattern, text, re.MULTILINE)
    if not start_match:
        return text_stripped
    query_text = text[start_match.start():]
    end_match = re.search(r'(.*?LIMIT\s+\d+)', query_text, re.DOTALL | re.IGNORECASE)
    if end_match:
        query = end_match.group(1).strip()
    else:
        reasoning_indicators = [
            r'\n\s*(?:Wait|However|But|Hmm|Alternatively|Actually|Note:|Also,)',
            r'\n\s*--\s*(?:assuming|but|since)',
            r'\?',
        ]
        cut_position = len(query_text)
        for indicator in reasoning_indicators:
            match = re.search(indicator, query_text, re.IGNORECASE)
            if match:
                cut_position = min(cut_position, match.start())
        query = query_text[:cut_position].strip()
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    return query.strip()


def clean_and_extract_query(raw_response: str) -> str:
    """Recover a bare openCypher query from a raw LLM response."""
    cypher = raw_response.strip()
    query_keywords = ['MATCH', 'OPTIONAL MATCH', 'WITH', 'RETURN', 'UNWIND', 'CREATE', 'MERGE']
    for keyword in query_keywords:
        match = re.search(f'(?:^|\\n)({keyword}\\s+\\()', cypher)
        if match:
            cypher = cypher[match.start():].strip()
            break
    if "```" in cypher:
        parts = cypher.split("```")
        if len(parts) >= 3:
            cypher = parts[1].strip()
            lines = cypher.split('\n')
            if lines and lines[0].lower() in ['cypher', 'opencypher', 'sql', '']:
                cypher = '\n'.join(lines[1:]).strip()
    return extract_final_query(cypher)


def validate_cypher(query: str) -> tuple[bool, str]:
    """Enhanced validation of openCypher syntax. Returns (is_valid, error_message)."""
    query_upper = query.strip().upper()
    query_original = query.strip()
    valid_starts = ['MATCH', 'RETURN', 'WITH', 'UNWIND', 'CREATE', 'MERGE', 'OPTIONAL']
    if not any(query_upper.startswith(start) for start in valid_starts):
        return False, f"Query doesn't start with valid keyword. Starts with: {query_original[:50]}"
    if 'RETURN' not in query_upper:
        return False, "Query missing RETURN clause"
    if query_upper.count('(') != query_upper.count(')'):
        return False, "Unbalanced parentheses"
    if query_upper.count('[') != query_upper.count(']'):
        return False, "Unbalanced brackets"
    if re.search(r'MATCH\s+\([^)]*\.[^)]*\)', query_upper):
        return False, ("Invalid dot syntax in MATCH clause. Use (variable:Label) format, access "
                       "properties in WHERE/RETURN only. Example: MATCH (h:Hunt) WHERE h.title = 'x' RETURN h.title")
    if re.search(r'\(\w+\.\w+:', query_original):
        return False, "Invalid syntax: dot found before label colon. Use (variable:Label) not (variable.property:Label)"
    if re.search(r'\w+=[\'"]', query_original) or re.search(r'[\'"]=[\'"]', query_original):
        return False, "Missing space around = operator. Use: WHERE h.status = 'active' not WHERE h.status='active'"
    if re.search(r'(MATCH|WHERE|RETURN|WITH)\(', query_original, re.IGNORECASE):
        return False, "Missing space after keyword. Use: MATCH (h:Hunt) not MATCH(h:Hunt)"
    if 'LIMIT' in query_upper and 'ORDER BY' in query_upper:
        if query_upper.find('ORDER BY') > query_upper.find('LIMIT'):
            return False, "ORDER BY must come before LIMIT. Correct order: RETURN ... ORDER BY ... LIMIT ..."
    if 'LIMIT' in query_upper and 'SKIP' in query_upper:
        if query_upper.find('SKIP') > query_upper.find('LIMIT'):
            return False, "SKIP must come before LIMIT. Correct order: ... SKIP ... LIMIT ..."
    return True, ""


# --- Actor / identity resolution (copied from experiment/run_model_tests.py) -------

_SYNTHETIC_MEMBER_ID = "context-api-internal"
_MITRE_RE = re.compile(r'^T\d{4}(?:\.\d+)?$', re.I)
_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_UUID_RE = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)
_HEX_RE = re.compile(r'\b[0-9a-f]{32,}\b', re.I)
_TOKEN_RE = re.compile(r'\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{5,}\b')
_DOTTED_NAME_RE = re.compile(r'\b[a-z][a-z]+\.[a-z][a-z]+\b')
_FILE_EXT = ('.exe', '.dll', '.py', '.sh', '.ps1', '.dylib')
_MAX_IDENTIFIERS = 8


def _synthesize_session_headers(organization_id: str) -> Dict[str, str]:
    """Unsigned Nebulock session header scoped to one org (dev-only)."""
    payload = {"organization_id": organization_id, "member_id": _SYNTHETIC_MEMBER_ID}
    return {"x-nebulock-session-data": json.dumps(payload)}


class _ResolverState:
    """Minimal stand-in for Strands AgentState: typed get/set over a dict."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = dict(data)

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class _NoOpSpan:
    """Tracing span stub — _resolve_to_actors only calls set_attribute/add_event."""

    def set_attribute(self, *args, **kwargs) -> None:
        pass

    def add_event(self, *args, **kwargs) -> None:
        pass


def extract_actor_identifiers(user_prompt: str):
    """Heuristically pull actor identifiers from a prompt as (lane, value) pairs."""
    text = user_prompt or ""
    found: list = []
    seen: set = set()
    claimed: set = set()

    def add(lane: str, value: str) -> None:
        v = value.strip()
        key = (lane, v.lower())
        if key in seen:
            return
        seen.add(key)
        claimed.add(v.lower())
        found.append((lane, v))

    for m in _EMAIL_RE.findall(text):
        add("user_email", m)
    for m in _UUID_RE.findall(text):
        add("endpoint_uid", m)
    for m in _HEX_RE.findall(text):
        if m.lower() not in claimed:
            add("endpoint_uid", m)
    for m in _TOKEN_RE.findall(text):
        if _MITRE_RE.match(m) or m.lower() in claimed or m.lower().endswith(_FILE_EXT):
            continue
        add("search_text", m)
    for m in _DOTTED_NAME_RE.findall(text):
        if m.lower() in claimed or m.lower().endswith(_FILE_EXT):
            continue
        add("search_text", m)

    return found[:_MAX_IDENTIFIERS]


def resolve_actor_params(organization_id: str, identifiers) -> Dict[str, list]:
    """Resolve actor identifiers to graph-joinable IDs via entity-correlation.

    Returns {"user_identifiers": [...], "endpoint_uids": [...]} — ALWAYS both keys
    (possibly empty) so a query referencing either never fails with MissingParameter.
    On any resolution failure (service unreachable/unconfigured, no match, error) the
    affected lane simply contributes nothing; never raises.
    """
    params: Dict[str, list] = {"user_identifiers": [], "endpoint_uids": []}
    identifiers = list(identifiers or [])
    if not identifiers or not organization_id:
        return params

    state = _ResolverState({
        "organization_id": organization_id,
        "nebulock_headers": _synthesize_session_headers(organization_id),
        "resolve_cache": {},
        "resolves": [],
    })
    span = _NoOpSpan()

    user_ids: set = set()
    endpoint_uids: set = set()
    for lane, value in identifiers:
        try:
            resolved = _resolve_to_actors(lane, value, state, span)
        except Exception as e:  # never let resolution break the workflow
            print(f"⚠️  actor resolution error for {lane}={value!r}: {e}")
            continue
        if isinstance(resolved, str):
            print(f"ℹ️  no actors for {lane}={value!r}: {resolved[:80]}")
            continue
        for a in resolved.actors:
            user_ids.update(a.user_identifiers)
            endpoint_uids.update(a.endpoint_uids)

    params["user_identifiers"] = sorted(user_ids)
    params["endpoint_uids"] = sorted(endpoint_uids)
    return params
