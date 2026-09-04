"""ClickHouse + org-scope engine for the ai-checker experiment.

Parity with ``examples/context_graph_pipeline.py``, but for ClickHouse SQL. Owns the
schema the model is shown, SQL validation, the server-side tenant-isolation rewrite,
query execution, and verdict parsing. The LLM under test is *not* called here — that
happens in ``ai_checker_task.py`` via llm_lab's ``client.generate`` so models can be
compared.

Org isolation (server-side SQL rewrite): ``inject_org_scope`` never trusts the model to
filter by tenant. It attaches an ``additional_table_filters`` ClickHouse setting that ANDs
``nebulock_organization_id = '<org>'`` onto every org-scoped table the query touches. This
is applied by the server at plan time, so it holds regardless of the model's WHERE / joins /
aggregations (an ``OR``/``UNION`` in the model's own SQL cannot weaken it), and fails closed
if the query references no org-scoped table. The org id is embedded as a strictly-validated
literal (``^[A-Za-z0-9:_.-]+$``) rather than a bound parameter, because parameter binding
inside a settings expression is not reliably supported; the validation guarantees no quote
can break out of the literal.

Schema note: ``CLICKHOUSE_SECTION`` below is seeded from hunt-vault
(``knowledge/OCSF_SCHEMA_REFERENCE.md`` and the ai-automation investigator prompt). Confirm
the exact column set against that reference before a production run.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

# Pin AWS to the prod profile/region for this experiment BEFORE llm_lab builds any
# Bedrock client. The runner imports this task module (via load_task) before it calls
# LabConfig.from_env(), and boto3 reads AWS_PROFILE at client-construction time — so
# forcing it here overrides whatever the invoking shell exported (a stray AWS_PROFILE=dev
# was sending the Bedrock judge to the wrong account). Mirrors the context-graph pipeline.
os.environ["AWS_PROFILE"] = "prod"
os.environ.setdefault("AWS_REGION", "us-east-1")

# ── configuration ────────────────────────────────────────────────────────────
ORG_COLUMN = "nebulock_organization_id"

# Tables that carry a tenant column and MUST be org-scoped. A query that references
# none of these cannot be isolated and is refused (fail closed).
ORG_SCOPED_TABLES = {
    "nocsf_unified_events",
    "nocsf_unified_events_intake",
    "ai_automation_api_results",
    "raw_events_archive",
}

MAX_EXECUTION_TIME = 60      # seconds; kills runaway generated queries
MAX_RESULT_ROWS = 10_000     # hard cap on rows returned to the pipeline

_ORG_ID_RE = re.compile(r"^[A-Za-z0-9:_.\-]+$")
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([`\w.]+)", re.IGNORECASE)

# org_id -> human-friendly name, surfaced to the org-refusal eval. Fill with the real
# mapping (hunt-vault resolves these via auth-app; hard-code the orgs in your pilot set).
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

# ── schema shown to the model ─────────────────────────────────────────────────
CLICKHOUSE_SECTION = """\
ClickHouse schema (query ONLY these tables; all are automatically scoped to the caller's org):

Table `nocsf_unified_events` — OCSF-normalized endpoint/process events. Dotted field names MUST
be backtick-quoted. Composite primary key: `endpoint.uid` + nebulock_event_id + nebulock_organization_id.
Key columns:
  nebulock_event_id          UUID          — unique event id (NOT dotted; no backticks needed)
  `time`                     DateTime64(3) — event timestamp; the time field is named `time`, NOT
                                             `timestamp`. Always add a bound on `time`. When using
                                             time functions, wrap it: toDateTime(`time`).
  nebulock_ingest_time    DateTime64(3) — when the event was ingested (partitioning/ordering key).
                                             ALWAYS also filter on this — it makes queries much
                                             faster. Bound it around the event's timestamp too.
  `endpoint.uid`             String        — unique endpoint id
  `nebulock_im_source`       String        — event source, e.g. 'im-process_create'
  `process.name`             String        — process image name (e.g. 'bash', 'Cursor.exe')
  `process.command_line`     String        — full command line
  `parent.process.name`      String        — parent process image name
  `parent.process.command_line` String     - parent process full command line
  `actor.user.name`          String        — acting user
  `endpoint.name`            String        — endpoint / host name (use this, not endpoint.hostname)
  `file.path`                String        — file path touched, when present
  `nebulock_organization_id` String        — tenant id (do NOT filter on this; auto-applied)
  `event.type`               String        - Identifier describing what type of event it is
  `endpoint.os.type`         String        - Operating system in use on the endpoint
  `api.request.user_name`    String        - API
  `api.service.name`         String        - API
  `product.name`             String        - Application name
  `software.name`            String        - Application name
  `destination.network_endpoint.domain`  String - DNS destination
  `dns.question.name`        String        - DNS 
"""


# ── logging ────────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    if os.getenv("AI_CHECKER_VERBOSE") or os.getenv("CONTEXT_GRAPH_VERBOSE"):
        print(msg, flush=True)


# ── SQL extraction + validation ─────────────────────────────────────────────────
def _strip_sql_comments(sql: str) -> str:
    """Remove -- line comments and /* */ block comments, but never touch text inside
    single-quoted string literals (command-line matches legitimately contain '--flags'
    like '--textconv'). Handles ClickHouse '' and \\' escapes."""
    out: list[str] = []
    i, n = 0, len(sql)
    in_str = False
    while i < n:
        c = sql[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:      # backslash escape: keep next char verbatim
                out.append(sql[i + 1])
                i += 2
                continue
            if c == "'":
                if i + 1 < n and sql[i + 1] == "'":  # doubled '' escape
                    out.append("'")
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            out.append(c)
            i += 1
        elif c == "-" and i + 1 < n and sql[i + 1] == "-":       # line comment
            while i < n and sql[i] not in "\r\n":
                i += 1
        elif c == "/" and i + 1 < n and sql[i + 1] == "*":       # block comment
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response.

    Strips markdown code fences and any preamble, removes SQL comments, then keeps
    everything from the first SELECT/WITH keyword to the first statement terminator.
    """
    t = text.strip()
    # Drop ```sql ... ``` fences if present.
    fence = re.search(r"```(?:sql)?\s*(.+?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    # Start at the first query keyword.
    kw = re.search(r"\b(WITH|SELECT)\b", t, re.IGNORECASE)
    if kw:
        t = t[kw.start():]
    # Remove comments (string-literal aware), then keep only the first statement.
    t = _strip_sql_comments(t)
    t = t.split(";")[0].strip()
    return t


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, error_message). Enforces SELECT-only single statements, blocks
    attempts to override our injected settings, then runs hunt-vault's cost-safety
    validator (time bound + LIMIT + no comments)."""
    if not sql.strip():
        return False, "Empty query."

    # Single statement only (extract_sql already trims, but guard raw input too).
    if ";" in sql.strip().rstrip(";"):
        return False, "Multiple statements detected; emit a single SELECT."

    lowered = sql.lower()
    head = re.sub(r"^\s*with\b.*?\bselect\b", "select", lowered, flags=re.DOTALL)
    if not head.lstrip().startswith("select"):
        return False, "Only read-only SELECT queries are allowed (optionally led by WITH)."

    # Block DML/DDL keywords anywhere (defense in depth beyond the SELECT-start check).
    if re.search(r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|optimize|grant|revoke|system)\b", lowered):
        return False, "Write/DDL statements are not permitted."

    # The model must not set its own tenancy-affecting settings.
    if re.search(r"additional_table_filters|readonly", lowered):
        return False, "Do not set additional_table_filters or readonly; org scoping is applied automatically."

    from query_validator import QueryValidator  # local, self-contained copy

    result = QueryValidator().validate(sql, target="clickhouse")
    # extract_sql already strips real comments (string-literal aware), so any "comment"
    # the validator flags is a '--'/'/*' inside a string literal (e.g. matching '--textconv')
    # and is a false positive — drop it.
    errors = [e for e in result.errors if "comment" not in e.lower()]
    if errors:
        return False, "; ".join(errors)
    return True, ""


# ── org scoping (server-side rewrite) ─────────────────────────────────────────────
def _referenced_org_scoped_tables(sql: str) -> list[str]:
    """Return the query's table references (as written) that map to org-scoped tables."""
    refs: list[str] = []
    for ident in _TABLE_REF_RE.findall(sql):
        bare = ident.strip("`").split(".")[-1].strip("`")
        if bare in ORG_SCOPED_TABLES:
            refs.append(ident.strip("`"))
    return refs


def inject_org_scope(sql: str, org_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Attach tenant isolation to a validated SELECT. Returns (sql, settings, params).

    Raises PermissionError (fail closed) if org_id is malformed or the query touches no
    org-scoped table.
    """
    if not org_id or not _ORG_ID_RE.match(org_id):
        raise PermissionError(f"Refusing to scope with an invalid organization id: {org_id!r}")

    tables = _referenced_org_scoped_tables(sql)
    if not tables:
        raise PermissionError(
            "Query references no org-scoped table; cannot enforce tenant isolation. "
            f"Anchor it on one of: {', '.join(sorted(ORG_SCOPED_TABLES))}."
        )

    filter_expr = f"{ORG_COLUMN} = '{org_id}'"
    escaped = filter_expr.replace("'", "\\'")  # embed inside the Map value string literal
    entries = ", ".join(f"'{ident}' : '{escaped}'" for ident in dict.fromkeys(tables))
    settings: dict[str, Any] = {
        "additional_table_filters": "{" + entries + "}",
        "max_execution_time": MAX_EXECUTION_TIME,
        "max_result_rows": MAX_RESULT_ROWS,
        "readonly": 2,  # allow the settings above, forbid any write
    }
    return sql, settings, {}


# ── execution ──────────────────────────────────────────────────────────────────
def run_query(client, sql: str, settings: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute a scoped query and return rows as a list of dicts."""
    result = client.execute_query(sql, parameters=params or None, settings=settings or None)
    columns = result["columns"]
    return [dict(zip(columns, row)) for row in result["data"]]


# ── verdict parsing ──────────────────────────────────────────────────────────────
def parse_verdict(text: str) -> str | None:
    """Extract the model's AI/NOT_AI verdict from a `VERDICT: ...` line. Returns
    'AI', 'NOT_AI', or None if no verdict token is found."""
    m = re.search(r"VERDICT\s*[:\-]\s*(NOT[_\s-]?AI|AI)\b", text, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).upper()
    return "NOT_AI" if "NOT" in token else "AI"


def parse_risk(text: str) -> str | None:
    """Extract the model's risk opinion from a `RISK: ...` line. Returns
    'BENIGN', 'SUSPICIOUS', 'MALICIOUS', or None if not found."""
    m = re.search(r"RISK\s*[:\-]\s*(BENIGN|SUSPICIOUS|MALICIOUS)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def org_display_name(org_id: str) -> str:
    return ORG_NAMES.get(org_id, org_id)
