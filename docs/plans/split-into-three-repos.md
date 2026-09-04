# Plan: split llm-lab into three repos

## Decisions (locked)

- **Topology:** three separate git repos; apps depend on llm-lab via a git dependency.
- **App repo names:** `context-graph-evals`, `ai-checker-evals`.
- **GitHub org:** `stephanie-nebulock` (same as `llm_lab`).
- **Git history:** start fresh (clean initial commit per app; no history extraction).
- Local dirs: `~/repos/context-graph-evals`, `~/repos/ai-checker-evals`.
- No GitHub repo creation/push by the assistant without explicit sign-off.

## What stays in llm-lab (the reusable library)

- `src/llm_lab/`, `tests/`, `docs/`
- Generic evals: `evals_kit/{demo-quality, example-helpfulness, hallucination, query-quality}`
- Generic examples: `examples/{my_task.py, prompt.txt, dataset.jsonl, org_scoped_task.py}`
- Its pyproject keeps the core + `openai`/`anthropic`/`analysis`/`dev` extras.
- **Remove** the `clickhouse` extra (it existed only for ai-checker) — moves to that app.

## The apps only touch a tiny, stable llm-lab surface

Both import exactly:
- `llm_lab.evals.template.fill_template`
- `llm_lab.object_types.GenerationRequest`
- `llm_lab.tasks.base.TaskOutput`

They do **not** import each other. Task entrypoints:
- ai-checker: `AICheckerTask` (`name = "ai-checker-query"`), in `ai_checker_task.py`
- context-graph: `ContextGraphQueryTask` (`name = "context-graph-query"`), in `context_graph_task.py`

## External-dependency findings (important)

- **ai-checker** is mostly self-contained: local copies of `query_validator.py` and
  `connect_clickhouse.py`; the `nebulock_hunt_vault` import is already optional/guarded.
  Deps: llm-lab (git) + `clickhouse-connect` + `python-dotenv`. Optional: hunt-vault.
- **context-graph** is hard-wired to the real context-graph **service** source:
  `context_graph_pipeline.py` imports `from src.clients.neptune_client ...`,
  `from src.services.agents.context_agent ...`, `from src.services.rules ...`,
  `from src.services.agents.shared.actor_resolution ...`.
  The service (`~/repos/context-graph`, `Nebulock-Inc/context-graph`) is a run-from-root
  app (`pythonpath=["src"]`), **not** a pip-installable package. So it cannot be added as
  a normal git dependency without repackaging. **Decision needed** (see below).

## Resolved: how each app reaches its non-llm-lab code

The true source of the context-graph prod code is **context-api**
(`~/repos/context-api`), not the context-graph service — the pipeline bootstraps
context-api onto `sys.path` and imports `src.clients.neptune_client`,
`src.services.agents.context_agent.graph_prompt`, `.tools.execute_cypher_query`,
`src.services.rules.rule_visibility`, `src.services.agents.shared.actor_resolution`.
Those pull a deep transitive tree (`src.config`, `src.logging_config`,
`src.services.agents.shared.budget`, `src.db.models.context_models`,
`src.clients.entity_correlation_client`), so vendoring is impractical and fights the
pipeline's deliberate live-import design.

- **ai-checker-evals → VENDORED.** Its `query_validator.py` / `connect_clickhouse.py` are
  already local copies; keep them in the repo. hunt-vault stays an optional guarded import.
- **context-graph-evals → context-api CHECKOUT on PYTHONPATH.** Keep the live-import
  design; replace the hardcoded `/Users/stephaniekirmer/repos/context-api` with a
  configurable `CONTEXT_API_ROOT` env var (documented in `.env.template`/README). No
  pyproject dep on context-api. Also make the fallback experiment `.env` path
  (`/Users/.../open_source_models/.env`) configurable, not hardcoded.

## Target layout per app (example: ai-checker-evals)

```
ai-checker-evals/
  pyproject.toml            # depends on llm-lab @ git, + clickhouse-connect, dotenv
  .python-version  .gitignore  README.md
  src/ai_checker/
    __init__.py
    task.py                 # AICheckerTask  (was ai_checker_task.py)
    pipeline.py             # was ai_checker_pipeline.py
    query_validator.py
    connect_clickhouse.py
    prompts/{query_gen.md, answer_synthesis.md}
  evals/                    # was evals_kit/ai-checker/*
    ai-determination/ query-answers-question/ ... risk-assessment/
  data/pilot.jsonl
```

context-graph-evals mirrors this with `src/context_graph/{task,pipeline}.py`,
`prompts/`, `evals/` (was `evals_kit/context-graph/*`), `data/context_graph_pilot.jsonl`.

## Import rewrites needed (flat/example-style -> package)

- ai-checker: `from ai_checker_pipeline import ...` -> `from ai_checker.pipeline import ...`;
  `from query_validator import QueryValidator` -> `from ai_checker.query_validator import ...`;
  `from connect_clickhouse import get_client` -> `from ai_checker.connect_clickhouse import ...`.
- context-graph: `from examples.context_graph.context_graph_pipeline import ...`
  -> `from context_graph.pipeline import ...`. The `src.*` service imports are handled per
  the open decision above.
- Run via installed console entry or `--task ai_checker.task:AICheckerTask` (cwd on sys.path
  via llm-lab's task loader, or install the app editable).

## App pyproject dependency on llm-lab (git dep)

```toml
[project]
dependencies = [
  "llm-lab @ git+ssh://git@github.com/stephanie-nebulock/llm_lab.git@main",
  # ai-checker: "clickhouse-connect>=1.8.0", "python-dotenv>=1.2.3"
]

# Optional: editable local override for co-development (not committed to lock):
[tool.uv.sources]
llm-lab = { path = "../llm_lab", editable = true }   # or { git = "...", branch = "main" }
```

**Pull llm-lab changes down:** `uv lock --upgrade-package llm-lab && uv sync`.
Pin to a tag (`@v0.2.0`) instead of `@main` for reproducible app builds.

## Execution order (all local; nothing pushed)

1. Scaffold `~/repos/ai-checker-evals` and `~/repos/context-graph-evals` (dirs, pyproject,
   README, .gitignore, .python-version).
2. Copy app files in, rename to the package layout, rewrite internal imports.
3. Resolve the context-graph service-code decision (A/B/C).
4. `git init` + initial commit in each app; `uv sync` to verify llm-lab resolves and the
   task modules import.
5. Only after apps verified: remove the app-specific dirs from llm-lab
   (`examples/ai_checker`, `examples/context_graph`, `evals_kit/{ai-checker,context-graph}`)
   and drop the `clickhouse` extra; commit llm-lab.
6. (You) create the two GitHub repos under `stephanie-nebulock` and push. Commands to be
   provided.

## Notes / cleanup spotted

- `examples/__pycache__`, `results.jsonl`, `runs/` are stray artifacts in llm-lab — gitignore/remove.
- ai-checker references hunt-vault schema (`CLICKHOUSE_SECTION`) and org-id mappings that
  are hard-coded for the pilot; keep in the app.
