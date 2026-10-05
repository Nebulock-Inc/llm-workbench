# Plan: make `register_scorer` a real, supported extension point

## Problem

`scoring/registry.py` exposes `register_scorer(type_name, factory)` to add a custom
scorer `type:` to `SCORER_FACTORIES`, but nothing ever calls it and there is no way for
a user to get their code to call it before a run. Scorer `type` strings are resolved by
`build_scorers` (`registry.py`), which runs only inside `run_comparison`
(`runner/comparison.py:42`). So a custom scorer is currently unreachable.

Goal: let a user define a scorer in their own module and reference it by `type:` in an
`eval.yaml`, mirroring how custom **tasks** already work via `--task module:Class`.

## The scorer contract (what a custom scorer must implement)

A scorer is any object satisfying the `Scorer` protocol (`scoring/base.py`):

- attribute `name: str`
- method `score(self, ctx: ScoreContext) -> ScoreResult`

`ScoreContext` gives it: `record` (input row dict), `output` (model output text),
`task_output` (full `TaskOutput`, incl. `Response` metadata like `latency_s`), and
`judge_client` (optional). `ScoreResult` fields: `scorer` (name), `score` (float),
plus optional `label`, `passed`, `reasoning`, `raw_response`, `error`.

A factory is `Callable[[ScorerSpec], Scorer]` — it receives the `ScorerSpec` from the
eval (`type`, `expected_field`, `params`) and returns a scorer instance. Built-in
scorers follow this exact shape (see `scoring/programmatic.py`, `operational.py`).

## Design decision: the load path

**Chosen approach: a `--scorer-module` CLI flag (repeatable) on `run compare`.**
Before evals are resolved, each named module is imported (with cwd on `sys.path`, same
as `load_task`), and the import side-effect calls `register_scorer(...)`. This mirrors
the existing `--task module:Class` ergonomics and needs no packaging/entry-point
machinery. (Python entry points remain a possible later addition for installed plugins;
out of scope here.)

Ergonomic sugar to add alongside: a `@scorer("type_name")` decorator in `registry.py`
that wraps `register_scorer`, so a user module reads:

```python
from llm_workbench.scoring.registry import scorer
from llm_workbench.scoring.base import ScoreResult

@scorer("keyword_density")
class KeywordDensity:
    name = "keyword_density"
    def __init__(self, spec):
        self.spec = spec
    def score(self, ctx):
        kw = self.spec.params["keyword"]
        n = ctx.output.lower().count(kw.lower())
        return ScoreResult(scorer=self.name, score=float(n))
```

## Implementation steps

1. **`scoring/registry.py`**
   - Keep `register_scorer`. Add an optional `overwrite: bool = False` guard so
     re-registering a built-in name raises (or warns) instead of silently shadowing it.
   - Add a `scorer(type_name)` decorator that calls `register_scorer` and returns the
     class unchanged.

2. **New loader `scoring/loader.py`** (parallels `tasks/loader.py`)
   - `load_scorer_modules(specs: Iterable[str]) -> None`: insert cwd into `sys.path`
     (once), `importlib.import_module` each spec so its `@scorer` / `register_scorer`
     side-effects run. Accept bare `module` (not `module:attr`) — importing the module
     is all that's needed.
   - Raise a clear error on `ImportError`.

3. **`cli/commands/run.py`**
   - Add `@click.option("--scorer-module", "scorer_modules", multiple=True, help=...)`.
   - Call `load_scorer_modules(scorer_modules)` **before** `_resolve_evals` /
     `run_comparison`, wrapped in try/except → `click.ClickException`.
   - Update the command epilog example to show a custom scorer.

4. **Validation (`cli/commands/evals.py::validate`)**
   - Today `validate` only calls `load_spec` and never checks scorer `type` strings, so
     an unknown/custom type is not flagged there — no regression. Decide whether to add
     a `--scorer-module` option to `validate` so it can optionally confirm that all
     referenced scorer types resolve in `SCORER_FACTORIES`. (Nice-to-have; can defer.)

5. **Error message quality**
   - `build_scorers` already raises `Unknown scorer type {type!r}. Known: ...` listing
     registered types. Confirm that message now includes custom types when a
     `--scorer-module` was loaded (it will, since they are in `SCORER_FACTORIES`), and
     that the "did you forget --scorer-module?" hint is added when the type is unknown.

## Tests (`tests/test_scoring.py` + maybe `tests/test_runner.py`)

- `register_scorer` + `build_scorers`: register a toy scorer, put its `type:` in a
  `ScorerSpec`, assert `build_scorers` returns it and `score()` runs.
- `scorer` decorator registers under the given name.
- `overwrite` guard: re-registering a built-in name without `overwrite=True` raises.
- `load_scorer_modules`: point it at a temp module file that registers a scorer; assert
  the type becomes resolvable. (Write the module into `tmp_path`, add to `sys.path`.)
- End-to-end (optional): a `run compare` with `--scorer-module` produces a results row
  carrying the custom scorer's score.

## Docs

- **`docs/eval-reference.md`**: new "Custom scorers" subsection under the scorers list —
  the `Scorer`/`ScoreContext`/`ScoreResult` contract, the `@scorer` decorator, and the
  `--scorer-module` flag. Show the `eval.yaml` `scorers:` entry referencing the custom
  `type:`.
- **`README.md`**: one line in the scorers area pointing to the reference, parallel to
  the existing custom-task pointer.
- Consider a short worked example under `examples/` (e.g. a keyword-density or
  regex-count scorer module) referenced from the docs.

## Out of scope (note for later)

- Python entry-point (`[project.entry-points."llm_workbench.scorers"]`) discovery for
  installed plugin packages.
- Per-eval scorer config schema validation beyond what `ScorerSpec` already does.
