# Eval Reference: `eval.yaml` + scorers

An eval is a folder with `eval.yaml` (how to score) and, for an LLM judge, `prompt.md`
(the rubric prompt). This is the field-by-field reference; for *what* evals to write
and how to design rubrics, see the FAQ link in the README.

## `eval.yaml` keys

Only the keys below are allowed — **any other key is rejected** (this keeps the format
tight; trace-query metadata from other eval tools must be stripped). An eval must
define a judge (`labels` + a prompt) and/or a non-empty `scorers` list.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | — (**required**) | Display name of the eval. |
| `description` | string | `""` | One-line description (shown in `evals list`). |
| `labels` | map `label: score` | `{}` | The judge rubric: each label maps to a numeric score. Presence of `labels` (+ a prompt) turns on the LLM judge. Use a graded scale (e.g. `1`–`5`) or binary (`0`/`1`). |
| `prompt` | string | `""` | Inline judge prompt template. Use this **or** `prompt_file`, not both. |
| `prompt_file` | path | `prompt.md` | File holding the judge prompt template (relative to the eval folder). If neither `prompt` nor `prompt_file` is set, a sibling `prompt.md` is used when present. |
| `direction` | `MAXIMIZE`/`MINIMIZE`/`NONE` | `MAXIMIZE` | Whether higher scores are better — used by the analysis. |
| `use_structured_output` | bool | `false` | Reserved flag for structured-output judging; the default judge parses `SCORE:`/`REASONING:` text. |
| `scorers` | list | `[]` | Programmatic / operational scorers (see below). Each entry: `type`, optional `expected_field`, optional `params`. |

### The judge prompt

The prompt template's `{variables}` are filled from three sources, merged in order:
**record fields → the task's `aux` dict → `output`** (the model's answer). So a prompt
can reference `{input}` (a record field), `{output}`, and anything a custom task puts
in `aux` (e.g. `{queries}`, `{query_results}`). `evals list` shows the variables each
eval expects; any variable nothing provides is left as the literal `{name}`.

At scoring time the judge model receives the filled prompt plus the rubric and is asked
to reply `SCORE: <n>` / `REASONING: <text>`. The parser reads that score, **snaps it to
the nearest legal rubric score**, and records the matching `label`. The judge reports a
graded `score` (and `label`), **not** a `passed` flag — pass rates come only from the
scorers below.

## Scorers (`scorers:` list)

Each scorer is `{type: <name>, expected_field: <record key>, params: {...}}`. `score` is
`1.0`/`0.0` for a check (or the measured value for operational scorers); `passed`
mirrors it. `expected_field` names a record field holding the ground truth.

| `type` | Config | Passes when |
|---|---|---|
| `exact_match` | `expected_field` | `output.strip()` equals the expected field's value. |
| `contains` | `params.substring` (or `expected_field`); `params.case_sensitive` (default false) | the substring appears in `output`. |
| `regex` | `params.pattern`; `params.ignore_case` | the pattern matches `output`. |
| `json_schema` | `params.schema` (optional JSON Schema) | `output` parses as JSON (and validates against `schema` if given). |
| `numeric_tolerance` | `expected_field`; `params.atol`, `params.rtol` | the first number in `output` is within `atol + rtol·|expected|` of the expected value. |
| `latency` | `params.max_s` (optional) | `score` = call latency in seconds; `passed` = `latency ≤ max_s` when `max_s` is set (needs `TaskOutput.response`). |

## Examples

Judge-only, graded 1–5 (rubric in `prompt.md`):

```yaml
name: Answer Quality
description: How well the answer addresses the question
direction: MAXIMIZE
labels:
  Fully and accurately answers the question.: 5
  Answers with minor gaps.: 4
  Partially answers.: 3
  Barely answers.: 2
  Does not answer.: 1
prompt_file: prompt.md
```

Programmatic checks only (no judge, so no `--judge-model` needed):

```yaml
name: Format Checks
description: Output must contain the ticket id and be valid JSON
scorers:
  - type: contains
    params: {substring: "TICKET-"}
  - type: json_schema
    params:
      schema:
        type: object
        required: [id, status]
  - type: numeric_tolerance
    expected_field: expected_count
    params: {atol: 0}
```

Judge **and** operational threshold together:

```yaml
name: Quality Under Latency Budget
direction: MAXIMIZE
labels: {Good: 1, Bad: 0}
prompt_file: prompt.md
scorers:
  - type: latency
    params: {max_s: 8}
```

Validate before running:

```bash
uv run llm-workbench evals validate path/to/eval        # one eval
uv run llm-workbench evals validate --dir evals_kit     # every eval in a directory
```
