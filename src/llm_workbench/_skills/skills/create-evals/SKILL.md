---
name: create-evals
description: Use to create, design, and validate llm-workbench evals (how a model's output is scored — an LLM-judge rubric and/or deterministic checks), and to calibrate the judge against a human before trusting its scores. Triggers on "create an eval", "design a rubric", "add a scorer / check", "validate my evals", "exact_match / contains / regex / json_schema / latency check", "calibrate the judge", "do I trust the judge", "judge vs human agreement".
---

# Create evals and calibrate the judge

An **eval** is a folder (`eval.yaml` + `prompt.md`) describing how to score an output: a scoring
**rubric** (an LLM judge picks one level) and/or deterministic **checks**. Evals live in
`evals_kit/` by default.

## Create one

Guided wizard (walks through the rubric, judge prompt, and any checks; previews and validates
before saving):
```bash
llm-workbench evals new
```
Non-interactive (CI):
```bash
llm-workbench evals new answer-quality --labels 1-5 --yes
```

## The `eval.yaml` contract

Only these keys are allowed — any other is rejected. An eval must define a judge
(`labels` + a prompt) **and/or** a non-empty `scorers` list.

- `name` (required), `description`, `direction` (`MAXIMIZE`/`MINIMIZE`/`NONE`, default MAXIMIZE).
- `labels` — the judge rubric: `label: score` pairs. Presence of `labels` + a prompt turns on
  the LLM judge. Graded (e.g. 1–5) or binary (0/1).
- `prompt` (inline) **or** `prompt_file` (default `prompt.md`) — the judge prompt template.
- `scorers` — programmatic/operational checks (below).

**Judge prompt variables** are filled from, in order: record fields → the task's `aux` dict →
`output` (the model's answer). So a prompt can use `{input}`, `{output}`, and anything a custom
task put in `aux`. The judge replies `SCORE: <n>` / `REASONING: <text>`; the parser snaps the
score to the nearest legal rubric score and records the label. A judge reports a graded score,
**not** pass/fail.

## Scorers (deterministic; no model needed)

| `type` | Config | Passes when |
|---|---|---|
| `exact_match` | `expected_field` | `output.strip()` equals the field value |
| `contains` | `params.substring` (or `expected_field`), `params.case_sensitive` | substring is in `output` |
| `regex` | `params.pattern`, `params.ignore_case` | pattern matches `output` |
| `json_schema` | `params.schema` (optional) | `output` is JSON (and validates against schema) |
| `numeric_tolerance` | `expected_field`, `params.atol`/`params.rtol` | first number in `output` is within tolerance |
| `latency` | `params.max_s` | `score` = call latency; `passed` = `latency ≤ max_s` |

Full field reference + examples:
https://github.com/Nebulock-Inc/llm-workbench/blob/main/docs/eval-reference.md

## Designing good rubrics

- Prefer a **graded** scale when quality is a spectrum (1–5); use **binary** only for genuinely
  pass/fail properties. Each label should be a concrete, mutually-exclusive description a human
  could apply consistently.
- Keep the judge prompt specific about what to weigh; reference the exact variables it reads.
- For *what* evals to build and deeper rubric design, see the README FAQ and the companion
  project: https://github.com/skirmer/customizing-llm-evals

## Manage and validate

```bash
llm-workbench evals list        # evals, their scales, and the variables each expects
llm-workbench evals validate    # every eval in evals_kit/ must parse — do this before running
```

## Calibrate the judge (before trusting its scores)

After a pilot run, check the judge agrees with a human on the same rubric:
```bash
llm-workbench calibrate runs/pilot            # interactive review; resumable
llm-workbench calibrate runs/pilot --stats    # just the calibration numbers
```
It reports, per eval: **bias** (mean judge − mean human; positive = judge scores higher) and
exact / within-1 agreement. A large bias or low agreement means the rubric or judge prompt is
ambiguous — **revise it and re-run before drawing conclusions**. See
`../../references/stats-interpretation.md`. (A custom `--task` may expose `review_context()` to
show live context while you score.)

**Gate for the guide:** `evals validate` passes, and judge calibration shows acceptable bias /
agreement (or the rubric was revised until it does).
