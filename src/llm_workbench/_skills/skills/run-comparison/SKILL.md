---
name: run-comparison
description: Use to run an llm-workbench model comparison — the model×record sweep that produces the scored results, for both a small pilot and the full study. Covers options, resume, smoke-testing, output files, and local-model memory. Triggers on "run the comparison", "run compare", "run a pilot", "compare these models", "resume a run", "grow my run", "my run crashed".
---

# Run the comparison

`run compare` is the core sweep: every model answers every record, and each eval scores each
pair. The same command shape serves the **pilot** (small dataset) and the **full study** (the
power-analysis N) — only the dataset changes.

Preconditions: the `setup` skill's `models ping` passes for all candidates and the judge; a
task (`--prompt-template` or `--task`) and validated evals exist.

```bash
llm-workbench run compare \
    --prompt-template my_prompt.txt \        # or --task module:Class (mutually exclusive)
    --dataset test_set.jsonl \
    --model ollama/llama3.1 --model anthropic/claude-sonnet-4-5 \   # repeat per candidate
    --eval answer-quality \                  # repeatable; omit = all evals in --eval-dir
    --judge-model anthropic/claude-sonnet-4-5 \   # enables LLM-judge scoring for rubric evals
    --out-dir runs/pilot
```

Key options (full list: `../../references/cli-cheatsheet.md`):
- `--models a,b,c` — comma-separated alternative to repeated `--model`.
- `--eval-dir DIR` — where evals are found (default `evals_kit`).
- `--judge-model` — **required for rubric scoring.** Without it, rubric evals are skipped (with
  a warning); programmatic checks still run.
- `--limit N` — only the first N records; use it to smoke-test before a full sweep.
- `--resume` (default) / `--no-resume`.

## The design (why it matters downstream)

The sweep is **model × record** — every model answers every prompt — which makes the analysis a
**paired** design (the unit is the per-prompt difference between models). Use the *same* models,
evals, and judge for the pilot as for the full run, so the measured effect is realistic.

## Resume and retries

Results stream crash-safe to `results.jsonl`. Re-running with the default `--resume` skips
`(model_id, record_id)` pairs already recorded — **including rows that errored**. So:
- To **grow** a run (more prompts, or add a model), reuse the same `--out-dir` and re-run;
  only the new pairs run.
- To **retry failures**, fix the cause, then use a fresh `--out-dir` or delete the run dir —
  resume will not re-run an errored row in place.
- Give records stable `id`s so resume and analysis line up across runs.

## Outputs

- `results.jsonl` — source of truth, one object per `(model, record)`: `output`, `prompt`,
  `latency_s`, `usage`, `error` (task-level), `scores` (per-scorer).
- `results.csv` — flattened to one row per `(model, record, scorer)` for spreadsheets/pandas.
  Task-level failures are dropped here; find them in `results.jsonl` `error` and the reliability
  table.
- End-of-run console tables: a comparison summary (per model/eval/scorer: n, mean, pass rate,
  errors) and per-model reliability (runs, task errors, mean latency).

## Local models and memory

If candidates and judge are all local Ollama models, a candidate and the local judge are
resident at once — stacking large ones can exhaust memory. Prefer an **API judge**, or set
`OLLAMA_MAX_LOADED_MODELS=1` + `OLLAMA_KEEP_ALIVE=0`, and smoke-test with `--limit 2` first.
See `../../references/model-ids-and-credentials.md`.

**Gate for the guide:** a run directory with `results.jsonl` covering every candidate — the
pilot before power analysis, the full N before final analysis.
