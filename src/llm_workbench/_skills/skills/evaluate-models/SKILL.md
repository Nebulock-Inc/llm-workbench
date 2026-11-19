---
name: evaluate-models
description: Use to run a full llm-workbench model evaluation end to end — guide someone from "I want to know which model is best for my task" to a statistically grounded recommendation, sequencing setup, task, evals, pilot, power analysis, the full run, and analysis with a checkpoint between each. Triggers on "which model should I use", "compare / evaluate LLMs for my task", "pick the best model", "set up an LLM evaluation", "get started with llm-workbench", "run a model comparison from scratch".
---

# Evaluate models (the guide)

You're a mentor, not an autopilot. Walk the user from **"here's my task and some candidate
models"** to **"here's the model to use, and why"** — a recommendation grounded in reliability,
significance, effect size, and equivalence, not vibes. Explain the *why* at each step, do the
legwork (commands, files, drafts), and let the human make the calls.

This skill holds the **order and the gates**. Each step names a point skill that holds the
detail — load it (or read `../<skill>/SKILL.md`) and follow it.

## How to run it

- **One step at a time, with a checkpoint between steps.** Say what's done, what's next, and
  what the gate is. Keep a short running checklist in the conversation; rebuild it from the
  files (`.env`, dataset, `evals_kit/`, `runs/`) after any context reset.
- **Don't skip a gate.** Each gate exists because the next step trusts it (e.g. don't trust
  judge scores before calibration; don't size the study before a pilot).
- **Cost awareness.** The full run is the expensive step. Smoke-test with `mock/echo` and
  `--limit`, confirm the pilot and calibration look right, and only then run the full N.

## Intake

Ask, briefly:
- What's the task? (one prompt per record, or something with retrieval/scoping/multiple steps?)
- Which candidate models, and what will judge? Any cost/latency constraints?
- Do they already have prompts/data, or need to build a test set?
- How familiar are they with the statistics?

## The path (each arrow is a gate)

1. **`setup`** — pick extras, configure `.env`, verify.
   → *Gate:* `models ping` succeeds for every candidate and the judge; `mock/echo` smoke test runs clean.

2. **`define-task`** — prompt template (default) or a custom `--task`.
   → *Gate:* a `--prompt-template` file or an importable `--task module:Class`.

3. **`create-evals`** — rubric and/or checks.
   → *Gate:* `evals validate` passes.

4. **`build-dataset`** (pilot size) — ~10–20 representative records.
   → *Gate:* a pilot JSONL with stable `id`s.

5. **`run-comparison`** (pilot) — same models/evals/judge you'll use for the full run.
   → *Gate:* `runs/pilot/results.jsonl` covering every candidate.

6. **`create-evals` → calibrate the judge** on the pilot.
   → *Gate:* acceptable judge-vs-human bias/agreement, or the rubric revised until it is. If the
   rubric changed, re-run the pilot (step 5) before trusting scores.

7. **`power-analysis`** — size the study from the pilot.
   → *Gate:* a target N (recommended N for the smallest effect, plus a margin).

8. **`build-dataset`** (full) — grow the test set to N; reuse the dataset so resume applies.
   → *Gate:* a full JSONL at ~N records.

9. **`run-comparison`** (full) — the full sweep (reuse/grow the run dir; resume fills in).
   → *Gate:* `results.jsonl` at full N for every candidate.

10. **`analyze-results`** — the statistics and the recommendation.
    → *Deliverable:* a plain-language recommendation that reads reliability, significance,
    effect size, and equivalence together, and names any quality-vs-cost/reliability tradeoff.

## Shape it to the user

- Already has a validated eval and data? Jump to the pilot.
- Just exploring? Do steps 1–2 and a `mock/echo` run so they see the loop before spending on
  real models.
- Models obviously far apart in the pilot? Power analysis will recommend a small N — say so and
  keep the full run cheap.
