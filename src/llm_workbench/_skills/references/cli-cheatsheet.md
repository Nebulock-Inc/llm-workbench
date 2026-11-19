# llm-workbench CLI cheatsheet

One CLI drives everything: `llm-workbench <command>`. Inside a `uv`-managed project, use
`uv run llm-workbench <command>`. Global options `--provider-module` / `--scorer-module`
(repeatable) load custom plugins.

## `init` — scaffold a starter
```bash
llm-workbench init [DIR]              # prompt.txt, dataset.jsonl, evals_kit/example-helpfulness/
llm-workbench init --with-skills      # also writes Claude Code skills into .claude/skills/
llm-workbench init --force            # overwrite existing files
```

## `models` — providers and connectivity
```bash
llm-workbench models list             # registered providers + example ids
llm-workbench models ping <id>        # one short call; prints latency/result
```

## `dataset` — build a test set (JSONL)
```bash
llm-workbench dataset from-csv prompts.csv -o test_set.jsonl --id-col qid
llm-workbench dataset generate --model <id> --instructions "..." -n 30 --with-expected -o test_set.jsonl
```

## `evals` — scoring definitions
```bash
llm-workbench evals new                       # guided wizard
llm-workbench evals new answer-quality --labels 1-5 --yes   # non-interactive (CI)
llm-workbench evals list                      # table of evals, scales, variables
llm-workbench evals validate [--dir evals_kit]  # check every eval parses
```

## `run compare` — the experiment sweep (model × record)
```bash
llm-workbench run compare \
    --prompt-template prompt.txt \        # OR --task module:Class (mutually exclusive)
    --dataset test_set.jsonl \
    --model <id> --model <id> \           # repeat per candidate (or --models a,b,c)
    --eval <name> \                       # repeatable; omit = all evals in --eval-dir
    --judge-model <id> \                  # enables LLM-judge scoring for rubric evals
    --out-dir runs/experiment-1
# --eval-dir DIR (default evals_kit), --limit N (smoke test), --resume/--no-resume (default --resume)
```
Resume keys on `(model_id, record_id)` incl. errored rows; give records stable `id`s.
Outputs: `results.jsonl` (source of truth), `results.csv`, end-of-run summary + reliability tables.

## `power` — sample size (needs the `analysis` extra)
```bash
llm-workbench power --from-run runs/pilot --reference <model>   # size from a pilot (recommended)
llm-workbench power --effect-size 0.5                           # or assume dz directly
llm-workbench power --mean-diff 0.4 --sd-diff 0.8               # dz = 0.4/0.8 = 0.5
# --alpha (0.05), --power (0.8), --alternative
```

## `analyze` — statistics on a run (needs the `analysis` extra)
```bash
llm-workbench analyze runs/experiment-1 --reference <model>
# --eval / --scorer (restrict), --exclude-metric, --alpha (0.05), --margin (0.2 TOST),
# --out-dir, --plots/--no-plots, --verbose/--simple
```
Writes `analysis.json`, `analysis.md`, and comparison plots.

## `calibrate` — judge vs. human (interactive)
```bash
llm-workbench calibrate runs/experiment-1
llm-workbench calibrate runs/experiment-1 --stats        # stats only, no review
# --eval / --model / --scorer (default scorer: judge), --re-review, --review-file, --task, --out-dir, --report/--no-report
```
Writes your scores to `<run>/review.jsonl`; calibration to `calibration.json` / `calibration.md`.
