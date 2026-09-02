# Find the LLM You Need


This library, `llm_lab`, gives you a toolkit to do all the tasks involved in running experiments testing different LLM options on your particular use case.

* Creating evals
* Running a small pilot study
* Running power analysis to know how many prompts you need
* Creating test set of prompts
* Running the experiment, collecting prompt/response pairs using LLMs from Anthropic, OpenAI, Ollama, or AWS Bedrock (including Bedrock Mantle)
* Analyzing the results, selecting the best model for your use case


Read on to learn how to use it for all these steps.

***

# Introduction

The purpose of this library is to help you decide what LLM is the right choice for your task. The process is to define your task (what your LLM needs to do), then create evals that will assess the measures of quality that matter for your project (examples provided in the library). After you've done this, you'll develop a pool of prompts that resemble what your LLM will be asked to do in production, and you'll need to create enough to achieve statistical significance. (That number is determined by the power analysis, which also has a tool here.)

After you create your prompt test set and your evals, you're ready to run the experiment on however many LLMs you want to test. Each LLM will be asked each prompt, and each eval will run on that prompt/response pair, producing scores. These scores are the raw material used for the result analysis.

In the result analysis, depending on the data size and quality, you'll get statistical output indicating which model works the best (if that can be determined).

# Install & Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url> llm_lab && cd llm_lab
uv sync                      # base install (Ollama + Bedrock work out of the box)
uv sync --extra openai       # add the OpenAI-compatible + Bedrock Mantle clients
uv sync --extra anthropic    # add the native Anthropic client
uv sync --extra analysis     # needed for `power` and `analyze` (scipy/statsmodels/matplotlib)
```

Everything is driven by one CLI, `llm-lab`:

```bash
uv run llm-lab --help
```

The workflow follows the sections below: **create evals → run a small pilot → size the study with power analysis → build the full test set → run the comparison → analyze the results.**

# Eval Creation

An *eval* is a folder (`eval.yaml` + `prompt.md`) describing how to score a model's output — a scoring **rubric** (an LLM judge picks one level) and/or deterministic **checks**. Evals live in `evals_kit/` by default.

Create one with the guided wizard — it walks you through the rubric and the judge prompt:

```bash
uv run llm-lab evals new
```

It asks for a name, description, a rubric (a 1-5 or binary preset you can edit, or a fully custom scale), which `{variables}` the judge reads (usually `input` = the question and `output` = the model's answer), and any programmatic checks. It previews the files and validates before saving. To skip the prompts (e.g. in CI):

```bash
uv run llm-lab evals new answer-quality --labels 1-5 --yes
```

Manage existing evals:

```bash
uv run llm-lab evals list                 # table of evals, their scales, and variables
uv run llm-lab evals validate             # check every eval in evals_kit/ parses
```

**Scoring options** an eval can use:
- **LLM judge** — a rubric of `label: score` pairs; a judge model reads the prompt and picks one level.
- **Programmatic checks** — deterministic, no model needed: `exact_match`, `contains`, `regex`, `json_schema`, `numeric_tolerance`.
- **Operational** — `latency`, `cost` thresholds read from each call's metadata.

For guidance on *what* evals to create and how to design good rubrics, see the FAQ.

# Pilot Study

In order to know how large your full experiment needs to be for statistical significance, you'll need to run a small pilot test first.

The number of prompts you need depends on how *different* your candidate models actually are on your task — a gap you can't know in advance. If they're far apart, a handful of prompts settles it; if they're close, you need many more to tell them apart reliably. A pilot measures that gap in miniature, and the power analysis (next section) turns it into a concrete target sample size.

**1. Make a small, representative prompt set.** Hand-write or generate ~10–20 prompts that resemble production traffic (see [Prompt Test Set Creation](#prompt-test-set-creation) for the JSONL format and the `dataset` builders). It doesn't need to be large — only representative enough to estimate the models' relative quality.

**2. Run the comparison on it**, using the same models, evals, and judge you intend to use for the full study, so the measured effect is realistic:

```bash
uv run llm-lab run compare \
    --prompt-template my_prompt.txt \
    --dataset pilot.jsonl \
    --model ollama/llama3.1 --model anthropic/claude-sonnet-4-5 \
    --eval answer-quality \
    --judge-model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \
    --out-dir runs/pilot
```

This writes `runs/pilot/results.jsonl`, which is the input to `llm-lab power --from-run` below. (Runs resume, so you can also grow the pilot later by re-running with more prompts — new prompts are appended, already-scored ones are skipped.)

A pilot is worth the cost because it uses *your* task and *your* evals; a sample size guessed from a generic effect size can be off by an order of magnitude. If you genuinely can't run one, the power tool also accepts an assumed effect size directly (see below).

# Power Analysis - How Many Prompts?

Because every prompt is run on every model, this is a **paired (repeated-measures)** design: the unit is the per-prompt difference between models. `llm-lab power` turns a difference of a given size (effect size `dz` = mean difference / SD of differences) into the number of prompts needed to detect it reliably. Needs the `analysis` extra.

**From your pilot run** (recommended) — size the full study from the effects you actually observed:

```bash
uv run llm-lab power --from-run runs/pilot --reference ollama/llama3.1
```

This reports, per model vs. the reference, the observed effect, the power your pilot already achieved, and the prompts needed to reach your target power — plus a recommended N for the *smallest* (hardest to detect) effect, since that gap sets the sample size for the whole study. Tune with `--alpha` (default 0.05), `--power` (default 0.8), and `--alternative`.

**Without a pilot** — if you can only assume an effect size, size it directly:

```bash
uv run llm-lab power --effect-size 0.5              # -> N prompts at alpha=0.05, power=0.8
uv run llm-lab power --mean-diff 0.4 --sd-diff 0.8  # dz = 0.4 / 0.8 = 0.5, so same N
```

> Sample sizes use the paired t-test; the Wilcoxon test used in `analyze` is ~95% as efficient, so add a small margin if you rely on it.

# Prompt Test Set Creation

Your test set is a **JSONL** file — one JSON object per line. `input` is the standard field the default task fills into your prompt template; add `expected` (used by checks like `exact_match`) and any other fields your task needs. An `id` gives each record a stable identity (used for resume); if omitted, one is generated.

```json
{"id": "q1", "input": "What hunts cover LaunchAgent persistence?", "expected": "..."}
{"id": "q2", "input": "Summarize the last 24h of failed logins for acme-corp."}
```

Author it by hand, or build it:

**From a spreadsheet** (every column becomes a field):

```bash
uv run llm-lab dataset from-csv prompts.csv -o test_set.jsonl --id-col qid
```

**Generated by an LLM** (a starter set to review and edit):

```bash
uv run llm-lab dataset generate --model anthropic/claude-sonnet-4-5 \
    --instructions "security-analyst questions about threat hunts and detections" \
    -n 30 --with-expected -o test_set.jsonl
```

# Running the Experiment

Point `run compare` at your dataset, list the candidate models, and choose the eval(s) to score with. Supply `--judge-model` to enable LLM-judge scoring.

```bash
uv run llm-lab run compare \
    --prompt-template my_prompt.txt \
    --dataset test_set.jsonl \
    --model ollama/llama3.1 --model openai/gpt-4o-mini --model anthropic/claude-sonnet-4-5 \
    --eval answer-quality \
    --judge-model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 \
    --out-dir runs/experiment-1
```

`--prompt-template` is a file containing `{input}` (and any other dataset fields) for the default one-prompt-per-row task. Results stream to `runs/experiment-1/results.jsonl` (crash-safe) and flatten to `results.csv`; re-running resumes where it left off (`--no-resume` to disable). Use `--limit N` for a quick pilot.

### `run compare` options

| Option | Required | Default | What it does |
|---|---|---|---|
| `--dataset FILE` | **yes** | — | JSONL test set; one record per line. `input` is filled into the prompt template; other fields are available to custom tasks and eval `{variables}`. |
| `--task module:Class` | one of `--task`/`--prompt-template` | — | Custom task (`run(record, client) -> TaskOutput`) for multi-step pipelines, scoping, or retrieval. Imported from the current directory (e.g. `examples.my_task:MyTask`). See [Complex Prompts and Custom Tasks](docs/complex-prompts-and-custom-tasks.md). |
| `--prompt-template FILE` | one of `--task`/`--prompt-template` | — | Template file for the default one-prompt-per-row task; contains `{input}` and any other dataset fields. Mutually exclusive with `--task`. |
| `--model provider/id` | **yes** (≥1) | — | A candidate model to test. Repeat the flag for each model (`--model a --model b`). See [Connecting Your LLMs](#connecting-your-llms) for id formats. |
| `--models a,b,c` | — | — | Comma-separated model ids, as an alternative or supplement to repeated `--model`. |
| `--eval NAME` | — | all evals in `--eval-dir` | An eval to score with: a folder name under `--eval-dir`, or a path to an eval directory. Repeatable. Omit to run every eval found in `--eval-dir`. |
| `--eval-dir DIR` | — | `evals_kit` | Directory searched for evals (each is a subfolder with `eval.yaml` + `prompt.md`). |
| `--judge-model provider/id` | — | none | Enables LLM-judge scoring for evals that have a rubric. Without it, rubric evals are skipped (a warning is printed); programmatic checks still run. |
| `--out-dir DIR` | — | `runs/<timestamp>` | Where `results.jsonl` / `results.csv` are written. Reuse a directory to resume or grow a run. |
| `--resume` / `--no-resume` | — | `--resume` | Skip `(model, record)` pairs already in the run's `results.jsonl`. `--no-resume` re-runs everything (appends, so prefer a fresh `--out-dir`). |
| `--limit N` | — | all records | Only the first N records — handy for a smoke test before a full sweep. |

Notes:
- Provide exactly one of `--task` or `--prompt-template`, and at least one `--model`.
- The sweep is **model × record**; every model answers every prompt and each eval scores each pair, which is what makes the [result analysis](#result-analysis) a paired design.
- Resume keys on the record's `id` field (or a generated hash if absent) — give records stable `id`s so resume and analysis line up across runs.

**Prompts with several inputs, org/tenant scoping, retrieval, or a multi-step pipeline?** Multi-variable templates work out of the box, and anything needing extra steps is a custom `--task` — see [Complex Prompts and Custom Tasks](docs/complex-prompts-and-custom-tasks.md).

## Connecting Your LLMs

Every model is referenced by a single id string, `provider/model`. The prefix selects the client:

| Prefix | Client | Example id | Needs extra |
|---|---|---|---|
| `ollama/` | local Ollama | `ollama/llama3.1` | — |
| `openai/` | any OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, vLLM, LM Studio, Azure) | `openai/gpt-4o-mini` | `openai` |
| `anthropic/` | native Anthropic API | `anthropic/claude-sonnet-4-5` | `anthropic` |
| `bedrock/` | AWS Bedrock (Claude / Llama / Mistral) | `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0` | — |
| `bedrock-mantle/` | Bedrock Mantle OpenAI-Responses (e.g. Gemma) | `bedrock-mantle/google.gemma-4-31b` | `openai` |

Credentials and endpoints come from the environment (a local `.env` is loaded automatically):

- **Ollama**: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_THINK=false` to disable reasoning-model thinking, `OLLAMA_KEEP_ALIVE`.
- **OpenAI-compatible**: `OPENAI_API_KEY`, and `OPENAI_BASE_URL` to point at a non-OpenAI endpoint.
- **Anthropic**: `ANTHROPIC_API_KEY`.
- **Bedrock / Mantle**: `AWS_PROFILE`, `AWS_REGION` (Mantle is SigV4-signed with your profile — no separate key).

Check providers and test connectivity:

```bash
uv run llm-lab models list                 # registered providers + example ids
uv run llm-lab models ping ollama/llama3.1  # one short call, prints latency/result
```

## Result Analysis

`llm-lab analyze` runs the statistics appropriate to your data and design (paired on prompt). Needs the `analysis` extra.

```bash
uv run llm-lab analyze runs/experiment-1 --reference anthropic/claude-sonnet-4-5
```

For each metric and an overall quality composite it reports, where appropriate:

- **Nonparametric**: Friedman test + Kendall's W (omnibus across ≥3 models), with Holm-corrected pairwise Wilcoxon signed-rank post-hocs.
- **Mixed-effects model**: `score ~ model` with a random intercept per prompt — each model's difference from the reference, with significance. (Graded metrics only; skipped for binary pass/fail.)
- **Paired Cohen's d**: practical effect-size magnitude for each model pair.
- **TOST equivalence**: whether two models are *statistically the same* within a margin (`--margin`) — something significance tests can't tell you.

It prints rich tables with plain-language interpretation and writes `analysis.json`, `analysis.md`, and comparison plots into the run directory. Filter with `--eval` / `--scorer`, and set `--alpha` / `--margin` as needed.




# FAQ
* How do I decide what evals to create, and how to design the rubrics?
A: Check out my other project, Customizing LLM Evaluations: https://github.com/skirmer/customizing-llm-evals

