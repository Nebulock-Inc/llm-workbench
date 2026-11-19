---
name: setup
description: Use to install and connect llm-workbench — pick the right install extras for the models someone wants to compare, configure provider credentials in a .env, and verify connectivity before any run. Triggers on "set up llm-workbench", "install llm-workbench", "which extras do I need", "connect my models", "configure my API keys", "models ping fails", "ANTHROPIC_API_KEY / OPENAI_API_KEY / Bedrock / Ollama not working".
---

# Setup: install and connect

Goal: every candidate model **and** the judge model respond to `llm-workbench models ping`
before anyone runs a comparison. Work in this order.

Read `../../references/model-ids-and-credentials.md` — it has the full provider table, extras,
env vars, and local-model memory guidance. This skill is the procedure.

## 1. Pick the extras from the models in play

Ask which models they want to compare and which will judge, then map prefixes → extras:

- `[analysis]` — needed by `power` and `analyze`. **Almost everyone needs it**; install it now.
- `[openai]` — for any `openai/` id **and** any `bedrock-mantle/` id.
- `[anthropic]` — for the native `anthropic/` client.
- base install already covers `ollama/` and `bedrock/`.

```bash
uv add "llm-workbench[analysis]"                 # add ,openai and/or ,anthropic as needed
```

## 2. Configure credentials in a `.env`

A local `.env` (in the project root) is auto-loaded. Only set the vars for providers in use:

- Ollama: `OLLAMA_BASE_URL` if not the default `http://localhost:11434`.
- OpenAI-compatible: `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for a non-OpenAI endpoint).
- Anthropic: `ANTHROPIC_API_KEY`.
- Bedrock / Mantle: `AWS_PROFILE`, `AWS_REGION`.

Never commit `.env` — confirm it is gitignored. Do not print secret values back to the user.

## 3. Verify

```bash
llm-workbench models list                 # confirms providers are registered
llm-workbench models ping <id>            # one per candidate AND the judge — must all succeed
```

If a ping fails: missing extra (ImportError on the SDK), missing/incorrect credential, wrong
region/profile for Bedrock, or Ollama not running / model not pulled (`ollama pull <model>`).

## 4. Zero-credential smoke test

Prove the pipeline runs end to end without any keys using the `mock/echo` provider:

```bash
llm-workbench init                        # scaffolds prompt.txt, dataset.jsonl, evals_kit/
llm-workbench run compare \
    --prompt-template prompt.txt --dataset dataset.jsonl \
    --model mock/echo --eval example-helpfulness --out-dir runs/example
```

`mock/echo` just echoes the prompt (no judge, so no scores) — a green run here means wiring is
correct and the only remaining variables are credentials and the models themselves.

## Local models and memory

If candidates and judge are **all local Ollama models**, warn about memory before a big sweep
and prefer an API judge or `OLLAMA_MAX_LOADED_MODELS=1` + `OLLAMA_KEEP_ALIVE=0`. See the
reference's "Running local models — memory".

**Gate for the guide:** `models ping` succeeds for every candidate and the judge, and the
`mock/echo` smoke test runs clean.
