# Model ids, providers, extras, and credentials

Every model in llm-workbench is one string: `provider/model`. The prefix picks the client.

| Prefix | Client | Example id | Needs extra |
|---|---|---|---|
| `ollama/` | local Ollama | `ollama/llama3.1` | — (base install) |
| `openai/` | any OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, vLLM, LM Studio, Azure) | `openai/gpt-4o-mini` | `openai` |
| `anthropic/` | native Anthropic API | `anthropic/claude-sonnet-4-5` | `anthropic` |
| `bedrock/` | AWS Bedrock (Claude / Llama / Mistral) | `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0` | — (base install) |
| `bedrock-mantle/` | Bedrock Mantle OpenAI-Responses (e.g. Gemma) | `bedrock-mantle/google.gemma-4-31b` | `openai` |
| `mock/` | test stub that echoes the prompt; ignores params; no credentials | `mock/echo` | — |

## Install extras

Base install ships the Ollama and Bedrock clients. Add the extras you need:

```bash
uv add "llm-workbench[analysis]"            # pandas/scipy/statsmodels/matplotlib — needed by `power` and `analyze`; most users want this
uv add "llm-workbench[openai]"              # for openai/ AND bedrock-mantle/ ids
uv add "llm-workbench[anthropic]"           # for the native anthropic/ client
uv add "llm-workbench[analysis,anthropic]"  # combine as needed
```

(`pip install "llm-workbench[...]"` works the same.)

## Credentials (environment variables; a local `.env` is auto-loaded from cwd or above)

- **Ollama:** `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_THINK=false` to
  disable reasoning-model thinking, `OLLAMA_KEEP_ALIVE`, `OLLAMA_MAX_LOADED_MODELS`.
- **OpenAI-compatible:** `OPENAI_API_KEY`, and `OPENAI_BASE_URL` to point at a non-OpenAI
  endpoint (OpenRouter, Together, vLLM, LM Studio, Azure).
- **Anthropic:** `ANTHROPIC_API_KEY`.
- **Bedrock / Mantle:** `AWS_PROFILE`, `AWS_REGION` (or `AWS_DEFAULT_REGION`). Mantle is
  SigV4-signed with your profile — no separate key.
- **Plugins:** `LLM_WORKBENCH_PROVIDERS` — comma/space-separated custom provider modules (or
  pass `--provider-module` on the CLI).

A `mock/echo` run needs no credentials at all — use it to prove the pipeline runs.

## Per-provider request-field support

`GenerationRequest` carries `system`, `temperature`, `stop`; `max_tokens` is honored
everywhere. All providers honor all three **except** `bedrock-mantle/`, whose OpenAI-Responses
API has no stop-sequence parameter — setting `stop` there is ignored and warns. (`system` maps
to `instructions` on Mantle.)

## Verify connectivity

```bash
llm-workbench models list                 # registered providers + example ids
llm-workbench models ping ollama/llama3.1  # one short call; prints latency/result
```

## Running local models — memory

`run compare` loads one model at a time, so there's no limit on how many you compare. API
models (Anthropic/OpenAI/Bedrock) run remotely and put no load on your machine. The risk is
**local Ollama models**: Ollama keeps recently-used models resident, and each candidate's
answer is judged right after it's produced, so a local candidate and a **local judge** are
loaded at once. Stacking several large local models can exhaust RAM/VRAM and, on Apple Silicon
(shared GPU memory), crash the OS. If candidates and judge are all local and memory is tight:

- Use an **API model as the judge** (e.g. `--judge-model anthropic/claude-sonnet-4-5`) so only
  one heavy model is ever loaded — least fragile, often a better grader.
- Or cap residency: `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_KEEP_ALIVE=0` (slower, but stays
  within memory).
- Smoke-test with `--limit 2` and add models incrementally — resume fills in the rest.

Full detail: the "Connecting Your LLMs" section of the README
(https://github.com/Nebulock-Inc/llm-workbench#readme).
