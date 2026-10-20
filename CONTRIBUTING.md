# Contributing to llm-workbench

Thanks for your interest in improving `llm-workbench`. Bug reports and feature
suggestions are welcome.


All reports and requests go through GitHub issues. Before opening one, please
search existing issues to avoid duplicates.

> For security vulnerabilities, do **not** open a public issue — follow
> [SECURITY.md](SECURITY.md) instead.

## Reporting a bug

Open a new issue and include as much of the following as you can. A clear,
reproducible report is far faster to act on:

- **What you ran** — the exact `llm-workbench` command (or code), including the
  prompt template / task and eval involved.
- **What you expected vs. what happened** — the actual result or error, and what
  you expected instead.
- **The full error output** — the complete traceback or message, not just the
  last line.
- **The model(s) involved** — the `provider/model` id(s) (e.g. `ollama/llama3.1`,
  `anthropic/claude-sonnet-4-5`), since behavior often depends on the backend.
- **Your environment** — output of `llm-workbench --version`, your Python
  version, and your OS.
- **A minimal reproduction** — the smallest dataset, eval, and command that
  still shows the problem, if you can narrow it down.

## Requesting a feature

Open an issue describing the problem you're trying to solve, not just a proposed
solution — what you're trying to do, why the current behavior falls short, and
any workaround you're using now. That context makes it much easier to evaluate
the request and suggest alternatives.
