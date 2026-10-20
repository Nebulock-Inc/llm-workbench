# Contributing to llm-workbench

Thanks for your interest in improving `llm-workbench`. Bug reports, feature
suggestions, and pull requests are all welcome.

## Getting set up

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management, and requires Python 3.12+.

```bash
git clone https://github.com/Nebulock-Inc/llm-workbench.git
cd llm-workbench
uv sync --extra analysis --extra dev   # lib + analysis deps + dev tools (pytest, ruff)
```

Run commands through `uv run` so they use the project environment:

```bash
uv run llm-workbench --help
```

Enable the pre-commit hook so lint runs automatically before each commit (same
check as CI):

```bash
uv run pre-commit install
```

## Before you open a PR

Run the test suite and the linter:

```bash
uv run pytest          # tests run offline via the mock/ provider — no API keys needed
uv run ruff check .    # lint
uv run ruff format .   # format
```

- The tests drive everything through the `mock/` provider and synthetic data,
  so they need no credentials or network access. Please keep new tests that
  way where possible.
- Add or update tests for any behavior you change, and call out any
  intentional change to existing behavior in your PR description.

## Pull request guidelines

- Keep changes focused; one logical change per PR.
- Describe what changed and why. Link any related issue.
- Make sure `pytest` and `ruff` pass.

## Extending the library

`llm-workbench` is designed to be extended from your own project without
modifying the library. Common extension points:

- **Custom tasks** — a class with a `name` and `run(record, client) -> TaskOutput`
  for multi-step, agentic, or retrieval workflows. See
  [`examples/`](examples/) and
  [docs/complex-prompts-and-custom-tasks.md](docs/complex-prompts-and-custom-tasks.md).
- **Custom scorers** — register your own scoring logic beyond the built-in
  checks and LLM judge.
- **New model backends** — implement a client against the model interface and
  register it under a provider prefix.

If you are adding a new model backend or scorer that would be broadly useful,
a PR to bring it into the library itself is welcome — open an issue first to
discuss the approach.

## Reporting bugs and requesting features

Open an issue on GitHub. For security vulnerabilities, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
