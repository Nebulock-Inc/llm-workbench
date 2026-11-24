# Installing with conda

The supported install path is uv (or plain pip into a virtualenv), as in the
[README](https://github.com/Nebulock-Inc/llm-workbench#install--setup). conda works too, but a few project
specifics trip people up.

## The short version

```bash
conda create -n llm-workbench python=3.12   # 3.12+ is required
conda activate llm-workbench
pip install "llm-workbench[analysis]"        # install the package with pip, not conda
```

Use conda for the environment and Python version, and use pip for the
package and its dependencies.

## Why these steps matter

**Create the env at Python 3.12+.** The package requires `python>=3.12`. If you
install into an env on an older Python (a 3.10/3.11 base env is common), pip
reports a confusing "no matching distribution" instead of a clear version error.
Set the Python version when you create the env, as above.

**Install the package with pip, not conda.** `llm-workbench` is published on PyPI,
not conda-forge, so `conda install llm-workbench` will not find it. Running pip
inside an activated conda env is the intended way to add it.

**Don't pre-`conda install` the analysis dependencies.** The `analysis` extra
pulls recent pins of pandas, scipy, statsmodels, and matplotlib. If you
`conda install pandas scipy ...` first and then pip-install the package, the two
resolvers fight over the same binary libraries and conda-forge may not yet carry
the pinned versions — this is the usual source of a broken analysis install. Let
pip resolve the `analysis` extra in one step instead.

**Versions are not pinned on this path.** The repo's `uv.lock` is only read by uv.
A conda + pip install resolves fresh from the lower bounds in `pyproject.toml`, so
you may get newer dependency versions than a uv user. That's fine for normal use;
just know that "works on my machine" differences between a uv user and a conda
user are expected, not a bug.

## Developing the project under conda

Working *on* llm-workbench (not just using it) is best done with uv, because the
project uses uv's build backend and lockfile:

- `pip install -e .` works, but pip must fetch the `uv_build` build backend during
  an isolated build (it fails offline or behind a restrictive proxy) and it will
  not honor `uv.lock`.
- The docs dependencies (mkdocs, mkdocstrings) live in a PEP 735 dependency group,
  which plain pip only installs via `pip install --group dev` on pip 25.1+. uv
  handles this automatically with `uv sync`.

If you're contributing, prefer `uv sync` over conda for a reproducible setup.
