#!/usr/bin/env python3
"""Sync the canonical skills into the Claude Code plugin tree.

The skills ship inside the package at ``src/llm_workbench/_skills`` so a PyPI install can
scaffold them with ``llm-workbench init --with-skills``. The marketplace plugin under
``plugins/llm-workbench`` needs committed copies of the same ``skills/`` and ``references/``
trees. This script copies the canonical tree into the plugin, so there is one source of truth.

Run it after editing anything under ``_skills``:

    python scripts/sync_skills.py          # write the plugin copy
    python scripts/sync_skills.py --check   # exit 1 if the plugin copy is stale (used in CI)

``tests/test_skills_sync.py`` runs the same check.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "src" / "llm_workbench" / "_skills"
PLUGIN = REPO_ROOT / "plugins" / "llm-workbench"
# Subtrees of _skills that are mirrored verbatim into the plugin root.
SUBTREES = ("skills", "references")


def _diff(src: Path, dst: Path) -> list[str]:
    """Return a list of human-readable differences between two trees (empty = in sync)."""
    cmp = filecmp.dircmp(src, dst)
    problems = [f"only in canonical: {src / n}" for n in cmp.left_only]
    problems += [f"only in plugin: {dst / n}" for n in cmp.right_only]
    problems += [f"differs: {dst / n}" for n in cmp.diff_files]
    for sub in cmp.common_dirs:
        problems += _diff(src / sub, dst / sub)
    return problems


def check() -> int:
    problems: list[str] = []
    for sub in SUBTREES:
        src, dst = CANONICAL / sub, PLUGIN / sub
        if not dst.exists():
            problems.append(f"missing in plugin: {dst}")
            continue
        problems += _diff(src, dst)
    if problems:
        print("Plugin skills are out of sync with src/llm_workbench/_skills:")
        for p in problems:
            print(f"  - {p}")
        print("Run: python scripts/sync_skills.py")
        return 1
    return 0


def write() -> int:
    for sub in SUBTREES:
        src, dst = CANONICAL / sub, PLUGIN / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"synced {sub}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only check; do not write.")
    args = parser.parse_args()
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
