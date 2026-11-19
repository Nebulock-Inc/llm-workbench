"""The plugin skill tree must stay a faithful copy of the canonical one, and the skills
themselves must be well-formed (frontmatter name matches the directory, reference links
resolve). The canonical tree ships in the wheel (scaffolded by `init --with-skills`); the
plugin copy is what the marketplace serves. `scripts/sync_skills.py` keeps them identical.
"""

from __future__ import annotations

import filecmp
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "src" / "llm_workbench" / "_skills"
PLUGIN = REPO_ROOT / "plugins" / "llm-workbench"
SUBTREES = ("skills", "references")

EXPECTED_SKILLS = {
    "evaluate-models", "setup", "define-task", "build-dataset",
    "create-evals", "run-comparison", "power-analysis", "analyze-results",
}


def _tree_diffs(src: Path, dst: Path) -> list[str]:
    cmp = filecmp.dircmp(src, dst)
    diffs = [f"only in canonical: {src / n}" for n in cmp.left_only]
    diffs += [f"only in plugin: {dst / n}" for n in cmp.right_only]
    diffs += [f"differs: {dst / n}" for n in cmp.diff_files]
    for sub in cmp.common_dirs:
        diffs += _tree_diffs(src / sub, dst / sub)
    return diffs


def test_plugin_is_synced_with_canonical():
    diffs: list[str] = []
    for sub in SUBTREES:
        assert (PLUGIN / sub).is_dir(), f"plugin missing {sub}/ — run scripts/sync_skills.py"
        diffs += _tree_diffs(CANONICAL / sub, PLUGIN / sub)
    assert not diffs, "out of sync (run scripts/sync_skills.py):\n" + "\n".join(diffs)


def test_expected_skills_present():
    found = {p.parent.name for p in (CANONICAL / "skills").glob("*/SKILL.md")}
    assert found == EXPECTED_SKILLS


@pytest.mark.parametrize("skill_md", sorted((CANONICAL / "skills").glob("*/SKILL.md")))
def test_skill_frontmatter_name_matches_dir(skill_md: Path):
    text = skill_md.read_text()
    assert text.startswith("---\n"), f"{skill_md} missing YAML frontmatter"
    front = text.split("---\n", 2)[1]
    name = re.search(r"^name:\s*(\S+)\s*$", front, re.MULTILINE)
    desc = re.search(r"^description:\s*\S", front, re.MULTILINE)
    assert name and name.group(1) == skill_md.parent.name, f"{skill_md}: name must equal dir"
    assert desc, f"{skill_md}: description is required (it is the trigger spec)"


@pytest.mark.parametrize("skill_md", sorted((CANONICAL / "skills").glob("*/SKILL.md")))
def test_reference_links_resolve(skill_md: Path):
    for rel in re.findall(r"\.\./\.\./references/([A-Za-z0-9._-]+)", skill_md.read_text()):
        assert (CANONICAL / "references" / rel).is_file(), f"{skill_md} links missing {rel}"
