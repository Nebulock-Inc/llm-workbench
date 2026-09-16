"""Standalone eval spec: the eval.yaml + prompt.md an author writes.

Trimmed from the Arize-oriented format — no integration/mappings/filter/subqueries/
model. An eval says how to score a model's output: an LLM-judge rubric (labels +
prompt) and/or a list of programmatic scorers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_lab.evals.template import find_variables

KNOWN_KEYS = {
    "name", "description", "labels", "prompt", "prompt_file",
    "direction", "use_structured_output", "scorers", "ground_truth",
    "requires_aux",
}


class EvalSpecError(Exception):
    """Raised when an eval.yaml is missing, malformed, or self-contradictory."""


class Direction(str, Enum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"
    NONE = "NONE"


class ScorerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str                                       # registry key, e.g. "exact_match"
    expected_field: str | None = None               # record key holding ground truth
    params: dict[str, Any] = Field(default_factory=dict)


class GroundTruthSpec(BaseModel):
    """Per-record ground-truth override for a judge eval. When a record supplies
    `expected_field`, the judge is bypassed and the criterion is scored deterministically
    against that gold value; otherwise the LLM judge runs as usual."""

    model_config = ConfigDict(extra="forbid")

    expected_field: str                             # record key holding the gold value
    aux_field: str | None = None                    # task-aux key with the parsed answer to
                                                    # compare (categorical). If omitted, the
                                                    # gold value is used as a direct numeric score.
    map: dict[str, str] = Field(default_factory=dict)  # canonicalization applied (case-insensitively)
                                                    # to BOTH the aux value and the gold before
                                                    # comparing, e.g. {MALICIOUS: SUSPICIOUS} to
                                                    # score a 3-level answer against a binary label.
    correct_score: float | None = None             # score on a match (default: 1.0 -> accuracy)
    incorrect_score: float | None = None            # score on a mismatch (default: 0.0 -> accuracy)


class EvalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    labels: dict[str, float] = Field(default_factory=dict)   # label -> score (judge rubric)
    prompt: str = ""                                         # resolved judge prompt template
    direction: Direction = Direction.MAXIMIZE
    use_structured_output: bool = False
    scorers: list[ScorerSpec] = Field(default_factory=list)
    ground_truth: GroundTruthSpec | None = None             # per-record override of the judge
    requires_aux: str | None = None                         # task-aux key that must be non-empty
                                                            # for this eval to apply; when the key is
                                                            # missing/blank the row is SKIPPED (scored
                                                            # as an error so it drops out of the stats),
                                                            # letting an eval scope to a subset of rows
    variables: tuple[str, ...] = ()                         # {placeholders} in the prompt
    source_dir: Path | None = None

    @property
    def has_judge(self) -> bool:
        return bool(self.labels) and bool(self.prompt.strip())

    @property
    def max_label_score(self) -> float:
        return max(self.labels.values()) if self.labels else 0.0

    @property
    def min_label_score(self) -> float:
        return min(self.labels.values()) if self.labels else 0.0


def load_spec(path: str | Path) -> EvalSpec:
    """Read and validate one eval. `path` may be the eval.yaml or its directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "eval.yaml"
    if not path.is_file():
        raise EvalSpecError(f"Eval spec not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise EvalSpecError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvalSpecError(f"{path} must contain a YAML mapping.")

    if unknown := sorted(set(raw) - KNOWN_KEYS):
        raise EvalSpecError(
            f"Unknown key(s) in {path}: {', '.join(unknown)}. Known keys: {', '.join(sorted(KNOWN_KEYS))}"
        )
    if "name" not in raw:
        raise EvalSpecError(f"{path} is missing required key 'name'.")

    # Resolve the judge prompt: inline, from prompt_file, or the prompt.md convention.
    if "prompt" in raw and "prompt_file" in raw:
        raise EvalSpecError(f"{path}: set only one of 'prompt' or 'prompt_file'.")
    if "prompt" in raw:
        prompt = str(raw["prompt"])
    elif "prompt_file" in raw:
        pf = path.parent / str(raw["prompt_file"])
        if not pf.is_file():
            raise EvalSpecError(f"prompt_file not found: {pf}")
        prompt = pf.read_text()
    else:
        default_pf = path.parent / "prompt.md"
        prompt = default_pf.read_text() if default_pf.is_file() else ""

    labels = raw.get("labels") or {}
    if labels and not isinstance(labels, dict):
        raise EvalSpecError(f"{path}: 'labels' must be a mapping of label -> numeric score.")
    for label, score in labels.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise EvalSpecError(f"{path}: label {label!r} must map to a number, got {score!r}.")

    scorers = raw.get("scorers") or []
    has_judge = bool(labels) and bool(prompt.strip())
    # Specific diagnostics first, generic catch-all last.
    if labels and not prompt.strip():
        raise EvalSpecError(f"{path}: 'labels' given but no judge prompt found (add prompt/prompt_file or prompt.md).")
    if prompt.strip() and not labels:
        raise EvalSpecError(f"{path}: a judge prompt needs 'labels' to score against.")
    if not has_judge and not scorers:
        raise EvalSpecError(
            f"{path}: an eval must define a judge (labels + prompt/prompt.md) and/or a non-empty 'scorers' list."
        )
    if "{}" in prompt:
        raise EvalSpecError(f"{path}: the prompt contains a literal '{{}}' — reword so no bare braces remain.")

    ground_truth_raw = raw.get("ground_truth")
    if ground_truth_raw is not None:
        if not isinstance(ground_truth_raw, dict):
            raise EvalSpecError(f"{path}: 'ground_truth' must be a mapping.")
        if not has_judge:
            raise EvalSpecError(
                f"{path}: 'ground_truth' overrides the judge per record, so it needs a judge "
                "rubric to fall back to (add labels + prompt/prompt.md)."
            )
        try:
            ground_truth = GroundTruthSpec(**ground_truth_raw)
        except ValidationError as exc:
            raise EvalSpecError(f"{path}: invalid 'ground_truth': {exc}") from exc
    else:
        ground_truth = None

    direction = str(raw.get("direction", "MAXIMIZE")).upper()
    if direction not in Direction.__members__:
        raise EvalSpecError(f"{path}: 'direction' must be one of {', '.join(Direction.__members__)}; got {direction!r}.")

    # requires_aux names an aux key; only a string is meaningful. Reject non-strings
    # (e.g. a YAML bool/number) rather than coercing (true -> "True") or silently
    # disabling the gate (false/0). A valid empty/blank string collapses to None.
    requires_aux_raw = raw.get("requires_aux")
    if requires_aux_raw is not None and not isinstance(requires_aux_raw, str):
        raise EvalSpecError(
            f"{path}: 'requires_aux' must be a string naming the aux key that must be "
            f"non-empty; got {type(requires_aux_raw).__name__} ({requires_aux_raw!r})."
        )
    requires_aux = (requires_aux_raw.strip() or None) if requires_aux_raw is not None else None

    try:
        return EvalSpec(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            labels={str(k): float(v) for k, v in labels.items()},
            prompt=prompt,
            direction=Direction(direction),
            use_structured_output=bool(raw.get("use_structured_output", False)),
            scorers=[ScorerSpec(**s) for s in scorers],
            ground_truth=ground_truth,
            requires_aux=requires_aux,
            variables=find_variables(prompt),
            source_dir=path.parent,
        )
    except ValidationError as exc:
        raise EvalSpecError(f"{path}: {exc}") from exc


def render_spec_files(
    *,
    name: str,
    description: str,
    labels: dict[str, float],
    prompt: str,
    direction: str = "MAXIMIZE",
    scorers: list[dict[str, Any]] | None = None,
    use_structured_output: bool = False,
) -> tuple[str, str]:
    """Render (eval.yaml text, prompt.md text) without writing anything."""
    config: dict[str, Any] = {"name": name}
    if description:
        config["description"] = description
    config["direction"] = direction
    if labels:
        config["labels"] = labels
    if use_structured_output:
        config["use_structured_output"] = use_structured_output
    if prompt.strip():
        config["prompt_file"] = "prompt.md"
    if scorers:
        config["scorers"] = scorers
    yaml_text = yaml.dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    return yaml_text, (prompt if prompt.strip() else "")


def write_spec(
    *,
    dest_dir: str | Path,
    name: str,
    description: str,
    labels: dict[str, float],
    prompt: str,
    direction: str = "MAXIMIZE",
    scorers: list[dict[str, Any]] | None = None,
    use_structured_output: bool = False,
    force: bool = False,
) -> EvalSpec:
    """Write eval.yaml + prompt.md into dest_dir, then re-load to validate."""
    dest_dir = Path(dest_dir)
    yaml_path, prompt_path = dest_dir / "eval.yaml", dest_dir / "prompt.md"
    if (yaml_path.exists() or prompt_path.exists()) and not force:
        raise EvalSpecError(f"{dest_dir} already contains an eval; pass force=True to overwrite.")

    yaml_text, prompt_text = render_spec_files(
        name=name, description=description, labels=labels, prompt=prompt, direction=direction,
        scorers=scorers, use_structured_output=use_structured_output,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text)
    if prompt_text:
        prompt_path.write_text(prompt_text)
    return load_spec(yaml_path)  # round-trip validation
