"""Deterministic scorers that run without an LLM."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import jsonschema

from llm_workbench.scoring.base import ScoreContext, ScoreResult

if TYPE_CHECKING:
    from llm_workbench.evals.spec import ScorerSpec


def _expected(ctx: ScoreContext, field: str | None):
    return ctx.record.get(field) if field else None


def _first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


# ```json … ``` (or a bare ``` … ```) that instruction-tuned models wrap output in.
# Matches only when a fence encloses the whole (stripped) text, so we don't mangle
# output that legitimately contains a fence in the middle.
_FENCE = re.compile(r"\A```[^\n`]*\n(?P<body>.*)\n```\Z", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding markdown code fence, if the whole text is one."""
    m = _FENCE.match(text.strip())
    return m.group("body") if m else text


class ExactMatch:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = f"exact_match:{spec.expected_field}" if spec.expected_field else "exact_match"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        exp = _expected(ctx, self.spec.expected_field)
        if exp is None:
            return ScoreResult(scorer=self.name, score=0.0, error=f"missing expected_field {self.spec.expected_field!r}")
        passed = ctx.output.strip() == str(exp).strip()
        return ScoreResult(scorer=self.name, score=1.0 if passed else 0.0, passed=passed)


class Contains:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = "contains"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        needle = self.spec.params.get("substring")
        if needle is None and self.spec.expected_field:
            needle = _expected(ctx, self.spec.expected_field)
        if needle is None:
            return ScoreResult(scorer=self.name, score=0.0, error="contains needs params.substring or expected_field")
        case_sensitive = bool(self.spec.params.get("case_sensitive", False))
        hay = ctx.output if case_sensitive else ctx.output.lower()
        needle_s = str(needle) if case_sensitive else str(needle).lower()
        passed = needle_s in hay
        return ScoreResult(scorer=self.name, score=1.0 if passed else 0.0, passed=passed)


class Regex:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = "regex"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        pattern = self.spec.params.get("pattern")
        if not pattern:
            return ScoreResult(scorer=self.name, score=0.0, error="regex needs params.pattern")
        flags = re.IGNORECASE if self.spec.params.get("ignore_case") else 0
        passed = re.search(pattern, ctx.output, flags) is not None
        return ScoreResult(scorer=self.name, score=1.0 if passed else 0.0, passed=passed)


class JSONSchemaValid:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = "json_schema"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        # Models routinely wrap JSON in a ```json … ``` fence; tolerate that by default
        # so we measure "is this valid JSON" not "did it skip the markdown". Set
        # params.strip_fence: false to require bare JSON.
        text = ctx.output if self.spec.params.get("strip_fence") is False else _strip_code_fence(ctx.output)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return ScoreResult(scorer=self.name, score=0.0, passed=False, reasoning=f"not valid JSON: {exc}")
        schema = self.spec.params.get("schema")
        if schema:
            try:
                jsonschema.validate(data, schema)
            except jsonschema.ValidationError as exc:
                return ScoreResult(scorer=self.name, score=0.0, passed=False, reasoning=str(exc).splitlines()[0])
        return ScoreResult(scorer=self.name, score=1.0, passed=True)


class NumericTolerance:
    def __init__(self, spec: "ScorerSpec") -> None:
        self.spec = spec
        self.name = f"numeric_tolerance:{spec.expected_field}" if spec.expected_field else "numeric_tolerance"

    def score(self, ctx: ScoreContext) -> ScoreResult:
        exp = _expected(ctx, self.spec.expected_field)
        if exp is None:
            return ScoreResult(scorer=self.name, score=0.0, error=f"missing expected_field {self.spec.expected_field!r}")
        got = _first_number(ctx.output)
        try:
            want = float(exp)
        except (TypeError, ValueError):
            return ScoreResult(scorer=self.name, score=0.0, error=f"expected value {exp!r} is not numeric")
        if got is None:
            return ScoreResult(scorer=self.name, score=0.0, passed=False, reasoning="no number found in output")
        atol = float(self.spec.params.get("atol", 0.0))
        rtol = float(self.spec.params.get("rtol", 0.0))
        passed = abs(got - want) <= atol + rtol * abs(want)
        return ScoreResult(scorer=self.name, score=1.0 if passed else 0.0, passed=passed)
