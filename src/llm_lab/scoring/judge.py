"""LLM-as-judge scorer: fill the eval prompt, ask a judge model to score against
the rubric, and parse the verdict."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from llm_lab.evals.template import fill_template
from llm_lab.object_types import GenerationRequest
from llm_lab.scoring.base import ScoreContext, ScoreResult

if TYPE_CHECKING:
    from llm_lab.evals.spec import EvalSpec


class LLMJudgeScorer:
    name = "judge"

    def __init__(self, spec: "EvalSpec") -> None:
        self.spec = spec

    def _judge_prompt(self, ctx: ScoreContext) -> str:
        values = {**ctx.record, **ctx.task_output.aux, "output": ctx.output}
        filled = fill_template(self.spec.prompt, values)
        rubric = "\n".join(
            f"- {score:g}: {label}"
            for label, score in sorted(self.spec.labels.items(), key=lambda kv: -kv[1])
        )
        return (
            filled
            + "\n\nSCORING CRITERIA (choose the single best-fitting score):\n"
            + rubric
            + "\n\nRespond in exactly this format:\nSCORE: <number>\nREASONING: <one or two sentences>"
        )

    def _parse(self, text: str) -> tuple[float | None, str]:
        reasoning = text.strip()
        allowed = set(self.spec.labels.values())
        match = re.search(r"SCORE:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        score: float | None = None
        if match:
            score = float(match.group(1))
            rmatch = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
            if rmatch:
                reasoning = rmatch.group(1).strip()
        else:  # fallback: first number in the text that is a valid rubric score
            for token in re.findall(r"-?\d+(?:\.\d+)?", text):
                if float(token) in allowed:
                    score = float(token)
                    break
        if score is not None and score not in allowed:  # snap to nearest legal score
            score = min(allowed, key=lambda a: abs(a - score))
        return score, reasoning

    def score(self, ctx: ScoreContext) -> ScoreResult:
        if ctx.judge_client is None:
            return ScoreResult(scorer=self.name, score=0.0, error="no judge_client provided")
        resp = ctx.judge_client.generate(GenerationRequest(prompt=self._judge_prompt(ctx), max_tokens=1024, temperature=0.0))
        if resp.error:
            return ScoreResult(scorer=self.name, score=0.0, error=f"judge error: {resp.error}", raw_response=resp.text)
        score, reasoning = self._parse(resp.text or "")
        if score is None:
            return ScoreResult(scorer=self.name, score=0.0, error="could not parse a score from judge output", raw_response=resp.text)
        label = next((lbl for lbl, s in self.spec.labels.items() if s == score), None)
        return ScoreResult(scorer=self.name, score=float(score), label=label, reasoning=reasoning, raw_response=resp.text)
