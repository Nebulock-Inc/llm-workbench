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


# Ground-truth-scored rows are emitted under this scorer name (not "judge") so the
# analysis treats them as their own binary accuracy metric — kept out of the judged
# quality composite and shown separately (mean = accuracy, pass_rate = accuracy).
GROUND_TRUTH_SCORER = "ground_truth"


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

    def _label_for(self, score: float) -> str | None:
        return next((lbl for lbl, s in self.spec.labels.items() if s == score), None)

    def _score_ground_truth(self, ctx: ScoreContext, gold) -> ScoreResult:
        """Deterministically score this criterion against a per-record gold value,
        bypassing the LLM judge. Emitted under the `ground_truth` scorer as a binary
        accuracy metric (correct -> 1.0, incorrect -> 0.0) so it reads as an accuracy /
        pass-rate and stays out of the judged quality composite."""
        gt = self.spec.ground_truth
        hi = gt.correct_score if gt.correct_score is not None else 1.0
        lo = gt.incorrect_score if gt.incorrect_score is not None else 0.0

        if gt.aux_field:
            got = ctx.task_output.aux.get(gt.aux_field)
            mapping = {k.strip().upper(): v.strip().upper() for k, v in gt.map.items()}

            def canon(x) -> str:
                s = str(x if x is not None else "").strip().upper()
                return mapping.get(s, s)

            matched = canon(got) == canon(gold)
            return ScoreResult(
                scorer=GROUND_TRUTH_SCORER, score=float(hi if matched else lo),
                label="correct" if matched else "incorrect", passed=matched,
                reasoning=(f"[ground truth] {gt.aux_field}={got!r} vs "
                           f"{gt.expected_field}={gold!r}: {'match' if matched else 'mismatch'}"),
            )

        # No aux_field: the gold value is a human-set numeric score for this criterion.
        try:
            score = float(gold)
        except (TypeError, ValueError):
            return ScoreResult(
                scorer=GROUND_TRUTH_SCORER, score=0.0,
                error=f"ground_truth {gt.expected_field}={gold!r} is not numeric and no aux_field is set",
            )
        return ScoreResult(
            scorer=GROUND_TRUTH_SCORER, score=score, label=self._label_for(score),
            reasoning=f"[ground truth] human-set score {score:g} from {gt.expected_field}",
        )

    def score(self, ctx: ScoreContext) -> ScoreResult:
        # Per-record ground-truth override: if this record carries the gold value for this
        # criterion, score against it deterministically instead of asking the judge.
        gt = self.spec.ground_truth
        if gt is not None:
            gold = ctx.record.get(gt.expected_field)
            if gold is not None and str(gold).strip() != "":
                return self._score_ground_truth(ctx, gold)

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
