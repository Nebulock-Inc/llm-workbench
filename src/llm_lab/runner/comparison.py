"""The model × record comparison runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from llm_lab.config import LabConfig
from llm_lab.datasets.jsonl import append_jsonl, read_jsonl
from llm_lab.models.factory import make_client
from llm_lab.runner.results import RunResult, ScoreRow, load_completed, print_summary, record_id, write_csv
from llm_lab.scoring.base import ScoreContext
from llm_lab.scoring.registry import build_scorers


def run_comparison(
    *,
    task,
    dataset_path,
    model_ids: list[str],
    specs: list,                       # list[EvalSpec]
    judge_model: str | None = None,
    config: LabConfig | None = None,
    out_dir,
    resume: bool = True,
    limit: int | None = None,
    on_event: Callable[[str, str, str], None] | None = None,
) -> Path:
    """Run every model over every record, score with each eval, save results."""
    config = config or LabConfig.from_env()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]
    ids = [record_id(r, i) for i, r in enumerate(records)]

    completed = load_completed(out_dir) if resume else set()
    judge_client = make_client(judge_model, config) if judge_model else None
    scorers_by_eval = {s.name: build_scorers(s, include_judge=judge_model is not None) for s in specs}

    for model_id in model_ids:
        client = make_client(model_id, config)
        for record, rid in zip(records, ids):
            if (model_id, rid) in completed:
                if on_event:
                    on_event("skip", model_id, rid)
                continue

            try:
                out = task.run(record, client)
            except Exception as exc:  # a custom task blew up — record and move on
                append_jsonl(results_path, RunResult(
                    model_id=model_id, record_id=rid, input=record, output="", prompt=None,
                    latency_s=0.0, usage={}, cost_usd=None,
                    error=f"task error: {type(exc).__name__}: {exc}", scores=[],
                ).to_dict())
                if on_event:
                    on_event("error", model_id, rid)
                continue

            score_rows: list[ScoreRow] = []
            for spec in specs:
                ctx = ScoreContext(record=record, output=out.output, task_output=out,
                                   eval_spec=spec, judge_client=judge_client)
                for scorer in scorers_by_eval[spec.name]:
                    res = scorer.score(ctx)
                    score_rows.append(ScoreRow(eval=spec.name, scorer=res.scorer, score=res.score,
                                               label=res.label, passed=res.passed,
                                               reasoning=res.reasoning, error=res.error))

            resp = out.response
            append_jsonl(results_path, RunResult(
                model_id=model_id, record_id=rid, input=record, output=out.output, prompt=out.prompt,
                latency_s=resp.latency_s if resp else 0.0,
                usage=({"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.total_tokens} if resp else {}),
                cost_usd=resp.cost_usd if resp else None,
                error=resp.error if resp else None,
                scores=score_rows,
            ).to_dict())
            if on_event:
                on_event("done", model_id, rid)

    write_csv(out_dir)
    print_summary(out_dir)
    return out_dir
