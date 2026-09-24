"""The model × record comparison runner."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from llm_workbench.config import LabConfig
from llm_workbench.datasets.jsonl import append_jsonl, read_jsonl
from llm_workbench.models.factory import make_client
from llm_workbench.runner.results import RunResult, ScoreRow, load_completed, print_summary, record_id, write_csv
from llm_workbench.scoring.base import ScoreContext, ScoreResult
from llm_workbench.scoring.registry import build_scorers


def _not_applicable_reason(spec, task_output) -> str | None:
    """A spec scoped with `requires_aux` only applies to rows carrying a non-empty value
    for that task-aux key. On other rows every scorer (deterministic and LLM-based alike)
    is skipped as not-applicable — emitted as an `error` so the row stays out of the score
    distribution (see analysis/frame.py) rather than being scored 0."""
    req = spec.requires_aux
    if req and not str(task_output.aux.get(req) or "").strip():
        return f"not applicable: task-aux {req!r} is empty"
    return None


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
    on_warning: Callable[[str], None] | None = None,
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
    # Construct every client up front so a bad model/judge id fails fast with a clear
    # error before any records run — never mid-sweep after partial work. (Clients build
    # lazily, so this makes no network calls.)
    judge_client = make_client(judge_model, config) if judge_model else None
    clients = {model_id: make_client(model_id, config) for model_id in model_ids}
    scorers_by_eval = {s.name: build_scorers(s, include_judge=judge_model is not None) for s in specs}

    # Pre-flight: warn (once) about models that look unavailable, before the sweep
    # spends every row on a call that's going to fail. Providers that can't check
    # cheaply return None here. This is the only place the runner makes a network
    # call before running records.
    if on_warning:
        for client in ([judge_client] if judge_client else []) + list(clients.values()):
            if (msg := client.availability_warning()) is not None:
                on_warning(msg)

    for model_id in model_ids:
        client = clients[model_id]
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
                    latency_s=0.0, usage={},
                    error=f"task error: {type(exc).__name__}: {exc}", scores=[],
                ).to_dict())
                if on_event:
                    on_event("error", model_id, rid)
                continue

            score_rows: list[ScoreRow] = []
            for spec in specs:
                ctx = ScoreContext(record=record, output=out.output, task_output=out,
                                   judge_client=judge_client)
                na = _not_applicable_reason(spec, out)
                for scorer in scorers_by_eval[spec.name]:
                    res = (ScoreResult(scorer=scorer.name, score=0.0, error=na)
                           if na is not None else scorer.score(ctx))
                    score_rows.append(ScoreRow(eval=spec.name, scorer=res.scorer, score=res.score,
                                               label=res.label, passed=res.passed,
                                               reasoning=res.reasoning, error=res.error))

            resp = out.response
            append_jsonl(results_path, RunResult(
                model_id=model_id, record_id=rid, input=record, output=out.output, prompt=out.prompt,
                latency_s=resp.latency_s if resp else 0.0,
                usage=({"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.total_tokens} if resp else {}),
                error=resp.error if resp else None,
                scores=score_rows,
            ).to_dict())
            if on_event:
                on_event("done", model_id, rid)

    write_csv(out_dir)
    print_summary(out_dir)
    return out_dir
