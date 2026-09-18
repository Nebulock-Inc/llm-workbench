"""Statistical analysis layer. Skipped unless the `analysis` extra is installed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")
pytest.importorskip("pandas")


def _write_results(path):
    """3 models x 8 prompts: a graded judge metric (clear ordering) + a binary metric."""
    judge = {"m1": [5, 5, 4, 5, 4, 5, 4, 5], "m2": [4, 3, 4, 3, 4, 3, 4, 3], "m3": [2, 3, 2, 2, 3, 2, 3, 2]}
    passes = {"m1": [1, 1, 1, 1, 1, 1, 1, 0], "m2": [1, 0, 1, 0, 1, 0, 1, 0], "m3": [0, 0, 1, 0, 0, 0, 1, 0]}
    lines = []
    for model in ("m1", "m2", "m3"):
        for i in range(8):
            lines.append(json.dumps({
                "model_id": f"mock/{model}", "record_id": f"q{i}", "input": {}, "output": "x",
                "prompt": "p", "latency_s": 0.5, "usage": {}, "error": None,
                "scores": [
                    {"eval": "E", "scorer": "judge", "score": float(judge[model][i]), "error": None},
                    {"eval": "E", "scorer": "exact_match", "score": float(passes[model][i]),
                     "passed": bool(passes[model][i]), "error": None},
                ],
            }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dedupes_to_latest_run(tmp_path):
    from llm_lab.analysis.frame import load_long, reliability

    lines = [
        # older run for m1/q0 -> judge 2
        {"model_id": "mock/m1", "record_id": "q0", "error": None, "timestamp": "2026-01-01T00:00:00",
         "scores": [{"eval": "E", "scorer": "judge", "score": 2.0, "error": None}]},
        # newer run for m1/q0 -> judge 5 (should win; not averaged to 3.5, not counted twice)
        {"model_id": "mock/m1", "record_id": "q0", "error": None, "timestamp": "2026-02-01T00:00:00",
         "scores": [{"eval": "E", "scorer": "judge", "score": 5.0, "error": None}]},
        {"model_id": "mock/m2", "record_id": "q0", "error": None, "timestamp": "2026-01-01T00:00:00",
         "scores": [{"eval": "E", "scorer": "judge", "score": 3.0, "error": None}]},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    df = load_long(path)
    m1q0 = df[(df.model == "mock/m1") & (df.prompt_id == "q0")]
    assert len(m1q0) == 1 and m1q0.iloc[0]["score"] == 5.0   # latest wins, not mean/dup

    rel = {r["model"]: r for r in reliability(path)}
    assert rel["mock/m1"]["n"] == 1                          # one unique prompt, not two rows


def test_quality_uses_latest_successful_not_latest_overall(tmp_path):
    from llm_lab.analysis.frame import load_long, reliability

    lines = [
        # m1/q0: earlier SUCCESS (4.0), then a LATER task FAILURE
        {"model_id": "mock/m1", "record_id": "q0", "error": None, "timestamp": "2026-01-01T00:00:00",
         "scores": [{"eval": "E", "scorer": "judge", "score": 4.0, "error": None}]},
        {"model_id": "mock/m1", "record_id": "q0", "error": "task error: boom",
         "timestamp": "2026-02-01T00:00:00", "scores": []},
        {"model_id": "mock/m2", "record_id": "q0", "error": None, "timestamp": "2026-01-01T00:00:00",
         "scores": [{"eval": "E", "scorer": "judge", "score": 3.0, "error": None}]},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    df = load_long(path)
    m1q0 = df[(df.model == "mock/m1") & (df.prompt_id == "q0")]
    assert len(m1q0) == 1 and m1q0.iloc[0]["score"] == 4.0   # quality keeps the earlier success

    rel = {r["model"]: r for r in reliability(path)}
    assert rel["mock/m1"]["n"] == 1 and rel["mock/m1"]["errors"] == 1   # reliability: latest = failure


def test_load_long_classifies_metrics(tmp_path):
    from llm_lab.analysis.frame import load_long

    df = load_long(_write_results(tmp_path / "results.jsonl"))
    kinds = dict(zip(df["metric"], df["kind"]))
    assert kinds["E / judge"] == "quality"        # >2 distinct values
    assert kinds["E / exact_match"] == "binary"   # 0/1


def test_load_long_drops_operational_scorers(tmp_path):
    """Latency is a run-time threshold metric, not part of the statistical analysis."""
    import json

    from llm_lab.analysis.frame import load_long

    path = tmp_path / "results.jsonl"
    lines = []
    for model in ("m1", "m2"):
        for i in range(3):
            lines.append(json.dumps({
                "model_id": f"mock/{model}", "record_id": f"q{i}", "error": None,
                "scores": [
                    {"eval": "E", "scorer": "judge", "score": 4.0, "error": None},
                    {"eval": "E", "scorer": "latency", "score": 0.42, "error": None},
                ],
            }))
    path.write_text("\n".join(lines) + "\n")

    df = load_long(path)
    assert "latency" not in set(df["scorer"])     # operational metric excluded
    assert "E / latency" not in set(df["metric"])


def test_run_analysis_verbose_reports_all_tests(tmp_path):
    from llm_lab.analysis.report import run_analysis

    _write_results(tmp_path / "results.jsonl")
    report = run_analysis(tmp_path, out_dir=tmp_path, make_plots=False, verbose=True)

    assert report["reference"] == "mock/m1"  # first alphabetically
    # Composite (quality) got the full graded treatment under --verbose.
    comp = report["composite"]
    assert comp is not None
    assert "friedman" in comp and comp["friedman"]["k"] == 3
    assert comp["mixed_model"]["error"] is None
    assert len(comp["mixed_model"]["effects"]) == 2  # m2, m3 vs reference m1
    assert comp["mixed_model"].get("summary_text")   # full statsmodels summary captured
    assert comp["cohens_d"] and comp["tost"]
    assert len(comp["skewness"]) == 3                # skewness per model
    assert len(comp["bootstrap"]) == 2               # bootstrap CI vs reference
    assert "overall_means" in report and len(report["overall_means"]) == 3

    # Individual metrics are means-only (full tests run on the aggregate).
    for per_metric in report["metrics"].values():
        assert per_metric["means"]
        assert "friedman" not in per_metric and "mixed_model" not in per_metric

    assert (tmp_path / "analysis.json").is_file() and (tmp_path / "analysis.md").is_file()


def test_run_analysis_simple_is_leaner(tmp_path):
    from llm_lab.analysis.report import run_analysis

    _write_results(tmp_path / "results.jsonl")
    report = run_analysis(tmp_path, out_dir=tmp_path, make_plots=False)  # verbose=False (default)

    # Headline only: overall means + composite mixed model are kept; the verbose-gated
    # extras (per-metric table, Friedman, Cohen's d, TOST, skewness, bootstrap) are omitted.
    assert "overall_means" in report
    comp = report["composite"]
    assert comp["mixed_model"]["error"] is None       # mixed model is always computed
    assert "friedman" not in comp
    assert "cohens_d" not in comp and "tost" not in comp
    assert "skewness" not in comp and "bootstrap" not in comp
    assert report["metrics"] == {}                    # per-metric table is verbose-only


def test_exclude_metric_from_aggregate(tmp_path):
    import random

    from llm_lab.analysis.report import run_analysis

    random.seed(1)
    lines = []
    for model in ("m1", "m2", "m3"):
        base = {"m1": 4.5, "m2": 3.5, "m3": 2.5}[model]
        for i in range(8):
            q = max(1, min(5, round(random.gauss(base, 0.6))))
            r = max(1, min(5, round(random.gauss(3, 0.6))))
            lines.append(json.dumps({
                "model_id": f"mock/{model}", "record_id": f"q{i}", "input": {}, "output": "x",
                "prompt": "p", "latency_s": 0.1, "usage": {}, "error": None,
                "scores": [{"eval": "Quality", "scorer": "judge", "score": float(q), "error": None},
                           {"eval": "Refusal", "scorer": "judge", "score": float(r), "error": None}]}))
    (tmp_path / "results.jsonl").write_text("\n".join(lines) + "\n")

    rep = run_analysis(tmp_path, reference="mock/m1", make_plots=False, exclude_metrics=["Refusal"])
    assert rep["excluded_from_aggregate"] == ["Refusal"]
    comp = rep["composite"]
    # only Quality remains in the aggregate -> single metric -> no C(metric) term, Refusal absent
    assert comp["mixed_model"]["formula"] == "score ~ C(model)"
    assert "Refusal" not in (comp["mixed_model"]["summary_text"] or "")

    # without exclusion, Refusal IS part of the combined model
    rep2 = run_analysis(tmp_path, reference="mock/m1", make_plots=False)
    assert "Refusal" in (rep2["composite"]["mixed_model"]["summary_text"] or "")


def test_incomplete_coverage_uses_per_model_prompts(tmp_path):
    from llm_lab.analysis.report import run_analysis

    lines = []
    for model in ("m1", "m2", "m3"):
        prompts = range(6) if model != "m3" else range(1, 6)  # m3 is missing q0
        for i in prompts:
            base = {"m1": 5, "m2": 4, "m3": 3}[model]
            score = min(5, base + (i % 2))  # >2 distinct values overall -> quality metric
            lines.append(json.dumps({
                "model_id": f"mock/{model}", "record_id": f"q{i}", "error": None,
                "timestamp": "2026-01-01T00:00:00", "input": {}, "output": "x", "prompt": "p",
                "latency_s": 0.1, "usage": {},
                "scores": [{"eval": "E", "scorer": "judge", "score": float(score), "error": None}]}))
    (tmp_path / "results.jsonl").write_text("\n".join(lines) + "\n")

    rep = run_analysis(tmp_path, reference="mock/m1", make_plots=False)
    # Friedman uses complete cases only (q0 excluded because m3 lacks it) -> 5
    assert rep["composite"]["n_prompts"] == 5
    # overall means use each model's own prompts (skipna), not the complete-case intersection
    assert rep["overall_means"]["mock/m1"]["count"] == 6
    assert rep["overall_means"]["mock/m3"]["count"] == 5


def test_analyze_needs_two_models(tmp_path):
    from llm_lab.analysis.report import run_analysis

    (tmp_path / "results.jsonl").write_text(json.dumps({
        "model_id": "mock/only", "record_id": "q0", "input": {}, "output": "x", "prompt": "p",
        "latency_s": 0.1, "usage": {}, "error": None,
        "scores": [{"eval": "E", "scorer": "judge", "score": 5.0, "error": None}],
    }) + "\n")
    with pytest.raises(ValueError, match="at least 2 models"):
        run_analysis(tmp_path, make_plots=False)


def test_cli_analyze(tmp_path):
    from click.testing import CliRunner

    from llm_lab.cli.main import cli

    _write_results(tmp_path / "results.jsonl")
    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--no-plots"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "analysis.json").is_file()


# --- power analysis ------------------------------------------------------

def test_power_n_for_effect():
    from llm_lab.analysis.power import n_for_effect

    assert n_for_effect(0.5, alpha=0.05, power=0.8) == 34   # standard reference value
    assert n_for_effect(1.0) < n_for_effect(0.3)            # bigger effect -> fewer prompts


def test_power_achieved_increases_with_n():
    from llm_lab.analysis.power import achieved_power

    assert achieved_power(0.5, 10) < achieved_power(0.5, 50)
    assert achieved_power(0.0, 100) == 0.0


def test_power_from_run(tmp_path):
    from llm_lab.analysis.power import from_run

    _write_results(tmp_path / "results.jsonl")
    result = from_run(tmp_path, reference="mock/m1", power=0.8)
    assert result["reference"] == "mock/m1"
    assert len(result["pairs"]) == 2
    assert all(p.n_needed is not None and p.n_needed > 0 for p in result["pairs"])


def test_cli_power(tmp_path):
    from click.testing import CliRunner

    from llm_lab.cli.main import cli

    result = CliRunner().invoke(cli, ["power", "--effect-size", "0.5"])
    assert result.exit_code == 0 and "34 prompts" in result.output


def test_power_from_run_default_standardizes_all_metrics(tmp_path):
    from llm_lab.analysis import power as P
    res = P.from_run(_write_results(tmp_path / "results.jsonl"))
    assert res["standardize"] is True
    assert res["metrics"] == ["E / exact_match", "E / judge"]   # all metrics pooled by default
    assert {p.model for p in res["pairs"]} == {"mock/m2", "mock/m3"}


def test_power_from_run_raw_uses_graded_quality_only(tmp_path):
    from llm_lab.analysis import power as P
    res = P.from_run(_write_results(tmp_path / "results.jsonl"), standardize=False)
    assert res["standardize"] is False
    assert res["metrics"] == ["E / judge"]          # --raw: binary metric excluded


def test_power_from_run_binary_scorer_with_raw_errors(tmp_path):
    """--raw drops binary metrics, so selecting only binary leaves nothing to size on."""
    from llm_lab.analysis import power as P
    with pytest.raises(ValueError, match="standardize"):
        P.from_run(_write_results(tmp_path / "results.jsonl"),
                   scorers={"exact_match"}, standardize=False)


def test_power_from_run_binary_scorer_with_standardize(tmp_path):
    from llm_lab.analysis import power as P
    res = P.from_run(_write_results(tmp_path / "results.jsonl"),
                     scorers={"exact_match"}, standardize=True)
    assert res["standardize"] is True
    assert res["metrics"] == ["E / exact_match"]    # sized on the accuracy track alone
    # m1 clearly beats m3 on the binary metric -> a detectable effect (dz != 0, N computed)
    by_model = {p.model: p for p in res["pairs"]}
    assert by_model["mock/m3"].dz != 0.0 and by_model["mock/m3"].n_needed is not None


def test_power_from_run_constant_nonzero_diff_raises(tmp_path):
    """Regression: a zero-variance paired difference with a nonzero mean is a perfectly
    consistent (unbounded) effect, not zero effect — it must raise, not report dz=0."""
    from llm_lab.analysis import power as P
    # m1 passes every prompt, m2 fails every prompt. The binary metric varies across the run
    # (so it survives z-scoring), but the m2-vs-m1 paired difference is a constant -1.
    lines = []
    for model, val in (("m1", 1), ("m2", 0)):
        for i in range(6):
            lines.append(json.dumps({
                "model_id": f"mock/{model}", "record_id": f"q{i}", "input": {}, "output": "x",
                "prompt": "p", "latency_s": 0.5, "usage": {}, "error": None,
                "scores": [
                    {"eval": "E", "scorer": "exact_match", "score": float(val),
                     "passed": bool(val), "error": None},
                ],
            }))
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="zero-variance paired"):
        P.from_run(path, scorers={"exact_match"}, standardize=True)


def test_power_from_run_bad_filter_errors(tmp_path):
    from llm_lab.analysis import power as P
    with pytest.raises(ValueError, match="No metrics match"):
        P.from_run(_write_results(tmp_path / "results.jsonl"), scorers={"nonexistent"})
