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
                "prompt": "p", "latency_s": 0.5, "usage": {}, "cost_usd": None, "error": None,
                "scores": [
                    {"eval": "E", "scorer": "judge", "score": float(judge[model][i]), "error": None},
                    {"eval": "E", "scorer": "exact_match", "score": float(passes[model][i]),
                     "passed": bool(passes[model][i]), "error": None},
                ],
            }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_long_classifies_metrics(tmp_path):
    from llm_lab.analysis.frame import load_long

    df = load_long(_write_results(tmp_path / "results.jsonl"))
    kinds = dict(zip(df["metric"], df["kind"]))
    assert kinds["E / judge"] == "quality"        # >2 distinct values
    assert kinds["E / exact_match"] == "binary"   # 0/1


def test_run_analysis_reports_all_appropriate_tests(tmp_path):
    from llm_lab.analysis.report import run_analysis

    _write_results(tmp_path / "results.jsonl")
    report = run_analysis(tmp_path, out_dir=tmp_path, make_plots=False)

    assert report["reference"] == "mock/m1"  # first alphabetically
    # Composite (quality) got the full graded treatment.
    comp = report["composite"]
    assert comp is not None
    assert "friedman" in comp and comp["friedman"]["k"] == 3
    assert comp["mixed_model"]["error"] is None
    assert len(comp["mixed_model"]["effects"]) == 2  # m2, m3 vs reference m1
    assert comp["cohens_d"] and comp["tost"]

    # Binary metric: nonparametric only, graded tests skipped with a note.
    binary = report["metrics"]["E / exact_match"]
    assert "friedman" in binary
    assert "mixed_model" not in binary
    assert any("binary" in n for n in binary["notes"])

    assert (tmp_path / "analysis.json").is_file() and (tmp_path / "analysis.md").is_file()


def test_analyze_needs_two_models(tmp_path):
    from llm_lab.analysis.report import run_analysis

    (tmp_path / "results.jsonl").write_text(json.dumps({
        "model_id": "mock/only", "record_id": "q0", "input": {}, "output": "x", "prompt": "p",
        "latency_s": 0.1, "usage": {}, "cost_usd": None, "error": None,
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
