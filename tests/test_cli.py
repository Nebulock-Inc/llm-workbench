"""CLI integration via click's CliRunner — all offline against mock/."""

from __future__ import annotations

from click.testing import CliRunner

from llm_lab.cli.main import cli
from llm_lab.evals.spec import write_spec


def test_help_lists_command_groups():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for group in ("evals", "models", "run"):
        assert group in result.output


def test_models_list():
    result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "ollama" in result.output and "bedrock-mantle" in result.output


def test_models_ping_mock_echo():
    result = CliRunner().invoke(cli, ["models", "ping", "mock/echo", "--prompt", "hello there"])
    assert result.exit_code == 0
    assert "hello there" in result.output


def test_evals_new_list_validate(tmp_path):
    runner = CliRunner()
    kit = tmp_path / "kit"

    created = runner.invoke(cli, ["evals", "new", "Answer Quality", "--labels", "1-5", "--yes", "--dir", str(kit)])
    assert created.exit_code == 0, created.output
    assert (kit / "answer-quality" / "eval.yaml").is_file()

    listed = runner.invoke(cli, ["evals", "list", "--dir", str(kit)])
    assert listed.exit_code == 0 and "Answer" in listed.output

    validated = runner.invoke(cli, ["evals", "validate", "--dir", str(kit)])
    assert validated.exit_code == 0 and "1/1 valid" in validated.output


def test_init_scaffolds_starter(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "prompt.txt").is_file()
    assert (tmp_path / "dataset.jsonl").is_file()
    assert (tmp_path / "evals_kit" / "example-helpfulness" / "eval.yaml").is_file()
    assert (tmp_path / "evals_kit" / "example-helpfulness" / "prompt.md").is_file()


def test_init_skips_existing_without_force(tmp_path):
    runner = CliRunner()
    assert runner.invoke(cli, ["init", str(tmp_path)]).exit_code == 0
    (tmp_path / "dataset.jsonl").write_text("MINE\n")

    again = runner.invoke(cli, ["init", str(tmp_path)])
    assert again.exit_code == 0
    assert (tmp_path / "dataset.jsonl").read_text() == "MINE\n"  # untouched

    forced = runner.invoke(cli, ["init", str(tmp_path), "--force"])
    assert forced.exit_code == 0
    assert (tmp_path / "dataset.jsonl").read_text() != "MINE\n"  # overwritten


def test_run_compare(tmp_path):
    runner = CliRunner()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id": "q1", "input": "2+2?"}\n')
    template = tmp_path / "prompt.txt"
    template.write_text("Q: {input}")
    kit = tmp_path / "kit"
    write_spec(dest_dir=kit / "help", name="Help", description="",
               labels={"good": 5, "bad": 1}, prompt="Q {input} A {output}")
    out = tmp_path / "runs"

    result = runner.invoke(cli, [
        "run", "compare",
        "--prompt-template", str(template),
        "--dataset", str(dataset),
        "--model", "mock/echo",
        "--eval", "help", "--eval-dir", str(kit),
        "--judge-model", "mock/canned:SCORE: 5",
        "--out-dir", str(out), "--no-resume",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "results.jsonl").is_file() and (out / "results.csv").is_file()
