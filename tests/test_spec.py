"""Eval spec loading/validation, serialization, discovery, and templating."""

from __future__ import annotations

import pytest

from llm_workbench.evals.loader import discover
from llm_workbench.evals.spec import EvalSpecError, load_spec, write_spec
from llm_workbench.evals.template import fill_template, find_variables


def _make(dirpath, yaml_text, prompt=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "eval.yaml").write_text(yaml_text)
    if prompt is not None:
        (dirpath / "prompt.md").write_text(prompt)
    return dirpath


def test_load_valid(tmp_path):
    d = _make(
        tmp_path / "e",
        "name: Q\nlabels:\n  good: 5\n  bad: 1\nprompt_file: prompt.md\n",
        "Judge {input} vs {output}",
    )
    spec = load_spec(d)
    assert spec.name == "Q"
    assert spec.has_judge
    assert spec.max_label_score == 5.0
    assert spec.variables == ("input", "output")


def test_unknown_key_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nintegration: foo\nlabels:\n  a: 1\nprompt: 'x {input}'\n")
    with pytest.raises(EvalSpecError, match="Unknown key"):
        load_spec(d)


def test_missing_name_rejected(tmp_path):
    d = _make(tmp_path / "e", "description: no name\nlabels:\n  a: 1\nprompt: 'x {input}'\n")
    with pytest.raises(EvalSpecError, match="missing required key 'name'"):
        load_spec(d)


def test_labels_without_prompt_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nlabels:\n  a: 1\n")  # no prompt, no prompt.md
    with pytest.raises(EvalSpecError, match="no judge prompt"):
        load_spec(d)


def test_prompt_without_labels_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nprompt: 'x {input}'\n")
    with pytest.raises(EvalSpecError, match="needs 'labels'"):
        load_spec(d)


def test_scorer_stray_key_points_to_params(tmp_path):
    # 'schema' at the scorer's top level (should be under params) is the common mistake;
    # the error should name the stray key and point to 'params', not surface a raw pydantic dump.
    d = _make(tmp_path / "e", "name: Q\nscorers:\n  - type: json_schema\n    schema:\n      type: object\n")
    with pytest.raises(EvalSpecError, match="under 'params'"):
        load_spec(d)


def test_scorer_params_accepted(tmp_path):
    d = _make(
        tmp_path / "e",
        "name: Q\nscorers:\n  - type: json_schema\n    params:\n      schema:\n        type: object\n",
    )
    spec = load_spec(d)
    assert spec.scorers[0].params["schema"] == {"type": "object"}


def test_scorer_missing_type_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nscorers:\n  - expected_field: gold\n")
    with pytest.raises(EvalSpecError, match="missing required key 'type'"):
        load_spec(d)


def test_literal_empty_braces_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nlabels:\n  a: 1\nprompt: 'has {} braces'\n")
    with pytest.raises(EvalSpecError, match="literal"):
        load_spec(d)


def test_bad_direction_rejected(tmp_path):
    d = _make(tmp_path / "e", "name: Q\ndirection: sideways\nlabels:\n  a: 1\nprompt: 'x {input}'\n")
    with pytest.raises(EvalSpecError, match="direction"):
        load_spec(d)


@pytest.mark.parametrize("value", ["true", "false", "0", "3"])
def test_non_string_requires_aux_rejected(tmp_path, value):
    # A YAML bool/number must be rejected, not coerced (true -> "True") or used to
    # silently disable the gate (false/0).
    d = _make(tmp_path / "e", f"name: Q\nrequires_aux: {value}\nlabels:\n  a: 1\nprompt: 'x {{input}}'\n")
    with pytest.raises(EvalSpecError, match="requires_aux"):
        load_spec(d)


def test_blank_requires_aux_becomes_none(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nrequires_aux: '  '\nlabels:\n  a: 1\nprompt: 'x {input}'\n")
    assert load_spec(d).requires_aux is None


def test_valid_requires_aux_string_kept(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nrequires_aux: ' verdict '\nlabels:\n  a: 1\nprompt: 'x {input}'\n")
    assert load_spec(d).requires_aux == "verdict"


def test_scorers_only_eval_is_valid(tmp_path):
    d = _make(tmp_path / "e", "name: Q\nscorers:\n  - type: exact_match\n    expected_field: expected\n")
    spec = load_spec(d)
    assert not spec.has_judge
    assert len(spec.scorers) == 1 and spec.scorers[0].type == "exact_match"


def test_write_spec_roundtrip_and_force(tmp_path):
    dest = tmp_path / "myeval"
    spec = write_spec(dest_dir=dest, name="MyEval", description="d",
                      labels={"good": 5, "bad": 1}, prompt="Judge {input} {output}")
    assert (dest / "eval.yaml").is_file() and (dest / "prompt.md").is_file()
    assert spec.name == "MyEval" and spec.variables == ("input", "output")

    with pytest.raises(EvalSpecError, match="already contains"):
        write_spec(dest_dir=dest, name="X", description="", labels={"good": 5}, prompt="x {input}")

    write_spec(dest_dir=dest, name="MyEval2", description="", labels={"good": 5},
               prompt="x {input}", force=True)
    assert load_spec(dest).name == "MyEval2"


def test_discover_loads_all(tmp_path):
    write_spec(dest_dir=tmp_path / "a", name="A", description="", labels={"g": 1}, prompt="p {input}")
    write_spec(dest_dir=tmp_path / "b", name="B", description="", labels={"g": 1}, prompt="p {input}")
    specs = discover(tmp_path)
    assert sorted(s.name for s in specs) == ["A", "B"]


def test_find_variables_and_fill():
    template = "Q: {input}\nData: {data}"
    assert find_variables(template) == ("input", "data")
    filled = fill_template(template, {"input": "hi", "data": [1, 2]})
    assert "hi" in filled
    assert "1" in filled and "2" in filled  # list json-encoded
