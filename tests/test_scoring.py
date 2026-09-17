"""Tests for suite loading/validation and scoring rules."""

import textwrap

import pytest

from llm_eval_harness import (
    EvalCase,
    ScoringRule,
    load_suite,
    score_case,
)
from llm_eval_harness.backends import StubBackend
from llm_eval_harness.scorers import (
    contains_match,
    exact_match,
    judge_score,
    max_length,
    regex_match,
)
from llm_eval_harness.suite import SuiteError


def make_case(**overrides):
    base = {"id": "c1", "prompt": "Say hello."}
    base.update(overrides)
    return EvalCase.from_dict(base)


def test_exact_match_pass_and_fail():
    assert exact_match("Paris", "paris")[0] == 1.0
    assert exact_match("  Paris\n", "Paris")[0] == 1.0  # stripped
    assert exact_match("London", "Paris")[0] == 0.0
    assert exact_match("Paris", "paris", case_sensitive=True)[0] == 0.0


def test_regex_match_and_invalid_pattern():
    assert regex_match("the answer is 42", r"\b42\b")[0] == 1.0
    assert regex_match("no digits here", r"\d+")[0] == 0.0
    score, detail = regex_match("x", r"([unclosed")
    assert score == 0.0 and "invalid regex" in detail


def test_contains_match_any_and_all():
    assert contains_match("The rubric gives a high score", ["rubric", "score"])[0] == 1.0
    assert contains_match("The rubric gives a high score", ["rubric", "score"], match="all")[0] == 1.0
    assert contains_match("nothing relevant", ["rubric", "score"], match="all")[0] == 0.0
    assert contains_match("NOTHING RELEVANT", ["nothing"], case_sensitive=True)[0] == 0.0


def test_max_length_budget():
    assert max_length("one two three", max_words=3)[0] == 1.0
    assert max_length("one two three four", max_words=3)[0] == 0.0
    assert max_length("abc", max_chars=2)[0] == 0.0


def test_judge_score_parses_and_clamps():
    judge = StubBackend(mode="canned", response="SCORE: 82\nSolid answer.")
    score, detail = judge_score("some output", "be helpful", judge)
    assert score == pytest.approx(0.82)
    assert "82" in detail

    judge = StubBackend(mode="canned", response="SCORE: 150\nToo generous.")
    assert judge_score("x", "y", judge)[0] == pytest.approx(1.0)  # clamped

    judge = StubBackend(mode="canned", response="no score here")
    score, detail = judge_score("x", "y", judge)
    assert score == 0.0 and "no parseable score" in detail


def test_score_case_weighted_aggregation():
    case = make_case(
        expected="Paris",
        rubric=[
            ScoringRule.from_dict({"type": "exact_match", "weight": 3.0}),
            ScoringRule.from_dict({"type": "contains", "expected": "Paris", "weight": 1.0}),
        ],
    )
    # exact_match fails (0.0 * 3) but contains passes (1.0 * 1) -> 0.25
    score, details = score_case(case, "The city is Paris.")
    assert score == pytest.approx(0.25)
    assert len(details) == 2

    # no rubric -> neutral 1.0
    assert score_case(make_case(), "anything")[0] == 1.0


def test_unknown_scorer_type_scores_zero():
    case = make_case(rubric=[ScoringRule.from_dict({"type": "telepathy"})])
    score, details = score_case(case, "output")
    assert score == 0.0
    assert "unknown scorer" in details[0].detail


def test_load_suite_yaml(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent("""\
        name: demo
        description: test suite
        version: "2"
        cases:
          - id: a
            prompt: "hi"
            expected: "hi"
            rubric:
              - type: exact_match
                weight: 1.0
    """))
    suite = load_suite(path)
    assert suite.name == "demo"
    assert suite.version == "2"
    assert len(suite.cases) == 1
    assert suite.cases[0].rubric[0].type == "exact_match"


def test_load_suite_rejects_bad_documents(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text("description: no name\ncases:\n  - {id: a, prompt: b}\n")
    with pytest.raises(SuiteError):
        load_suite(missing)

    dupes = tmp_path / "dupes.yaml"
    dupes.write_text("name: x\ncases:\n  - {id: a, prompt: b}\n  - {id: a, prompt: c}\n")
    with pytest.raises(SuiteError):
        load_suite(dupes)

    with pytest.raises(SuiteError):
        load_suite(tmp_path / "does-not-exist.yaml")


def test_load_suite_json(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text('{"name": "j", "cases": [{"id": "a", "prompt": "b"}]}')
    suite = load_suite(path)
    assert suite.name == "j" and len(suite.cases) == 1
