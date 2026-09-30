"""Tests for the heuristic judge backend and the JSONL dataset loader."""

import json
import subprocess
import sys

import pytest

from llm_eval_harness import (
    HeuristicJudgeBackend,
    StubBackend,
    load_cases_jsonl,
    load_suite,
    run_suite,
    suite_from_jsonl,
)
from llm_eval_harness.scorers import JUDGE_PROMPT, judge_score
from llm_eval_harness.suite import EvalSuite, SuiteError


def judge_prompt(criteria: str, output: str) -> str:
    return JUDGE_PROMPT.format(criteria=criteria, output=output)


# ------------------------------------------------------- heuristic judge
def test_heuristic_judge_scores_good_answer_high():
    judge = HeuristicJudgeBackend()
    resp = judge.generate(
        judge_prompt(
            "The answer names Paris as the capital of France.",
            "Paris is the capital of France, a city of about two million people.",
        )
    )
    score, detail = judge_score("unused", "unused", _FixedJudge(resp.text))
    # 3 of 5 criteria terms hit ("paris", "capital", "france"); the
    # boilerplate terms "answer"/"names" are missed, which is expected.
    assert score > 0.55
    assert "SCORE:" in resp.text


def test_heuristic_judge_scores_irrelevant_answer_low():
    judge = HeuristicJudgeBackend()
    resp = judge.generate(
        judge_prompt(
            "The answer names Paris as the capital of France.",
            "France is a beautiful country with a long and storied history.",
        )
    )
    score, _ = judge_score("unused", "unused", _FixedJudge(resp.text))
    assert score < 0.5


def test_heuristic_judge_names_missed_terms():
    judge = HeuristicJudgeBackend()
    resp = judge.generate(judge_prompt("mention apples and oranges", "I like apples."))
    assert "oranges" in resp.text  # missed term called out
    assert "apples" in resp.text  # matched term called out


def test_heuristic_judge_rejects_unparseable_prompt():
    judge = HeuristicJudgeBackend()
    resp = judge.generate("this is not a judge prompt at all")
    score, detail = judge_score("x", "y", _FixedJudge(resp.text))
    assert score == 0.0
    assert "Could not parse" in resp.text


def test_heuristic_judge_flows_through_runner():
    from llm_eval_harness import EvalCase
    from llm_eval_harness.suite import ScoringRule

    case = EvalCase(
        id="j",
        prompt="What is the capital of France?",
        rubric=[ScoringRule(type="judge", params={"criteria": "names Paris"}, weight=1.0)],
    )
    suite = EvalSuite(name="jsuite", cases=[case])
    result = run_suite(
        suite,
        StubBackend(
            mode="canned",
            response=(
                "Paris is the capital of France. It sits on the Seine and is "
                "home to about two million residents in the city proper."
            ),
        ),
        judge_backend=HeuristicJudgeBackend(),
    )
    assert result.case_results[0].score > 0.5
    assert result.case_results[0].passed


class _FixedJudge:
    """Adapter: expose a canned reply string as a judge backend."""

    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, system=None, **kwargs):
        from llm_eval_harness.backends import ModelResponse

        return ModelResponse(text=self._text)


# ------------------------------------------------------------- JSONL loader
def write_jsonl(path, lines):
    out = []
    for line in lines:
        out.append("" if line == "" else json.dumps(line))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_load_cases_jsonl(tmp_path):
    p = tmp_path / "cases.jsonl"
    write_jsonl(p, [
        {"id": "a", "prompt": "hi", "expected": "hi",
         "rubric": [{"type": "exact_match"}]},
        {"id": "b", "prompt": "yo"},  # no rubric
        "",  # blank lines are skipped
    ])
    cases = load_cases_jsonl(p)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].rubric[0].type == "exact_match"
    assert cases[0].expected == "hi"
    assert cases[1].rubric == []


def test_suite_from_jsonl_applies_default_rubric(tmp_path):
    p = tmp_path / "cases.jsonl"
    write_jsonl(p, [
        {"id": "a", "prompt": "hi", "expected": "hi"},
        {"id": "b", "prompt": "yo", "rubric": [{"type": "contains", "expected": "yo"}]},
    ])
    suite = suite_from_jsonl(p, name="demo", default_rubric=[{"type": "exact_match"}])
    assert suite.name == "demo"
    assert suite.cases[0].rubric[0].type == "exact_match"  # default applied
    assert suite.cases[1].rubric[0].type == "contains"  # line rubric wins


def test_load_suite_dispatches_jsonl(tmp_path):
    p = tmp_path / "eval.jsonl"
    write_jsonl(p, [{"id": "a", "prompt": "hi"}])
    suite = load_suite(p)
    assert suite.name == "eval"
    assert len(suite.cases) == 1


def test_jsonl_malformed_line_reports_lineno(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a", "prompt": "hi"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SuiteError, match=":2:"):
        load_cases_jsonl(p)


def test_jsonl_missing_required_field_reports_lineno(tmp_path):
    p = tmp_path / "bad.jsonl"
    write_jsonl(p, [{"id": "a"}])  # no prompt
    with pytest.raises(SuiteError, match=":1:"):
        load_cases_jsonl(p)


def test_jsonl_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dup.jsonl"
    write_jsonl(p, [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}])
    with pytest.raises(SuiteError, match="duplicate"):
        load_cases_jsonl(p)


def test_jsonl_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SuiteError, match="no cases"):
        load_cases_jsonl(p)


# ------------------------------------------------------------------ CLI
def test_cli_heuristic_judge_end_to_end(tmp_path):
    suite_path = tmp_path / "suite.jsonl"
    write_jsonl(suite_path, [
        {"id": "good", "prompt": "Name the capital of France.",
         "rubric": [{"type": "judge",
                     "criteria": "the answer names Paris as the capital of France"}]},
    ])
    r = subprocess.run(
        [sys.executable, "-m", "llm_eval_harness", "run", str(suite_path),
         "--backend", "stub", "--stub-mode", "canned",
         "--stub-response", "Paris is the capital of France, with two million residents.",
         "--judge-backend", "heuristic", "--markdown"],
        capture_output=True, text=True, cwd="/tmp",
    )
    assert r.returncode == 0, r.stderr
    assert "passed=1/1" in r.stdout
