"""Tests for backends, the runner, and the reporter."""

import json

import pytest

from llm_eval_harness import (
    EvalCase,
    EvalSuite,
    MockBackend,
    OpenAICompatibleBackend,
    StubBackend,
    load_suite,
    run_suite,
    to_json,
    to_markdown,
)
from llm_eval_harness.backends import BackendError
from llm_eval_harness.reporter import save_report
from llm_eval_harness.suite import ScoringRule


def demo_suite():
    cases = [
        EvalCase(id="pass", prompt="capital?", expected="Paris",
                 rubric=[ScoringRule(type="exact_match", weight=1.0)]),
        EvalCase(id="fail", prompt="capital?", expected="London",
                 rubric=[ScoringRule(type="exact_match", weight=1.0)]),
    ]
    return EvalSuite(name="demo", cases=cases)


def test_stub_backend_modes():
    assert StubBackend(mode="echo").generate("hello").text == "hello"
    assert StubBackend(mode="canned", response="fixed").generate("hello").text == "fixed"
    assert StubBackend(mode="template", response="Q: {prompt}").generate("hi").text == "Q: hi"
    with pytest.raises(ValueError):
        StubBackend(mode="bogus")


def test_mock_backend_routing(tmp_path):
    path = tmp_path / "mock.json"
    path.write_text(json.dumps({
        "default": "fallback",
        "routes": [
            {"pattern": "capital of France", "response": "Paris"},
            {"pattern": "\\d+ \\+ \\d+", "response": "a sum"},
        ],
    }))
    backend = MockBackend.from_file(path)
    assert backend.generate("What is the capital of France?").text == "Paris"
    assert backend.generate("compute 2 + 2").text == "a sum"
    assert backend.generate("something else").text == "fallback"


def test_run_suite_aggregation_and_summary():
    backend = StubBackend(mode="canned", response="Paris")
    result = run_suite(demo_suite(), backend)
    assert result.total == 2
    assert result.passed == 1
    assert result.pass_rate == pytest.approx(0.5)
    assert result.mean_score == pytest.approx(0.5)
    assert "passed=1/2" in result.summary()
    breakdown = result.scorer_breakdown()
    assert breakdown["exact_match"]["count"] == 2.0


def test_run_case_captures_backend_errors():
    class Boom:
        name = "boom"

        def generate(self, prompt, system=None, **kwargs):
            raise RuntimeError("kaput")

    result = run_suite(demo_suite(), Boom())  # type: ignore[arg-type]
    assert result.passed == 0
    assert all(r.error and "kaput" in r.error for r in result.case_results)


def test_quickstart_example_runs_end_to_end():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    suite = load_suite(root / "examples" / "quickstart" / "suite.yaml")
    backend = MockBackend.from_file(root / "examples" / "quickstart" / "mock_map.json")
    result = run_suite(suite, backend)
    assert result.total == 4
    assert result.passed == 3  # intentional-failure fails by design
    failed = [r for r in result.case_results if not r.passed]
    assert [r.case_id for r in failed] == ["intentional-failure"]


def test_reporters_render_and_save(tmp_path):
    backend = StubBackend(mode="canned", response="Paris")
    result = run_suite(demo_suite(), backend)

    payload = json.loads(to_json(result))
    assert payload["suite"] == "demo"
    assert payload["summary"]["total"] == 2
    assert len(payload["cases"]) == 2

    md = to_markdown(result)
    assert "# Eval report: demo" in md
    assert "`pass`" in md and "`fail`" in md

    paths = save_report(result, tmp_path / "report")
    assert paths["json"].is_file() and paths["markdown"].is_file()


def test_openai_backend_surfaces_http_errors(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(model="x", api_key="bad")
    with pytest.raises(BackendError, match="401"):
        backend.generate("hi")


def test_judge_backend_flows_through_runner():
    case = EvalCase(
        id="j", prompt="p",
        rubric=[ScoringRule(type="judge", params={"criteria": "be short"}, weight=1.0)],
    )
    suite = EvalSuite(name="jsuite", cases=[case])
    judge = StubBackend(mode="canned", response="SCORE: 90\nGood.")
    result = run_suite(suite, StubBackend(mode="canned", response="ok"), judge)
    assert result.case_results[0].score == pytest.approx(0.9)

    # without a judge backend the judge rule fails closed
    result2 = run_suite(suite, StubBackend(mode="canned", response="ok"))
    assert result2.case_results[0].score == 0.0
