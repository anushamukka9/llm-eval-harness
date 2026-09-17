"""llm-eval-harness: a lightweight harness for evaluating LLM outputs.

Define evaluation suites as YAML or JSON, run them against a pluggable
model backend, score each case with exact-match / regex / contains /
LLM-judge rubrics, and produce aggregated reports.

Typical usage::

    from llm_eval_harness import load_suite, StubBackend, run_suite

    suite = load_suite("examples/quickstart/suite.yaml")
    result = run_suite(suite, StubBackend(mode="echo"))
    print(result.summary())
"""

from llm_eval_harness.backends import (
    ModelBackend,
    ModelResponse,
    MockBackend,
    OpenAICompatibleBackend,
    StubBackend,
)
from llm_eval_harness.reporter import CaseResult, SuiteResult, to_json, to_markdown
from llm_eval_harness.runner import run_suite
from llm_eval_harness.scorers import ScoreDetail, score_case
from llm_eval_harness.suite import EvalCase, EvalSuite, ScoringRule, load_suite

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EvalCase",
    "EvalSuite",
    "ScoringRule",
    "ModelBackend",
    "ModelResponse",
    "StubBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "ScoreDetail",
    "score_case",
    "CaseResult",
    "SuiteResult",
    "run_suite",
    "to_json",
    "to_markdown",
    "load_suite",
]
