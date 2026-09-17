"""Run an eval suite against a backend and aggregate the results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_eval_harness.backends import ModelBackend
from llm_eval_harness.scorers import ScoreDetail, score_case
from llm_eval_harness.suite import EvalCase, EvalSuite


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    output: str
    score: float
    passed: bool
    details: list[ScoreDetail] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class SuiteResult:
    suite_name: str
    backend_name: str
    case_results: list[CaseResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def mean_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.score for r in self.case_results) / len(self.case_results)

    @property
    def mean_latency_s(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.latency_s for r in self.case_results) / len(self.case_results)

    def scorer_breakdown(self) -> dict[str, dict[str, float]]:
        """Per-scorer-type mean score and pass rate."""
        by_type: dict[str, list[ScoreDetail]] = {}
        for r in self.case_results:
            for d in r.details:
                by_type.setdefault(d.scorer, []).append(d)
        out: dict[str, dict[str, float]] = {}
        for scorer, details in by_type.items():
            out[scorer] = {
                "count": float(len(details)),
                "mean_score": sum(d.score for d in details) / len(details),
                "pass_rate": sum(1 for d in details if d.passed) / len(details),
            }
        return out

    def summary(self) -> str:
        return (
            f"suite={self.suite_name} backend={self.backend_name} "
            f"passed={self.passed}/{self.total} "
            f"pass_rate={self.pass_rate:.1%} mean_score={self.mean_score:.3f} "
            f"mean_latency={self.mean_latency_s:.2f}s"
        )


def run_case(case: EvalCase, backend: ModelBackend,
             judge_backend: Optional[ModelBackend] = None,
             pass_threshold: float = 0.5) -> CaseResult:
    """Run one case: generate, score, and wrap the outcome."""
    try:
        response = backend.generate(case.prompt, system=case.system_prompt)
        score, details = score_case(case, response.text, judge_backend)
        return CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            output=response.text,
            score=score,
            passed=score >= pass_threshold,
            details=details,
            latency_s=response.latency_s,
        )
    except Exception as exc:  # a backend blow-up fails the case, not the run
        return CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            output="",
            score=0.0,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_suite(suite: EvalSuite, backend: ModelBackend,
              judge_backend: Optional[ModelBackend] = None,
              pass_threshold: float = 0.5) -> SuiteResult:
    """Run every case in ``suite`` against ``backend`` sequentially."""
    start = time.perf_counter()
    results = [run_case(c, backend, judge_backend, pass_threshold) for c in suite.cases]
    return SuiteResult(
        suite_name=suite.name,
        backend_name=backend.name,
        case_results=results,
        duration_s=time.perf_counter() - start,
    )
