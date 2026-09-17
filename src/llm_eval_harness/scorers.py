"""Scoring rules: turn a model output into a numeric score.

Each scorer returns a :class:`ScoreDetail` with a 0..1 ``score`` and a
boolean ``passed`` (``score >= pass_threshold``, default 0.5 unless the
rule overrides it). :func:`score_case` aggregates a case's rubric with a
weight-weighted mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_eval_harness.suite import EvalCase


@dataclass
class ScoreDetail:
    scorer: str
    score: float
    passed: bool
    detail: str
    weight: float = 1.0


def _passed(score: float, params: dict[str, Any]) -> bool:
    return score >= float(params.get("pass_threshold", 0.5))


def exact_match(output: str, expected: str, case_sensitive: bool = False,
               strip: bool = True) -> tuple[float, str]:
    """1.0 when ``output`` equals ``expected`` (after optional strip)."""
    a, b = (output, expected) if case_sensitive else (output.lower(), expected.lower())
    if strip:
        a, b = a.strip(), b.strip()
    ok = a == b
    return (1.0 if ok else 0.0, "exact match" if ok else f"expected {expected!r}, got {output!r}")


def regex_match(output: str, pattern: str) -> tuple[float, str]:
    """1.0 when ``pattern`` matches anywhere in ``output``."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        return 0.0, f"invalid regex {pattern!r}: {exc}"
    m = compiled.search(output)
    return (1.0, f"matched {m.group(0)!r}") if m else (0.0, f"no match for {pattern!r}")


def contains_match(output: str, expected: str | list[str], match: str = "any",
                   case_sensitive: bool = False) -> tuple[float, str]:
    """1.0 when all (``match="all"``) or any of the phrases are present."""
    phrases = [expected] if isinstance(expected, str) else list(expected)
    hay = output if case_sensitive else output.lower()
    hits = [p for p in phrases if (p if case_sensitive else p.lower()) in hay]
    if match == "all":
        ok, detail = len(hits) == len(phrases), f"{len(hits)}/{len(phrases)} phrases found"
    else:
        ok, detail = bool(hits), f"found {hits[0]!r}" if hits else "no phrase found"
    return (1.0 if ok else 0.0, detail)


def max_length(output: str, max_words: int | None = None,
               max_chars: int | None = None) -> tuple[float, str]:
    """1.0 when the output stays within the given length budget."""
    problems = []
    if max_words is not None:
        n = len(output.split())
        if n > max_words:
            problems.append(f"{n} words > {max_words} max")
    if max_chars is not None and len(output) > max_chars:
        problems.append(f"{len(output)} chars > {max_chars} max")
    return (0.0, "; ".join(problems)) if problems else (1.0, "within length budget")


JUDGE_PROMPT = """\
You are an impartial evaluator. Rate the assistant's answer against the criteria.

Criteria: {criteria}

Assistant's answer:
---
{output}
---

Reply with exactly one line in the form: SCORE: <number between 0 and 100>
followed by a one-sentence justification on the next line.
"""


def judge_score(output: str, criteria: str, judge_backend: Any) -> tuple[float, str]:
    """Ask a judge model to score ``output``; parse ``SCORE: <0-100>``."""
    response = judge_backend.generate(JUDGE_PROMPT.format(criteria=criteria, output=output))
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response.text)
    if not m:
        return 0.0, f"judge returned no parseable score: {response.text[:120]!r}"
    raw = max(0.0, min(100.0, float(m.group(1))))
    return raw / 100.0, f"judge score {raw:.0f}/100"


def score_rule(rule_type: str, output: str, params: dict[str, Any],
               case: EvalCase, judge_backend: Any = None) -> ScoreDetail:
    """Score one rubric rule against a model output."""
    weight = float(params.get("weight", 1.0))
    rule_params = {k: v for k, v in params.items() if k != "weight"}

    if rule_type == "exact_match":
        expected = rule_params.get("expected", case.expected)
        if expected is None:
            return ScoreDetail(rule_type, 0.0, False, "no expected value provided", weight)
        score, detail = exact_match(output, str(expected),
                                    case_sensitive=bool(rule_params.get("case_sensitive", False)))
    elif rule_type == "regex":
        pattern = rule_params.get("pattern", "")
        score, detail = regex_match(output, str(pattern))
    elif rule_type == "contains":
        expected = rule_params.get("expected", case.expected)
        if expected is None:
            return ScoreDetail(rule_type, 0.0, False, "no expected value provided", weight)
        score, detail = contains_match(output, expected,
                                       match=str(rule_params.get("match", "any")),
                                       case_sensitive=bool(rule_params.get("case_sensitive", False)))
    elif rule_type == "max_length":
        score, detail = max_length(output,
                                   max_words=rule_params.get("max_words"),
                                   max_chars=rule_params.get("max_chars"))
    elif rule_type == "judge":
        if judge_backend is None:
            return ScoreDetail(rule_type, 0.0, False, "judge scorer needs --judge-backend", weight)
        criteria = rule_params.get("criteria", case.expected or "Answer the prompt well.")
        score, detail = judge_score(output, str(criteria), judge_backend)
    else:
        return ScoreDetail(rule_type, 0.0, False, f"unknown scorer type {rule_type!r}", weight)

    return ScoreDetail(rule_type, score, _passed(score, rule_params), detail, weight)


def score_case(case: EvalCase, output: str,
               judge_backend: Optional[Any] = None) -> tuple[float, list[ScoreDetail]]:
    """Score ``output`` against the case's whole rubric.

    Returns the weight-weighted mean score plus the per-rule details.
    A case with no rubric scores a neutral 1.0 (nothing to check).
    """
    if not case.rubric:
        return 1.0, []
    details = [score_rule(r.type, output, {**r.params, "weight": r.weight}, case, judge_backend)
               for r in case.rubric]
    total_weight = sum(d.weight for d in details) or 1.0
    aggregate = sum(d.score * d.weight for d in details) / total_weight
    return aggregate, details
