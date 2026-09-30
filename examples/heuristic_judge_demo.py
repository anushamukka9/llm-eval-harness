"""Runnable example: LLM-as-judge scoring with the local heuristic judge.

Writes a small JSONL case file (dataset-loader path), loads it as a suite,
then runs it against canned stub answers while the HeuristicJudgeBackend
scores each answer against the rubric criteria — no API key needed.

The heuristic judge rewards keyword overlap with the criteria, so a fluent
but off-topic answer scores low. It is a plumbing smoke test, not a real
evaluator; see HeuristicJudgeBackend's docstring for the honest caveats.

Run:  python examples/heuristic_judge_demo.py
"""

import json
import tempfile
from pathlib import Path

from llm_eval_harness import (
    HeuristicJudgeBackend,
    StubBackend,
    load_suite,
    run_suite,
    suite_from_jsonl,
    to_markdown,
)

CASES = [
    {
        "id": "refund-policy",
        "prompt": "Can I get a refund after 45 days?",
        "rubric": [
            {
                "type": "judge",
                "criteria": "The answer states the 30-day refund window and mentions contacting support.",
                "weight": 1.0,
            }
        ],
    },
    {
        "id": "capital",
        "prompt": "What is the capital of France?",
        "rubric": [
            {
                "type": "judge",
                "criteria": "The answer names Paris as the capital of France.",
                "weight": 1.0,
            }
        ],
    },
]

# A good answer to the refund question, a wrong-but-fluent answer to the
# capital question.
ANSWERS = {
    "refund-policy": (
        "Our refund window is 30 days from purchase. Since 45 days have "
        "passed you are outside it, but please contact support and we will "
        "review your case."
    ),
    "capital": (
        "France is a beautiful country in Western Europe, famous for its "
        "cuisine, art, and long history stretching back centuries."
    ),
}


class AnswerBackend(StubBackend):
    """Route each prompt to its canned answer by case id order."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(mode="canned")
        self._answers = answers
        self._n = 0
        self.name = "demo-answers"

    def generate(self, prompt, system=None, **kwargs):
        from llm_eval_harness.backends import ModelResponse
        import time

        start = time.perf_counter()
        text = self._answers[self._n % len(self._answers)]
        self._n += 1
        return ModelResponse(text=text, latency_s=time.perf_counter() - start)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "cases.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8"
        )

        print("=== JSONL dataset loader ===")
        suite = suite_from_jsonl(jsonl, name="heuristic-judge-demo")
        print(f"loaded {len(suite.cases)} cases from {jsonl.name}: "
              f"{[c.id for c in suite.cases]}")
        # load_suite dispatches on extension too:
        assert len(load_suite(jsonl).cases) == 2

        print()
        print("=== run with the heuristic judge (no API key) ===")
        backend = AnswerBackend([ANSWERS["refund-policy"], ANSWERS["capital"]])
        result = run_suite(suite, backend, judge_backend=HeuristicJudgeBackend())
        print(result.summary())
        for r in result.case_results:
            detail = r.details[0].detail if r.details else ""
            print(f"- {r.case_id}: score={r.score:.2f} passed={r.passed}")
            print(f"  judge says: {detail}")

        print()
        print("=== markdown report ===")
        print(to_markdown(result))


if __name__ == "__main__":
    main()
