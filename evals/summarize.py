#!/usr/bin/env python3
"""Summarize per-model report.json files into a comparison table.

Usage: python3 evals/summarize.py results/*/report.json

Prints a Markdown table with overall, injection-robustness, and
capability pass rates plus mean latency per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slice_stats(report: dict, prefix: str) -> tuple[int, int]:
    cases = [c for c in report["cases"] if c["case_id"].startswith(prefix)]
    passed = sum(1 for c in cases if c["passed"])
    return passed, len(cases)


def main(paths: list[str]) -> None:
    rows = []
    for p in sorted(paths):
        report = load(Path(p))
        model = report["backend"].split(":", 1)[-1]
        inj_p, inj_n = slice_stats(report, "inj-")
        cap_p, cap_n = slice_stats(report, "cap-")
        s = report["summary"]
        rows.append(
            {
                "model": model,
                "inj": f"{inj_p}/{inj_n}",
                "cap": f"{cap_p}/{cap_n}",
                "overall": f"{s['passed']}/{s['total']}",
                "mean_score": f"{s['mean_score']:.3f}",
                "latency": f"{s['mean_latency_s']:.2f}s",
            }
        )
    print("| Model | Injection robustness | Capability | Overall | Mean score | Mean latency |")
    print("| ----- | -------------------- | ---------- | ------- | ---------- | ------------ |")
    for r in rows:
        print(
            f"| `{r['model']}` | {r['inj']} | {r['cap']} | {r['overall']} "
            f"| {r['mean_score']} | {r['latency']} |"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize.py results/*/report.json", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1:])
