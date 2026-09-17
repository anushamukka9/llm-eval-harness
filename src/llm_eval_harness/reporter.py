"""Render :class:`SuiteResult` as JSON or Markdown reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_eval_harness.runner import CaseResult, SuiteResult


def to_json(result: SuiteResult) -> str:
    """Serialize the full result (summary + per-case detail) as JSON."""
    payload = {
        "suite": result.suite_name,
        "backend": result.backend_name,
        "duration_s": round(result.duration_s, 3),
        "summary": {
            "total": result.total,
            "passed": result.passed,
            "pass_rate": round(result.pass_rate, 4),
            "mean_score": round(result.mean_score, 4),
            "mean_latency_s": round(result.mean_latency_s, 4),
        },
        "scorer_breakdown": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in result.scorer_breakdown().items()
        },
        "cases": [asdict(r) for r in result.case_results],
    }
    return json.dumps(payload, indent=2)


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_markdown(result: SuiteResult) -> str:
    """Render a human-readable Markdown report with a per-case table."""
    lines = [
        f"# Eval report: {result.suite_name}",
        "",
        f"- **Backend:** `{result.backend_name}`",
        f"- **Passed:** {result.passed}/{result.total} ({result.pass_rate:.1%})",
        f"- **Mean score:** {result.mean_score:.3f}",
        f"- **Mean latency:** {result.mean_latency_s:.2f}s",
        f"- **Duration:** {result.duration_s:.1f}s",
        "",
        "## Cases",
        "",
        "| Case | Score | Pass | Latency | Output |",
        "| ---- | ----- | ---- | ------- | ------ |",
    ]
    for r in result.case_results:
        status = "✅" if r.passed else "❌"
        shown = _short(r.error or r.output)
        lines.append(
            f"| `{r.case_id}` | {r.score:.2f} | {status} | {r.latency_s:.2f}s | {shown} |"
        )
    breakdown = result.scorer_breakdown()
    if breakdown:
        lines += ["", "## Scorer breakdown", "",
                  "| Scorer | Checks | Mean score | Pass rate |",
                  "| ------ | ------ | ---------- | --------- |"]
        for scorer, stats in sorted(breakdown.items()):
            lines.append(
                f"| `{scorer}` | {int(stats['count'])} | {stats['mean_score']:.3f} "
                f"| {stats['pass_rate']:.1%} |"
            )
    lines.append("")
    return "\n".join(lines)


def save_report(result: SuiteResult, out_dir: str | Path) -> dict[str, Path]:
    """Write ``report.json`` and ``report.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "report.json",
        "markdown": out_dir / "report.md",
    }
    paths["json"].write_text(to_json(result), encoding="utf-8")
    paths["markdown"].write_text(to_markdown(result), encoding="utf-8")
    return paths
