"""Command-line interface for llm-eval-harness.

Commands
--------
run     Run a suite YAML/JSON against a backend and write reports.
init    Scaffold a starter suite in the current directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from llm_eval_harness import __version__, load_suite
from llm_eval_harness.backends import (
    MockBackend,
    ModelBackend,
    OpenAICompatibleBackend,
    StubBackend,
)
from llm_eval_harness.reporter import save_report, to_markdown
from llm_eval_harness.runner import run_suite
from llm_eval_harness.suite import SuiteError

STARTER_SUITE = """\
name: "starter"
description: "Starter suite scaffolded by llm-eval init"
version: "1"
cases:
  - id: "echo-check"
    prompt: "hello"
    expected: "hello"
    rubric:
      - type: "exact_match"
        weight: 1.0
  - id: "length-check"
    prompt: "Say hi in three words."
    rubric:
      - type: "max_length"
        max_words: 5
"""


def build_backend(args: argparse.Namespace) -> ModelBackend:
    if args.backend == "stub":
        return StubBackend(mode=args.stub_mode, response=args.stub_response)
    if args.backend == "mock":
        if not args.mock_file:
            raise SuiteError("--backend mock requires --mock-file")
        return MockBackend.from_file(args.mock_file)
    if args.backend == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleBackend(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    raise SuiteError(f"unknown backend: {args.backend}")


def build_judge_backend(args: argparse.Namespace) -> ModelBackend | None:
    if not args.judge_backend:
        return None
    if args.judge_backend == "stub":
        return StubBackend(mode="canned", response=args.judge_stub_response)
    if args.judge_backend == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleBackend(
            model=args.judge_model or args.model,
            api_key=api_key,
            base_url=args.base_url,
        )
    raise SuiteError(f"unknown judge backend: {args.judge_backend}")


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    backend = build_backend(args)
    judge = build_judge_backend(args)
    result = run_suite(suite, backend, judge, pass_threshold=args.pass_threshold)
    print(result.summary())
    if args.out:
        paths = save_report(result, args.out)
        print(f"wrote {paths['json']}")
        print(f"wrote {paths['markdown']}")
    elif args.markdown:
        print()
        print(to_markdown(result))
    return 0 if result.passed == result.total else 1


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir or ".") / "suite.yaml"
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.write_text(STARTER_SUITE, encoding="utf-8")
    print(f"scaffolded {target}")
    print("run it with: llm-eval run suite.yaml --backend stub")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Run YAML/JSON eval suites against LLM backends and score the outputs.",
    )
    parser.add_argument("--version", action="version", version=f"llm-eval-harness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an eval suite")
    run_p.add_argument("suite", help="path to suite YAML or JSON")
    run_p.add_argument("--backend", default="stub",
                       choices=["stub", "mock", "openai"], help="model backend (default: stub)")
    run_p.add_argument("--stub-mode", default="echo",
                       choices=["echo", "canned", "template"], help="stub response mode")
    run_p.add_argument("--stub-response", default="", help="canned/template response for stub backend")
    run_p.add_argument("--mock-file", help="JSON route mapping for the mock backend")
    run_p.add_argument("--model", default="gpt-4o-mini", help="model name for openai backend")
    run_p.add_argument("--base-url", default="https://api.openai.com/v1",
                       help="base URL for OpenAI-compatible APIs (vLLM, Ollama, ...)")
    run_p.add_argument("--api-key", help="API key (or set OPENAI_API_KEY)")
    run_p.add_argument("--max-tokens", type=int, default=512)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--judge-backend", choices=["stub", "openai"],
                       help="backend used for 'judge' rubric rules")
    run_p.add_argument("--judge-model", help="model name for the judge backend")
    run_p.add_argument("--judge-stub-response", default="SCORE: 100\nLooks good.",
                       help="canned judge reply when --judge-backend stub")
    run_p.add_argument("--pass-threshold", type=float, default=0.5,
                       help="case score >= threshold counts as passed")
    run_p.add_argument("--out", help="directory for report.json / report.md")
    run_p.add_argument("--markdown", action="store_true", help="print the Markdown report to stdout")
    run_p.set_defaults(func=cmd_run)

    init_p = sub.add_parser("init", help="scaffold a starter suite.yaml")
    init_p.add_argument("--dir", default=".", help="directory to write suite.yaml into")
    init_p.add_argument("--force", action="store_true", help="overwrite an existing suite.yaml")
    init_p.set_defaults(func=cmd_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
