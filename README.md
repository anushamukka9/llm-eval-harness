# llm-eval-harness

A lightweight harness for evaluating LLM outputs. Define eval suites as
YAML or JSON, run them against a pluggable model backend, score each case
with rubric rules (exact match, regex, contains, length budget, or an
LLM judge), and get aggregated reports — from the CLI or the Python API.

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and `pyyaml`.

## Quickstart

No API key needed — the quickstart runs against the bundled mock backend:

```bash
llm-eval run examples/quickstart/suite.yaml \
    --backend mock --mock-file examples/quickstart/mock_map.json \
    --out report --markdown
```

Expected output: `passed=3/4` — one case is designed to fail so you can see
what failures look like in `report/report.md` and `report/report.json`.

Scaffold your own suite:

```bash
mkdir my-evals && cd my-evals
llm-eval init
llm-eval run suite.yaml --backend stub --out report
```

With a real model (any OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY=...
llm-eval run suite.yaml --backend openai --model gpt-4o-mini --out report
```

## API

```python
from llm_eval_harness import load_suite, run_suite, to_markdown
from llm_eval_harness import MockBackend

suite = load_suite("suite.yaml")
result = run_suite(suite, MockBackend.from_file("mock_map.json"))
print(result.summary())   # suite=... backend=mock passed=3/4 pass_rate=75.0% ...
print(to_markdown(result))
```

## Architecture

```
src/llm_eval_harness/
├── suite.py      # EvalSuite / EvalCase / ScoringRule: YAML/JSON loading + validation
├── backends.py   # ModelBackend interface: Stub, Mock (regex-routed), OpenAI-compatible
├── scorers.py    # exact_match, regex, contains, max_length, judge; weighted aggregation
├── runner.py     # run_suite / run_case: generation, scoring, stats (SuiteResult)
├── reporter.py   # report.json + report.md rendering
└── cli.py        # `llm-eval run|init` command-line interface
```

**Design notes:**

- Scorers return a 0–1 score plus a boolean pass, so partial credit and
  per-rule thresholds compose naturally; a case's score is the
  weight-weighted mean of its rubric.
- Backends implement one method (`generate`) and return a `ModelResponse`
  with text, latency, and metadata — adding a new backend is ~20 lines.
- The `judge` scorer talks to a *separate* judge backend and parses a
  `SCORE: <0-100>` line, keeping the model under test and the evaluator
  decoupled.
- Backend failures fail the case, not the run, so one flaky endpoint
  doesn't lose the whole report.

See [docs/usage.md](docs/usage.md) for the full usage guide and scorer reference.

## Author

Anusha Mukka — [anushamukka.com](https://anushamukka.com)

## License

MIT — see [LICENSE](LICENSE).
