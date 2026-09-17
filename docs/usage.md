# Usage guide

This guide covers the full workflow: writing a suite, picking a backend,
running it, and reading the reports. Also see the
[quickstart example](../examples/quickstart/README.md).

## 1. Write a suite

A suite is a YAML (or JSON) file with a `name` and a `cases` list.
Each case has an `id`, a `prompt`, and a `rubric` — a list of scoring
rules. A rule needs a `type` and may carry a `weight` plus type-specific
params. The case score is the weight-weighted mean of its rules, and a
case passes when its score is `>= pass_threshold` (default `0.5`).

```yaml
name: "regression-v3"
description: "Weekly regression checks for the support bot"
version: "3"
cases:
  - id: "refund-policy"
    system_prompt: "You are a helpful support agent."
    prompt: "Can I get a refund after 45 days?"
    rubric:
      - type: contains
        expected: ["30 days", "refund"]
        match: all          # every phrase must appear
        weight: 2.0
      - type: max_length
        max_words: 120
        weight: 0.5
```

## 2. Scorer reference

| Type | What it checks | Key params |
| ---- | -------------- | ---------- |
| `exact_match` | output equals `expected` (stripped, case-insensitive by default) | `expected`, `case_sensitive` |
| `regex` | `pattern` matches somewhere in the output | `pattern` |
| `contains` | phrase(s) appear in the output | `expected` (string or list), `match: any\|all` |
| `max_length` | output stays within budget | `max_words`, `max_chars` |
| `judge` | a judge model rates the output against `criteria` | `criteria` (requires a judge backend) |

Any rule also accepts `pass_threshold` (default `0.5`) to decide its own
pass/fail, and `weight` (default `1.0`) for the aggregation. If a rule
omits `expected`, the case-level `expected` field is used as fallback.

The `judge` scorer sends the output to a second model with a rubric
prompt and parses a `SCORE: <0-100>` line from its reply. Scores clamp
to the 0–100 range; an unparseable reply fails the rule.

## 3. Backends

- `stub` — deterministic canned responses (`echo`, `canned`, `template`
  modes). Perfect for CI and for testing your suite definitions.
- `mock` — regex-routed responses from a JSON mapping file. Give it to a
  teammate and they reproduce your exact run.
- `openai` — any OpenAI-compatible `/chat/completions` endpoint
  (OpenAI, vLLM, Ollama, ...). Uses only the standard library.

```bash
# reproducible demo, no API key
llm-eval run suite.yaml --backend mock --mock-file mock_map.json --out report

# real model via an OpenAI-compatible server
export OPENAI_API_KEY=...
llm-eval run suite.yaml --backend openai --model gpt-4o-mini --out report

# local vLLM / Ollama server
llm-eval run suite.yaml --backend openai --model llama3 \
    --base-url http://localhost:8000/v1 --out report

# judge-model scoring with a stub judge (deterministic)
llm-eval run suite.yaml --backend mock --mock-file mock_map.json \
    --judge-backend stub --out report
```

## 4. Reports

`--out DIR` writes `report.json` (full machine-readable results) and
`report.md` (a summary table plus per-scorer breakdown). The CLI prints
a one-line summary, e.g.
`suite=quickstart backend=mock passed=3/4 pass_rate=75.0% ...`, and exits
nonzero when any case fails — handy as a CI gate.

`llm-eval init` scaffolds a starter `suite.yaml` in the current directory.

## 5. Python API

```python
from llm_eval_harness import load_suite, run_suite, to_markdown
from llm_eval_harness import MockBackend

suite = load_suite("suite.yaml")
backend = MockBackend.from_file("mock_map.json")
result = run_suite(suite, backend)
print(result.summary())
print(to_markdown(result))
```
