# Run this demo end to end (no API key needed):
#
#   llm-eval run suite.yaml --backend mock --mock-file mock_map.json \
#       --out report --markdown
#
# Expected: 3/4 cases pass; `intentional-failure` fails on purpose so the
# generated report shows what a failing case looks like. Inspect
# `report/report.md` and `report/report.json` afterwards.
#
# With a real model (any OpenAI-compatible endpoint):
#
#   export OPENAI_API_KEY=...
#   llm-eval run suite.yaml --backend openai --model gpt-4o-mini --out report
