#!/usr/bin/env bash
# Run the model-eval suite against real models and print a comparison table.
#
# Usage:
#   export OPENAI_API_KEY=...          # or pass --api-key
#   ./evals/run_real_models.sh [extra llm-eval args, e.g. --base-url ...]
#
# Models to test (space-separated):
#   MODELS="gpt-4o-mini gpt-4o o4-mini" ./evals/run_real_models.sh
#
# Any OpenAI-compatible endpoint works: pass --base-url and --api-key for
# OpenRouter, vLLM, Ollama, etc. Results land in results/<model>/ and the
# script finishes by printing the cross-model comparison table.
#
# Estimated cost for the default 3 models x 16 cases: under $0.10 total.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-gpt-4o-mini gpt-4o o4-mini}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "error: OPENAI_API_KEY is not set (or pass --api-key). No key, no run." >&2
  exit 1
fi

for model in $MODELS; do
  echo "=== $model ==="
  llm-eval run evals/model-eval-suite.yaml \
    --backend openai \
    --model "$model" \
    --temperature 0 \
    --out "results/$model" \
    "$@" || echo "warning: $model run did not fully pass"
done

echo
echo "=== comparison ==="
python3 evals/summarize.py results/*/report.json
