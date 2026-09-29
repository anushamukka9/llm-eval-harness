# Real-model evals

This directory holds everything needed to run the eval suite against
real models. No API key is committed here; bring your own.

## One command

```bash
pip install -e .
export OPENAI_API_KEY=...
chmod +x evals/run_real_models.sh   # one time; the API upload does not preserve the exec bit
./evals/run_real_models.sh
```

That runs `evals/model-eval-suite.yaml` against three models
(`gpt-4o-mini`, `gpt-4o`, `o4-mini` by default; override with
`MODELS="..."`), writes per-model `report.json` / `report.md` to
`results/<model>/`, and prints a cross-model comparison table.

Any OpenAI-compatible endpoint works:

```bash
./evals/run_real_models.sh --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"
```

## The suite

16 cases in two slices:

- **Injection robustness** (`inj-*`, 10 cases): classic prompt-injection
  shapes drawn from the [llm-sentinel](https://github.com/anushamukka9/llm-sentinel)
  test corpus (instruction override, delimiter smuggling, DAN/developer-mode
  jailbreaks, roleplay extraction, prompt extraction, base64 payload).
  A pass means the model refused or did not comply, detected by
  refusal-phrase matching.
- **Capability** (`cap-*`, 6 cases): factual QA, arithmetic, summarization
  with a length budget, exact instruction following, structured JSON output,
  simple reasoning.

## Honest limitations

- Refusal detection is keyword-based. A model can pass by apologizing while
  complying, or fail with a safe completion that uses none of the phrases.
  This is a smoke signal, not a security certification.
- 16 cases, one run each, temperature 0. No variance analysis; treat small
  differences between models as noise.
- The injection cases are textbook shapes. A targeted red-team would be
  harder and this suite does not claim to be one.
