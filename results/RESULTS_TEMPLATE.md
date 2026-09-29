# Eval results

Results from running `evals/model-eval-suite.yaml` against real models.
Fill this in after a run; `./evals/run_real_models.sh` prints the table.

## Method

- Harness: llm-eval-harness (`llm-eval run`), OpenAI-compatible backend, temperature 0.
- Suite: 16 cases (10 injection-robustness, 6 capability), one run each.
- Date run: <!-- fill in -->
- Harness version / commit: <!-- fill in -->

## Results

| Model | Injection robustness | Capability | Overall | Mean score | Mean latency |
| ----- | -------------------- | ---------- | ------- | ---------- | ------------ |
| `gpt-4o-mini` | _/_ | _/_ | _/_ | _ | _ |
| `gpt-4o` | _/_ | _/_ | _/_ | _ | _ |
| `o4-mini` | _/_ | _/_ | _/_ | _ | _ |

Per-model detail: `results/<model>/report.md` (and `report.json` for machines).

## Reading the numbers

- The injection slice is scored by refusal-phrase detection. It measures
  whether the model said no, not whether it is actually safe. See
  `evals/README.md` for the full list of limitations.
- With 16 cases and a single run per model, treat gaps of 1-2 cases as
  noise, not signal.
