# SoNLI - lm-eval adapter

Adapter for **SocialNLI: A Dialogue-Centric Social Inference Dataset**
(Deo, Sanders, Van Durme; arXiv:2510.05458).

SoNLI is a scalar social-inference benchmark: each item gives a multi-party
Friends dialogue, a motivating question, and a free-form inference. The gold is
a human plausibility score in `[0, 1]`. The paper's primary metric is Pearson
correlation with human scores; MAE is the companion calibration metric.

## Tasks

| Task | Role | Metric |
|---|---|---|
| `sonli-proxy` | In-harness direct scalar plausibility proxy over `eval.json` | `pearson`, `mae`, `parse_rate` |
| `sonli_supporting` | Stage-1 generation of supporting counterfactual explanations | `nonempty_rate` only; use `--predict_only --log_samples` |
| `sonli_opposing` | Stage-1 generation of opposing counterfactual explanations | `nonempty_rate` only; use `--predict_only --log_samples` |
| `sonli_judge` | Stage-2 judge scoring over materialized generated explanations | Bayes-posterior `pearson`, `mae`, `parse_rate`, `paired_rate` |

## Architecture

`sonli-proxy` uses the 1,400-row human evaluation split from
`benchmarks/SoNLI/datasets/socialnli/eval.json`. Each doc is one
`(dialogue, question, inference)` triple. The prompt asks the model to emit a
single `SCORE: <number>` in `[0, 1]`; `process_results` parses the score and
custom aggregation functions compute Pearson and MAE against
`human_annotated_score`.

The full paper protocol is multi-stage: the evaluated model generates a
supporting explanation, generates an opposing explanation, judges each
explanation on the benchmark's 0-10 rubric, then combines the normalized scores
with the benchmark's Bayes posterior formula. lm-eval cannot pass one task's
generation directly into another task, so this adapter keeps the generation and
judge prompts in lm-eval and uses `offline_score.py` for the handoff.

## Full Pipeline Path

For HF chat models, use `--apply_chat_template` on every `sonli_supporting`,
`sonli_opposing`, and `sonli_judge` run when emulating the source API path,
which sends each prompt as one user message. Omit it only when intentionally
emulating the source local-vLLM path, which passes raw prompt text. Do not
compare runs unless this serialization choice is held fixed and reported.

Run the stage-1 explanation prompts:

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks sonli_supporting --include_path lm-evaluation-harness/lm_eval/tasks \
  --predict_only --log_samples --output_path outputs/sonli_supporting

PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks sonli_opposing --include_path lm-evaluation-harness/lm_eval/tasks \
  --predict_only --log_samples --output_path outputs/sonli_opposing
```

Prepare judge inputs from the two `samples_sonli_*.jsonl` files:

```bash
python lm-evaluation-harness/lm_eval/tasks/sonli/offline_score.py prepare \
  --support-samples outputs/sonli_supporting/<model>/samples_sonli_supporting_*.jsonl \
  --oppose-samples outputs/sonli_opposing/<model>/samples_sonli_opposing_*.jsonl \
  --output outputs/sonli_offline/judge_inputs.jsonl
```

Run the benchmark judge prompt in lm-eval:

```bash
$env:SONLI_JUDGE_DATA = "outputs/sonli_offline/judge_inputs.jsonl"
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks sonli_judge --include_path lm-evaluation-harness/lm_eval/tasks \
  --log_samples --output_path outputs/sonli_judge
```

Optionally recompute the same final metrics from the judge sample log:

```bash
python lm-evaluation-harness/lm_eval/tasks/sonli/offline_score.py score \
  --judge-samples outputs/sonli_judge/<model>/samples_sonli_judge_*.jsonl \
  --output outputs/sonli_offline/final_scores.json
```

## Direct Task Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks sonli-proxy --include_path lm-evaluation-harness/lm_eval/tasks \
  --limit 10 --log_samples --output_path outputs/sonli_proxy
```

Use `--apply_chat_template` for chat/instruction models when comparing chat
runs. The direct prompt is a documented proxy, not the paper's counterfactual
pipeline.

## Metrics

- `pearson`: Pearson correlation between parsed model plausibility and human
  score. This is the paper's primary signal.
- `mae`: mean absolute error against the human score.
- `parse_rate`: fraction of generations from which a valid numeric score was
  parsed.
- `paired_rate` (`sonli_judge` only): fraction of items with both supporting
  and opposing judge rows available, allowing the Bayes posterior to be
  computed even when one side failed numeric parsing.

`pearson` and `mae` are computed over parsed direct predictions for
`sonli-proxy`, and over complete support/opposition row pairs for `sonli_judge`.
For `sonli_judge`, failed judge parses keep `parsed: false` for `parse_rate` but
enter Bayes scoring as `-1.0`, matching the source pipeline's clamped
parse-failure behavior. Treat low `parse_rate` as a model/output-format warning;
low `paired_rate` means the offline stage handoff is incomplete.

## Faithfulness Deviations

1. **Direct `sonli-proxy` is a proxy.** The paper does not ask the model for one
   direct scalar. It evaluates a multi-stage counterfactual reasoning pipeline.
   The direct task is included because scalar regression, extraction, and
   Pearson/MAE aggregation are expressible in lm-eval and useful for quick
   model comparisons.
2. **Stage handoff is offline.** lm-eval cannot feed `sonli_supporting` and
   `sonli_opposing` generations directly into a downstream judge task. The
   adapter writes/reads sample logs and materializes `sonli_judge` JSONL through
   `offline_score.py`.
3. **Prompt provenance.** `sonli_supporting`, `sonli_opposing`, and
   `sonli_judge` vendor the benchmark's prompt strings from
   `src/prompts/*` and the judge score parser/Bayes formula from
   `src/experiments/experiment_one/experiment_one.py`.
4. **Decoding.** Stage tasks use the benchmark constants `TEMPERATURE=0.7` and
   `MAX_TOKENS=5000`. `sonli-proxy` direct uses greedy decoding with a short
   numeric budget because it is not a paper protocol and should be reproducible.
5. **Parse failures.** The benchmark pipeline expects a numeric judge score but
   normalizes failed parses to `-1`; the Bayes helper clamps that to the 0-side
   of the scale. `sonli_judge` reproduces that behavior for Pearson/MAE while
   still reporting `parse_rate` as a compliance diagnostic.

## Validation Notes

The released split has 1,400 rows. Natural partitions are:

- `classification`: concerning reality 560, belief 429, emotion 411.
- `inference_type`: cot 707, no_cot 693.
- `model`: gpt-4o 707, gpt-3.5-turbo-1106 693.

`offline_score.py released-baseline` computes Pearson/MAE for the released
`counterfactual_score` against human scores as a model-free sanity check.
