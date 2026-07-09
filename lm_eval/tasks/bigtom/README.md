# BigToM (lm-eval adapter)

Faithful lm-eval port of **BigToM** - "Understanding Social Reasoning in LLMs with LLMs"
(arXiv:2306.15448). The source benchmark lives at
`benchmarks/procedural-evals-tom/` and is treated as read-only.

## Architecture

The source repo ships 200 raw multi-variant rows in
`benchmarks/procedural-evals-tom/data/bigtom/bigtom.csv` plus
`code/src/generate_conditions.py`, which splices each story into per-condition examples. There is
no frozen released set of expanded test items. `utils.py` vendors that expansion logic as pure
Python and materializes docs in memory.

Grading follows the repo's `--mcq` protocol in `evaluate_conditions.py`: the model sees both
options as `a)`/`b)`, generates free text, and is graded by a correct-key-first substring check.
The original fallback LLM judge for neither-letter-found cases is replaced by deterministic
option-text matching; if that also fails, the doc is counted under `error_rate`.

The fixed one-shot Kofi/fishing-net exemplar is supplied through lm-eval's native
`fewshot_config.samples`, so `--apply_chat_template --fewshot_as_multiturn` renders real separate
turns matching the original chat prompt shape.

## Tasks

Registered leaves are intentionally coarse:

`{0shot, 0shot_cot, 1shot, 1shot_cot} x {backward_belief, forward_action, forward_belief}`

That is 12 leaf tasks plus 4 groups:

- `bigtom_0shot`
- `bigtom_0shot_cot`
- `bigtom_1shot`
- `bigtom_1shot_cot`

Each leaf loads all 8 core cells for one variable:

`init_belief{0,1} x condition{true_belief, false_belief, true_control, false_control}`

So each leaf has 1,600 docs. The old cell-level granularity is reported through metric names
rather than separate YAML files.

The 24 core cells match the paper's own results-formatting script
(`code/analysis/format_model_results.py`, `VARIABLES = ['belief', 'action']` x both directions).
`percept_to_belief` remains implemented in `utils.py` because it exists in the source expansion
script, but it is not registered as a task because the paper formatting script excludes it.

## Metrics

- `acc`: marginal mean over all 8 cells in a variable leaf.
- `error_rate`: fraction of docs where neither the letter nor option text could be parsed.
- `acc_<init>_<condition>`: marginal accuracy for a cell, e.g. `acc_0_false_belief`.
- `error_rate_<init>_<condition>`: matching parse-failure diagnostic by cell.
- `tb_fb_acc_<init>_<pair_kind>`: paper-faithful paired contingency metric, e.g.
  `tb_fb_acc_0_belief` or `tb_fb_acc_1_control`.

`tb_fb_acc_*` pairs rows by `(variable, init_belief, pair_kind, raw_idx)` and scores 1.0 only
when both the true-side row and its false-side row are correct. This is BigToM's `TB & FB`
contingency metric while preserving ordinary marginal `acc`.

Group rows aggregate the three variable leaves with an unweighted mean. Since every variable has
the same number of examples for every reported cell, this is equivalent to a pooled average for
the core BigToM grid.

## Running

```bash
# One variable leaf
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=gpt2 \
  --tasks bigtom_0shot_forward_belief --limit 5 --log_samples --output_path out/

# Full core groups, chat model
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<chat-model> \
  --tasks bigtom_0shot,bigtom_0shot_cot,bigtom_1shot,bigtom_1shot_cot \
  --apply_chat_template --fewshot_as_multiturn --batch_size auto --output_path out/ --log_samples
```

Reasoning-capable chat models such as Qwen3 and DeepSeek-R1 think by default. The task therefore
uses no blank-line stop and keeps `max_gen_toks: 1024` for all four variants; scoring strips text
before the last `</think>` before applying the BigToM answer extractor.

## Faithfulness Notes

1. Uses `generate_until`, not `multiple_choice`, because the source protocol grades generated
   free text after showing both answer options.
2. Replaces the source repo's rare LLM-judge fallback with deterministic option-text matching and
   an explicit `error_rate` metric.
3. Preserves the source's deterministic answer shuffle with `random.Random(0)` consumed in raw-row
   order per cell.
4. Builds all 4 prompting variants: `0shot`, `0shot_cot`, `1shot`, `1shot_cot`.
5. Excludes `backward_desire`; it is dead code in `generate_conditions.py` because the driving
   variable loop never includes it.
6. Preserves the inherited `backward_belief` control quirk: `true_control` and `false_control`
   are byte-identical in the source expansion logic.
7. Keeps the source `parse_chat_response` behavior for CoT parsing, including its absent-`Answer:`
   slicing quirk.
8. Uses full 200-item cells, not the paper scripts' occasional `-n 100` cost-driven subsets.
9. Uses greedy decoding (`do_sample: false`, `temperature: 0.0`).

## Provenance

- Expansion logic: `code/src/generate_conditions.py`
- Grading and prompt logic: `code/src/evaluate_conditions.py`, `code/src/evaluate_llm.py`
- System instructions: `code/prompt_instructions/evaluate.txt`,
  `code/prompt_instructions/evaluate_cot_chat_model.txt`
- Headline-table scope: `code/analysis/format_model_results.py`
