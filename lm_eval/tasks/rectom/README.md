# RecToM (lm-eval adapter)

Faithful lm-eval port of **RecToM: A Benchmark for Evaluating Machine Theory of
Mind in LLM-based Conversational Recommender Systems** (Li, Shi, Deng;
arXiv:2511.22275, accepted by AAAI 2026). The local benchmark submodule ships
the static data under `benchmarks/RecToM/data/` and OpenAI-style eval scripts
under `benchmarks/RecToM/evaluate/`; neither is modified.

## Architecture

RecToM is a per-item generative MCQ protocol, not lm-eval `multiple_choice`:
the official scripts send a dialogue/question/options prompt to a chat model,
generate letter(s), parse a set of letters, and score exact set equality.

This adapter defines two paper variants:

- `rectom`: vanilla zero-shot prompting.
- `rectom_cot`: CoT prompting with the official "Let's think step by step."
  cue and final-answer regex.

Each group exposes the 10 source metric columns reported by the paper/code.
Groups are only collections of leaf rows; they intentionally do not report a
benchmark-wide aggregate because the released scripts do not compute one.

| runnable task suffix | paper/code column | source file | n |
|---|---|---|---:|
| `fine_intent_rec` | Fine Intention (Rec) | `1_intent_rec.json` | 2205 |
| `coarse_intent_rec` | Coarse Intention (Rec) | `1_coarse_intent_rec.json` | 2205 |
| `belief_rec` | Belief (Rec) | `8_belief_rec_2_com.json` | 1762 |
| `fine_intent_seeker` | Fine Intention (Seek) | `2_intent_seeker.json` | 2205 |
| `coarse_intent_seeker` | Coarse Intention (Seek) | `2_coarse_intent_seeker.json` | 2205 |
| `desire_seeker` | Desire (Seek) | `7_desire_seeker_com.json` | 1448 |
| `prediction_rec` | Prediction (Rec) | `3_pred_rec.json` | 2098 |
| `judgement_rec` | Judgement (Rec) | `5_reverse_judge_rec.json` | 2098 |
| `prediction_seeker` | Prediction (Seek) | `4_pred_seeker.json` | 2149 |
| `judgement_seeker` | Judgement (Seek) | `6_judge_seeker.json` | 2149 |

Belief and desire are intentionally one-sided in RecToM. The paper defines the
belief task as the **recommender's belief** about the seeker's attitude toward a
movie, and the desire task as the **seeker's latent interest/intent** in the
dialogue. The released data therefore contains `8_belief_rec_2_com.json` but no
seeker-belief file, and `7_desire_seeker_com.json` but no recommender-desire
file.

## Metrics

Each leaf reports `acc`: `1.0` iff the sorted predicted letter set exactly
equals the sorted gold letter set, else `0.0`.

The vanilla extractor vendors the released scripts' non-CoT behavior: strip all
non-letters, dedupe characters, and accept only letters in the valid option
range. Direct tasks use `repeats: 5` plus a `take_first_k` filter named
`first_valid`; `process_results` scores the first repeated response whose parsed
letter set contains no out-of-range letters. An empty parsed direct response is
valid but wrong, matching the source loop's `all(...)` gate.

The CoT extractor vendors the released regex for final answers after `answer
is`, `answer:`, `the answer is`, or a final-answer lead followed by
`\boxed{X}`. CoT no-match cases are scored wrong on the first response, matching
the source scripts' sentinel path rather than retrying.

## Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf \
  --model_args pretrained=<model> \
  --tasks rectom \
  --output_path outputs/rectom --log_samples

PYTHONIOENCODING=utf-8 lm-eval run --model hf \
  --model_args pretrained=<model> \
  --tasks rectom_cot \
  --output_path outputs/rectom_cot --log_samples
```

If your lm-eval entry point is not using this vendored checkout's task tree,
also pass `--include_path lm-evaluation-harness/lm_eval/tasks`.

For chat-tuned models, compare runs consistently with or without
`--apply_chat_template`. Do not also pass `--system_instruction`; the official
system instruction is already baked into `doc_to_text`.

Smoke-test a small slice before a full run:

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model dummy \
  --tasks rectom_fine_intent_rec,rectom_coarse_intent_seeker,rectom_desire_seeker,rectom_cot_fine_intent_rec,rectom_cot_coarse_intent_seeker,rectom_cot_desire_seeker \
  --limit 2 --batch_size 1 --log_samples --output_path outputs/rectom_smoke
```

`--limit` is for plumbing only; do not report limited-run numbers. Full RecToM
runs should report the 10 leaf task rows. The original code itself does not
emit a benchmark-wide aggregate.

## Faithfulness deviations

1. **System message baked into the prompt.** The official scripts use a chat
   system message for most models and concatenate system+user for `o1`/`gemma`.
   lm-eval task YAML has no per-task system-role field, and RecToM's direct and
   CoT system prompts differ, so the adapter bakes the system instruction into
   the plain prompt. This matches the released fallback path and keeps the task
   self-contained.
2. **Bounded direct retry.** The official direct scripts retry until a valid
   parsed candidate appears. lm-eval cannot perform an unbounded conditional
   generate loop, so this adapter requests 5 sampled repeats and scores the
   first source-valid candidate. If all 5 direct responses are invalid, the item
   is scored wrong. Exact unbounded retry would require a separate offline/API
   runner.
3. **Released script toggles corrected to the paper/data.** The four Python
   scripts are comment-toggled templates, and the committed defaults are not a
   valid configuration for every data file (for example, seeker fine intent has
   A-P options while the default validation range is narrower). The adapter
   derives the valid letter range from each data file and the paper's Table 1.
4. **No benchmark-wide aggregate.** Earlier adapter drafts exposed six pooled
   family leaves and a convenience macro-mean. The source scripts and paper
   report the 10 source columns, so the canonical groups now expose those leaves
   only.
5. **Harness stop strings.** The official API calls set no explicit stop
   sequence. The adapter includes only chat end-token stops (`</s>`,
   `<|im_end|>`) and intentionally does not stop on newlines, because CoT
   answers may span multiple lines before the final answer.

## Provenance

- Dataset counts and option cardinalities are from `benchmarks/RecToM/data/*.json`.
- Prompt frames, extraction, temperature `0.1`, retry behavior, and
  `max_tokens=700` are from `benchmarks/RecToM/evaluate/*_ds_RecommenderToM.py`.
- The 10 paper columns and direct/CoT experiment variants are verified against
  the arXiv paper, especially Table 1 and Table 4.

## Verification

- Model-free loader/metric checks passed for the 10 source leaves: counts, valid
  gold letters, direct retry selection, empty-output handling, all-invalid
  direct repeats, and CoT no-match behavior.
- `conda run -n eval_env lm-eval validate --tasks rectom,rectom_cot
  --include_path lm-evaluation-harness/lm_eval/tasks/rectom` found both groups
  and validated them.
- `conda run -n eval_env lm-eval run --model dummy --tasks
  rectom_fine_intent_rec,rectom_coarse_intent_seeker,rectom_desire_seeker,rectom_cot_fine_intent_rec,rectom_cot_coarse_intent_seeker,rectom_cot_desire_seeker
  --limit 2 --batch_size 1 --log_samples --output_path outputs/rectom_smoke`
  built contexts, ran `generate_until`, produced `acc` for all six smoke leaves,
  and logged 5 direct repeats under the `first_valid` filter while keeping CoT
  single-shot under the default `none` filter.
