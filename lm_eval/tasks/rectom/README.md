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

Each group exposes 6 runnable task families covering the 10 paper columns:

| runnable task suffix | underlying paper columns | n | reported metrics |
|---|---|---:|---|
| `coarse_intent` | Coarse Intention (Rec), Coarse Intention (Seek) | 4410 | `acc`, `acc_rec`, `acc_seeker` |
| `belief` | Belief (Rec) | 1762 | `acc` |
| `desire` | Desire (Seek) | 1448 | `acc` |
| `fine_intent` | Fine Intention (Rec), Fine Intention (Seek) | 4410 | `acc`, `acc_rec`, `acc_seeker` |
| `judgement` | Judgement (Rec), Judgement (Seek) | 4247 | `acc`, `acc_rec`, `acc_seeker` |
| `prediction` | Prediction (Rec), Prediction (Seek) | 4247 | `acc`, `acc_rec`, `acc_seeker` |

The `rectom` and `rectom_cot` group rows include a convenience unweighted
macro-mean over the 6 runnable leaves. The paper reports per-column accuracies;
for the four dual-role leaves, those columns are preserved as `acc_rec` and
`acc_seeker`.

Belief and desire are intentionally one-sided in RecToM. The paper defines the
belief task as the **recommender's belief** about the seeker's attitude toward a
movie, and the desire task as the **seeker's latent interest/intent** in the
dialogue. The released data therefore contains `8_belief_rec_2_com.json` but no
seeker-belief file, and `7_desire_seeker_com.json` but no recommender-desire
file.

## Metrics

Each leaf reports:

- `acc`: `1.0` iff the sorted predicted letter set exactly equals the sorted
  gold letter set, else `0.0`.
- `acc_rec` / `acc_seeker`: role-specific exact set-match accuracies, reported
  on the four dual-role merged leaves only.

The vanilla extractor vendors the released scripts' non-CoT behavior: strip all
non-letters, dedupe characters, and accept only uppercase letters in the valid
option range. The CoT extractor vendors the released regex for final answers
after `answer is`, `answer:`, `the answer is`, or `\boxed{X}`.

## Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf \
  --model_args pretrained=<model> \
  --tasks rectom \
  --include_path lm-evaluation-harness/lm_eval/tasks \
  --output_path outputs/rectom --log_samples

PYTHONIOENCODING=utf-8 lm-eval run --model hf \
  --model_args pretrained=<model> \
  --tasks rectom_cot \
  --include_path lm-evaluation-harness/lm_eval/tasks \
  --output_path outputs/rectom_cot --log_samples
```

For chat-tuned models, compare runs consistently with or without
`--apply_chat_template`. Do not also pass `--system_instruction`; the official
system instruction is already baked into `doc_to_text`.

Smoke-test a small slice before a full run:

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model dummy \
  --tasks rectom_fine_intent,rectom_cot_fine_intent \
  --include_path lm-evaluation-harness/lm_eval/tasks/rectom \
  --limit 2 --batch_size 1
```

`--limit` is for plumbing only; do not report limited-run numbers. Full RecToM
runs should report the 6 family leaves and the role-specific metrics on
dual-role leaves. The group macro-mean is only a convenience roll-up.

## Faithfulness deviations

1. **System message baked into the prompt.** The official scripts use a chat
   system message for most models and concatenate system+user for `o1`/`gemma`.
   lm-eval task YAML has no per-task system-role field, and RecToM's direct and
   CoT system prompts differ, so the adapter bakes the system instruction into
   the plain prompt. This matches the released fallback path and keeps the task
   self-contained.
2. **No conditional retry loop.** The official scripts retry generation until
   every parsed character is in the valid letter set. lm-eval cannot perform a
   per-item generate-until-valid loop; invalid or over-verbose generations are
   scored wrong on the first sample.
3. **Released script toggles corrected to the paper/data.** The four Python
   scripts are comment-toggled templates, and the committed defaults are not a
   valid configuration for every data file (for example, seeker fine intent has
   A-P options while the default validation range is narrower). The adapter
   derives the valid letter range from each data file and the paper's Table 1.
4. **Six runnable leaves, ten paper columns.** The paper reports 10 columns, but
   four pairs differ only by Rec/Seeker role. To avoid excess task YAMLs, this
   adapter merges those pairs into one runnable family while preserving role
   splits as `acc_rec` and `acc_seeker`. `answer_coarse` inside
   `1_intent_rec.json` and `2_intent_seeker.json` is analysis metadata, not an
   extra leaf.
5. **Harness stop strings.** The official API calls set no explicit stop
   sequence. The adapter includes only chat end-token stops (`</s>`,
   `<|im_end|>`) and intentionally does not stop on newlines, because CoT
   answers may span multiple lines before the final answer.

## Provenance

- Dataset counts and option cardinalities are from `benchmarks/RecToM/data/*.json`.
- Prompt frames, extraction, temperature `0.1`, and `max_tokens=700` are from
  `benchmarks/RecToM/evaluate/*_ds_RecommenderToM.py`.
- The 10 paper columns, 6 runnable task families, and direct/CoT experiment
  variants are verified against the arXiv paper, especially Table 1 and Table 4.

## Verification

- Model-free loader/metric checks passed over all 20,524 docs per variant:
  counts, valid gold letters, prompt rendering, all-correct synthetic generations,
  and all-wrong synthetic generations.
- `conda run -n eval_env lm-eval validate --tasks rectom,rectom_cot
  --include_path lm-evaluation-harness/lm_eval/tasks/rectom` found both groups
  and validated them.
- `conda run -n eval_env lm-eval run --model dummy --tasks
  rectom_fine_intent,rectom_cot_fine_intent --include_path
  lm-evaluation-harness/lm_eval/tasks/rectom --limit 2 --batch_size 1`
  built contexts, ran `generate_until`, and produced `acc`, `acc_rec`, and
  `acc_seeker` for both merged dual-role leaves.
