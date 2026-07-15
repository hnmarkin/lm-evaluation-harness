# UnifiedToM - lm-eval adapter

Faithful static adapter for the direct question-answering baseline in
`benchmarks/unifiedtombenchmark` / UniToMBench (arXiv:2506.09450).

## Scope

The paper evaluates models in a zero-shot multiple-choice setup both with and
without SimToM perspective-taking. This adapter implements the **without
SimToM** direct baseline only. The SimToM condition in the repo is a three-call
pipeline per item (character identification, perspective filtering, then QA),
so it is not a single lm-eval model forward pass.

## Tasks

`unifiedtom` is an adapter-derived, non-paper macro group over the 10 Table-1-style
direct leaves:

| task | source | n |
|---|---|---:|
| `unifiedtom_uot` | ToMBench `Unexpected Outcome Test` | 300 |
| `unifiedtom_sit` | ToMBench `Scalar Implicature Test` | 200 |
| `unifiedtom_pst` | ToMBench `Persuasion Story Task` | 100 |
| `unifiedtom_fbt` | ToMBench `False Belief Task` | 600 |
| `unifiedtom_ast` | ToMBench `Ambiguous Story Task` | 200 |
| `unifiedtom_ht` | ToMBench `Hinting Task Test` | 103 |
| `unifiedtom_sst` | ToMBench `Strange Story Task` | 407 |
| `unifiedtom_frt` | ToMBench `Faux-pas Recognition Test` | 560 |
| `unifiedtom_evolving_stories` | `evolving_stories_250.xlsx` | 250 |
| `unifiedtom_multi_interaction` | `multi_interaction_100.xlsx` | 97 |

Companion groups:

- `unifiedtom_tombench` - adapter-derived, non-paper macro mean over the 8
  ToMBench leaves.
- `unifiedtom_custom` - adapter-derived, non-paper macro mean over the 2
  README-canonical custom leaves.

The three macro aggregates are conveniences only. For faithful source/paper
comparisons, report the individual leaf `acc` rows rather than an aggregate.

## Architecture

- **Doc unit:** one MCQ.
- **Output type:** `generate_until`, not `multiple_choice`. The paper/code query
  chat models and score the generated option letter; they do not loglikelihood-rank
  fixed choices.
- **Prompt:** reconstructed from the repo scripts:
  - ToMBench rows: `Story: ... Question: ... Option A: ...`
  - Custom rows: `Story: ... Question: ... Options: {raw dict string}`
- **Decoding:** sampled with `temperature: 0.7`, matching the paper's experimental
  setup.
- **Metric:** per-leaf `acc`. The scorer is intentionally strict: after
  `strip()`, uppercasing, and removing periods, the whole response must be exactly
  the gold letter. It does not regex-extract a letter from verbose text.

## Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks unifiedtom --include_path lm-evaluation-harness/lm_eval/tasks \
  --batch_size auto --output_path outputs/unifiedtom --log_samples

# one leaf for plumbing:
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks unifiedtom_uot --include_path lm-evaluation-harness/lm_eval/tasks \
  --limit 5 --log_samples
```

Use `--apply_chat_template` for chat/instruction models when comparing to the
paper's chat-model setup. The generic source system message (`You are a helpful
assistant.`) is included through the task `description`, so base-model runs remain
self-contained.

## Faithfulness deviations

1. **Direct baseline only.** The SimToM pipeline is out-of-harness because it
   needs extra live model calls before the scored answer. A faithful SimToM run
   should be a separate offline/original-eval bridge, not a YAML task.
2. **Released custom data is smaller than the paper prose.** The paper says the
   custom dataset contains 1,025 scenarios (500 multi-interaction and 525 evolving
   story tasks). The repo README and runnable direct scripts name only
   `evolving_stories_250.xlsx` and `multi_interaction_100.xlsx`; the other shipped
   spreadsheets are overlapping/superseded snapshots and still do not reconstruct
   the stated 1,025 count. The adapter uses the README/script-canonical files.
3. **Two-choice ToMBench rows stay two-choice.** The paper describes a four-choice
   format, but the released ToMBench workbook has missing C/D options in Strange
   Story and Faux-pas rows. The loader renders only present options instead of
   inventing empty C/D choices.
4. **System role is flattened.** The source scripts send a generic system message
   and a user prompt. lm-eval task configs are model-agnostic, so the system text is
   supplied through `description` rather than as a chat role. For exact chat
   formatting, run with `--apply_chat_template`.
5. **Generation cap and stop set.** The source OpenAI calls do not set a token cap
   or stop strings. The adapter uses `max_gen_toks: 32` and EOS/chat-end stops so
   local harness runs terminate predictably without cutting on newlines.
6. **Repeated custom prompts follow source dictionary behavior.** The source custom
   evaluators use the formatted prompt as a dictionary key. The adapter therefore
   evaluates the 97 unique prompts in `multi_interaction_100.xlsx`, rather than all
   100 spreadsheet rows. It retains the first source-row identifier and applies the
   last duplicated row's gold answer, matching Python dictionary assignment;
   source-row metadata records the first and last rows for auditability.

## Retired experiment: paper-fragile variant (removed 2026-07-15)

A companion group `unifiedtom_fragile` (10 `*_fragile` leaves) previously lived in this
directory to test whether two adapter choices explain a gap between this adapter and
UniToMBench Table 1's reported baseline, measured on Llama-3-8B-Instruct:

1. **`nan` options restored.** Missing Strange-Story / Faux-pas C/D options rendered as
   the literal `Option C: nan, Option D: nan` (exactly as `tom.py`'s pandas f-string
   does), instead of being dropped (deviation #3 above). Every row became four-option.
2. **Strict scorer.** A `process_results_fragile` that compared `response.strip() ==
   target` with no uppercasing or dot-removal, reproducing the paper's exact-match
   sensitivity to casing, trailing punctuation, and verbose answers.

Everything else (prompt wording, system message, `temperature: 0.7`, `max_gen_toks: 32`)
was identical to the headline task, isolating the effect of just those two deviations.

**What it found:** reversing both deviations left scores essentially unchanged (macro
64.8% fragile vs. 64.9% base, vs. the paper's 56.9%) — full table in
`proofs/unifiedtom/unifiedtom_lmeval_vs_paper.md`. The two leaves most suspected of being
inflated by dropping empty options (`frt`, `sst`) barely moved (67.9→67.5, 77.4→75.7): a
competent instruct model ignores injected `nan` text either way.

**Verdict, corrected 2026-07-14** (`proofs/unifiedtom/2026-07-14-faithfulness-correction.md`):
the original writeup overstated this as "the gap is not an adapter-faithfulness problem."
A re-audit found the comparable scope is narrower than all 10 leaves — the paper's PST
cell uses only the 76 released rows tagged `Desire: Desires influence on actions`, not
all 100, and the paper's two custom corpora have 525/500 items of which only 250/97 were
ever released — and the paper's exact serving stack ("Llama 3 8B Instruct Turbo") is not
definitively identified, so the local BF16 checkpoint is a proxy, not "the paper's own
model." The supportable conclusion is only that NaN-rendering and scorer-normalization
are ruled out as causes of the *local-model* gaps on the leaves where paper/local scope
actually match (SIT, PST-76, HT) — not that the full paper-reproduction gap is explained.
Verdict: **inconclusive paper reproduction**, not "adapter deviations cleared."

The `unifiedtom_fragile` group/leaves and their `utils.py` functions
(`_format_tombench_prompt_fragile`, `process_results_fragile`) were deleted once this
conclusion was reached; the code remains recoverable from git history if the ablation
needs to be rerun (e.g. against a real paper-identified serving stack).

## Verification

Model-free checks performed during build:

- loader counts match the table above (2,817 docs total across the 10 leaves);
- every gold letter is reachable from the emitted choices;
- prompt samples match the source scripts' wording for direct ToMBench/custom rows;
- all-correct synthetic generations score `1.0`; all-wrong generations score `0.0`;
- `lm-eval validate --tasks unifiedtom,unifiedtom_tombench,unifiedtom_custom`
  passes;
- dummy-model `--limit 1` smokes pass for all three groups, each rendering its
  derived, non-paper macro alias while preserving the leaf aliases.

Real-model smoke should use a chat/instruction model with `--apply_chat_template`
for paper-comparable prompting. `--limit` is only a plumbing check.
