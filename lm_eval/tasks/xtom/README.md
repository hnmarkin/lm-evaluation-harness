# XToM

This adapter reconstructs the multilingual generative evaluation reported for
Qwen-2.5-7B-Instruct in XToM Tables 8--10. The released XToM repository ships
the translated data archive but no evaluation code, so the paper's prompt
tables and metric definitions are the executable specification.

## Tasks

Each leaf evaluates all five paper languages (`en`, `zh`, `de`, `fr`, `ja`).
This keeps the public surface to six independently runnable tasks:

| Task | Paper table | Documents per language | Reported metrics |
|---|---:|---:|---|
| `xtom_xfantom` | 8 | 618 | belief choice (overall/first/second/acyclic/cyclic), fact choice |
| `xtom_xfantom_cot` | 8 | 618 | same, with CoT cue |
| `xtom_xtomi` | 9 | 647 | first-order, second-order, weighted belief average, reality |
| `xtom_xtomi_cot` | 9 | 647 | same, with CoT cue |
| `xtom_xnegtom` | 10 | 1,760 | belief exact match, desire exact match, intention micro/macro F1 |
| `xtom_xnegtom_cot` | 10 | 1,760 | same, with CoT cue |

`xtom` is the top-level group containing all six leaves. Metrics are emitted as
percentages and prefixed by language, for example `zh_belief_acc`. Every task
also emits a non-paper `LANG_invalid_rate` diagnostic. An invalid generation is
wrong in the paper metric and remains in its denominator; `invalid_rate` does
not alter accuracy or F1.

The loaders read `benchmarks/XToM/XToM_DataSet.zip` in place. They neither
extract nor modify anything under `benchmarks/`.

## Published protocol represented here

- Prompt text is transcribed from paper Tables 12--18. CoT appends the literal
  English cue `let's think step by step.` to every language, as described by
  the paper.
- Qwen decoding uses sampling with temperature `0.5` and top-p `0.7`.
  Direct leaves allow 256 generated tokens and CoT leaves allow 1,024.
- XFANToM uses `short_context`, one fact question per dialogue, and only
  inaccessible belief questions. The breakdown is 300 fact + 217 first-order
  + 51 acyclic second-order + 50 cyclic second-order.
- XToMi includes 123 first-order `_tom`, 224 second-order `_tom`, and 300
  reality questions. Memory and `_no_tom` controls are excluded. `belief_acc`
  is the document-weighted average over the 347 belief questions.
- XNegotiationToM has 586 belief prompts, 586 desire prompts, and all 588
  released intention prompts per language. Belief/desire require the complete
  ordered three-letter answer. Intention is a multi-label A--I prediction.

The conservative answer parser checks the final answer tail first, then an
explicit answer letter, then an unambiguous option value for single-answer
tasks. Negotiation belief/desire accept exactly three A--D letters; intention
accepts a set of A--I letters. Reasoning text is permitted, but ambiguous or
unparseable output is invalid rather than guessed.

## Run Qwen-2.5-7B-Instruct

Use one SLURM job per leaf so the six tasks can run concurrently:

```bash
sbatch slurm/run_xtom_qwen25.sbatch xtom_xfantom
sbatch slurm/run_xtom_qwen25.sbatch xtom_xfantom_cot
sbatch slurm/run_xtom_qwen25.sbatch xtom_xtomi
sbatch slurm/run_xtom_qwen25.sbatch xtom_xtomi_cot
sbatch slurm/run_xtom_qwen25.sbatch xtom_xnegtom
sbatch slurm/run_xtom_qwen25.sbatch xtom_xnegtom_cot
```

For a small plumbing run, add a per-task limit:

```bash
sbatch --export=ALL,LM_EVAL_LIMIT=25 slurm/run_xtom_qwen25.sbatch xtom_xfantom
```

Do not report metrics from a limited run: documents are interleaved by
language for useful smoke tests, but a small limit may omit a metric family and
produce `NaN`. The full six-leaf run makes 30,250 generations. The direct and
CoT tasks use batch sizes 16 and 8 respectively by default; override with
`--export=ALL,BATCH_SIZE=N` after a hardware-specific smoke test.

Completed results land under `slurm/outputs/qwen2.5-7b/<task>/`. Compare the
newest results for all six tasks with the exact paper rows using:

```bash
python lm-evaluation-harness/lm_eval/tasks/xtom/compare_paper.py \
  slurm/outputs/qwen2.5-7b --strict \
  --output outputs/xtom-qwen25-paper-comparison.md
```

The report shows adapter minus paper deltas in percentage points. Sampling is
nondeterministic unless the model/backend seed behavior is also held fixed, so
small rerun differences are expected.

## Reconstructed details and released-data repairs

These are the known places where literal parity with unavailable evaluator code
cannot be established:

1. **Output extraction.** The paper specifies requested answer forms but does
   not publish the parser. The conservative parser above is a reconstruction;
   `invalid_rate` makes its impact visible.
2. **XFANToM option ordering.** The XToM data contains correct/wrong answer
   fields but no displayed order. This adapter reuses FANToM's deterministic
   seed-99 binary shuffle independently for each language. This affects letter
   placement, not option-value scoring or denominators.
3. **Generation length and stop strings.** The paper reports temperature and
   top-p but not an exact generation cap or evaluator stop list. The adapter
   uses 256/1,024 tokens and Qwen EOS markers. Typography and line wrapping
   visible in the PDF are normalized in source strings.
4. **Chinese XFANToM duplicate.** The archive has 301 Chinese rows but only 300
   unique `set_id` values; the second `24-0-0` row is removed deterministically.
5. **XToMi translated gold drift.** Three French and two Japanese gold strings
   are absent from their localized choice pair. All other aligned rows preserve
   the English gold position exactly, so those five use the aligned English
   option index while displaying and scoring the localized choice.
6. **Chinese negotiation speaker typo.** Dialogue `61-8` intention slot 1
   stores `人物1`, but the final utterance and aligned English row identify the
   speaker as person 2. The prompt uses the actual utterance speaker.
7. **Negotiation intention count.** Table 6 reports 586 intention examples per
   language, while the released archive contains 588 with no published rule for
   dropping two. The adapter keeps all 588 instead of inventing an exclusion,
   giving 1,760 rather than 1,758 documents per language. Consequently this
   leaf evaluates 8,800 documents, ten more than the paper's stated total.

The paper also studies cross-lingual consistency outside the target Tables
8--10. It is intentionally not an extra offline component of this six-task
adapter.

## Verification

The implementation checks exact language/family counts while loading. Local
verification covers Python compilation, all-valid and all-invalid full-corpus
metric invariants, `lm-eval validate`, and a dummy-model generation/scoring
run. A real Qwen-2.5-7B smoke and the six full jobs remain cluster runs because
the model does not fit the development machine's available GPU memory.
