# DialToM — lm-eval adapter

Faithful adaptation of DialToM (Yadav, Achananuparp, Jiang & Lim, SMU; arXiv:2604.20443,
"DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories")
for `lm-evaluation-harness`. Prompts are transcribed verbatim from the f-strings in the
read-only submodule `benchmarks/DialToM/benchmark.py` (never imported, never edited).

## v1 scope

The paper's own "core" is two MCQ tasks (§3.3: *"the DialToM benchmark consists of two
distinct multiple-choice QA tasks"*) — **Retrospective** (Literal ToM) and **Prospective**
(Functional ToM, context withheld). This adapter builds those two, plus **Written**
inference (§2.2/App. G, "Supplementary Semantic Verification").

**Deferred** (each is a distinct pipeline, not a static reproducible dataset — a scoped
decision, not an oversight):

| Deferred | Why |
|---|---|
| Prospective-Easy (Table 6) | §6.1 distractor-complexity *ablation* validating the Hard set isn't a lexical-shortcut artifact, not a headline capability number. Frozen data exists (`*_prospective-easy_verified.json`) if built later. |
| Counterfactual-NOTA (Table 7) | §6.2 robustness-vs-memorization ablation; needs a live GPT-4o counterfactual-generation pass (`counterfactual_test.py`), not a frozen task. |
| Teacher-Student injection (Table 8) | §6.3; replays one frozen historical Gemini-3-Pro reasoning trace — not reproducible as a general task. |
| Dialogue Context-Degradation (App. F) | A separate 100-item sample with Gemini-generated summaries at 3 context levels. |
| LLM-as-a-Judge (Table 14, App. G) | Needs a live judge-model call; the paper itself calls these scores *"qualitative explorations rather than primary empirical claims."* |

## Tasks

| Task | Domains | output_type | Metric(s) |
|---|---|---|---|
| `dialtom_retrospective_{mi,esc,pfg}` (+ group `dialtom_retrospective`) | MI/ESC/PFG | `generate_until` | `acc` |
| `dialtom_prospective_{mi,esc,pfg}` (+ group `dialtom_prospective`) | MI/ESC/PFG | `generate_until` | `acc` |
| `dialtom_written` | all 3 (single flat task; paper reports one overall number, not per-domain) | `generate_until` | `bleu`, `rouge_l`, `bertscore_f1` |

Top-level group `dialtom` bundles all three (`dialtom_retrospective`, `dialtom_prospective`,
`dialtom_written`) for a single `--tasks dialtom` invocation. It has no
`aggregate_metric_list` — the three children don't share a metric (`acc` vs.
`bleu`/`rouge_l`/`bertscore_f1`), so there's nothing meaningful to average across them;
each child still reports its own metrics in the results table.

## Architecture

- **doc = one item** (per-item protocol; no batching). Retrospective/Prospective each
  explode to one doc per verified item (already the file's own unit — MI 306/ESC 354/
  PFG 339 retro, MI 136/ESC 203/PFG 184 prospective, matching the paper's Table 11
  `# Retained` exactly). Written explodes `written_inference.csv` (900 rows = 300 items
  × 3 human references) grouped by `(dataset, mental_state, sub_id)` into 300 docs, each
  carrying a `refs: [3 strings]` field, joined against `combined_written_data.json` for
  the source dialogue.
- **Gate A (`generate_until`, not `multiple_choice`).** The paper's formalism (§3.3,
  argmax over `M(s'|H_i)`) reads as loglikelihood ranking; the actual run code
  (`benchmark.py`) uses provider JSON-schema-constrained generation + exact-match, and
  explicitly permits reasoning-model "thinking" traces ("Internal reasoning traces are
  permitted for all reasoning models, e.g., Gemini 3 Pro and GPT-5"). Built as
  `generate_until` + a shared letter EXTRACTOR (the hitom pattern) since that's what
  actually produced the paper's headline numbers.
- **Retrospective options are not re-shuffled.** The verified JSON already stores a
  fixed, valid A–D option order + `correct_option` letter; `benchmark.py`'s runtime
  `np.random.shuffle` on top of that is an extra re-randomization that only changes the
  label, not the content, and reproducing its exact sequentially-consumed RNG state buys
  nothing. (Same "already eval-ready; don't re-shuffle" call as the `dyntom` adapter.)
- **Prospective options get a deterministic shuffle**, seeded on a per-**row**-unique key
  (`random.Random(f"{domain}|{id}|{state}|{row_index}")`) — there is no frozen order to
  reuse here (`distractors` ships as an unordered list). The row index is load-bearing:
  the dialogue `id` is *not* unique (e.g. ESC has 203 rows but only 53 ids, and same-`id`
  rows share the same `correct_action`), so an `id`-only seed would give every row in a
  group the same permutation → the same gold slot → a gold-letter distribution skewed
  away from uniform (measured ESC A:47/B:19/C:66/D:71 under the old `id`-only seed vs.
  A:57/B:41/C:54/D:51 after the fix — the original's independent per-item `np.random`
  shuffle is ~uniform). Reproducible across our own re-runs; not letter-identical to any
  one paper run (nothing would be, since a fresh shuffle happens every invocation of the
  original script).
- **Written scores the FULL generated line against the full 3 references** (steer prefix
  included on both sides, e.g. "I feel stressed about..."), matching what the refs
  actually contain. `gen_prefix` was deliberately **not** used — it would only capture
  the *completion after* the prefix, corrupting scoring against refs that include it.
  The model is told to start its own line with the steer text via the same in-prompt
  instruction the original uses.

## Metrics

- **`acc`** — exact match of the extracted letter vs. `gold_letter`.
- **`bleu` / `rouge_l` / `bertscore_f1`** — reconstructed multi-reference driver (the
  repo ships no such driver; App. G: *"we report the average similarity scores across
  all references to handle multiple gold references"*). Per doc: score against each of
  the 3 refs independently (sacrebleu `BLEU.sentence_score(...).score`; `rouge`'s
  `rouge-l` F1; `bert_score.BERTScorer(lang="en", rescale_with_baseline=True)` F1), then
  **mean over the 3**. `bertscore_f1` batches the 3 pairs in one `scorer.score()` call.
  BERTScore is vendored **in-harness** (same pattern as the `fantom` adapter's RoBERTa
  cosine scorer) rather than offloaded — deterministic, no live judge call, just a heavy
  (~1.4GB) embedder load on first use.

## Running

Run inside the harness env (Python 3.13 — **use `conda activate eval_env` via
`source .../etc/profile.d/conda.sh`, not `source activate`, which silently no-ops and
leaves you on base's Python 3.12** in this repo's shell setup). On Windows set
`PYTHONIOENCODING=utf-8`.

```bash
lm-eval run --model hf --model_args pretrained=<model> \
    --tasks dialtom_retrospective,dialtom_prospective,dialtom_written \
    --include_path lm-evaluation-harness/lm_eval/tasks \
    --apply_chat_template --log_samples --output_path out/
```

The first `dialtom_written` run downloads/caches the RoBERTa embedder (~1.4GB) plus
`rouge`/`bert-score`/`sacrebleu` must be installed in the run env (`pip install rouge
bert-score`; `sacrebleu` ships with the harness).

For instruction-tuned/chat models add `--apply_chat_template` (the original queries
chat/reasoning APIs; base models won't follow the letter-only instruction well).

> **Generation budget is reasoning-model-sized on purpose.** All 3 templates use
> `max_gen_toks: 1024` and drop `"\n\n"` from `until`. A tight budget (or an early
> paragraph-break stop) truncates a `<think>...</think>` preamble before any answer
> appears — found live while smoke-testing Qwen3-1.7B, whose default thinking mode alone
> exceeded a 16-token budget and scored 0/5 on both MCQ tasks with no error, just silent
> truncation. The EXTRACTOR and written scorer both strip everything through the last
> `</think>` before parsing, so a model that finishes its reasoning within budget scores
> correctly; a model that runs out of budget mid-thought legitimately has no answer to
> extract (scores as wrong/low, same as it would if the original's API call were cut off).
> If you see collapsed accuracy on a heavy reasoning model, raise `max_gen_toks` further
> via `--gen_kwargs max_gen_toks=<N>` before assuming a task bug.

## Deviations from the paper

1. **`generate_until`, not `multiple_choice`.** See Architecture — Gate A decided in
   favor of matching the paper's actual run protocol over its formalism. A
   `multiple_choice` variant would be a cheap, mechanical follow-up (loglikelihood over
   the same 4 option texts, no extractor needed) if pure ranking fidelity is wanted.
2. **Constrained JSON-schema decoding is not reproduced.** The paper elicits answers via
   provider structured output (`pydantic` enum schema); lm-eval has no such constraint
   mechanism, so a free-text generation + regex extractor is used instead. Functionally
   equivalent for well-behaved instruction-tuned models; a base or poorly-instructed
   model may fail to emit a bare letter where the original's schema would have forced one.
3. **Decoding params unspecified by the paper → greedy default.** §4.1 says only
   "zero-shot prompting"; no temperature is given (constrained JSON output from a
   provider API isn't directly comparable to open-weights sampling anyway). Ran
   `do_sample: false, temperature: 0.0` as the faithful default for a
   forced-single-answer task; document if you deviate.
4. **Retrospective/Prospective option shuffling deviates from the paper's exact runtime
   RNG**, as detailed in Architecture. Faithful in effect (a fixed, non-degenerate,
   position-unbiased A–D layout with one correct answer) but not letter-identical to any
   specific paper run.
5. **Prospective's hardcoded "client"** in *"internal Mental State profile ... of client
   during the conversation"* is transcribed verbatim from `benchmark.py`, including for
   ESC/PFG where the recipient role is actually Seeker/Persuadee (the *next* sentence,
   "Mental state of {role}", does use the correct role name). This looks like a bug in
   the original prompt-assembly code; not fixed here per "transcribe verbatim, never
   invent/correct prompt wording."
6. **Written scoring is a from-scratch multi-reference driver**, not lifted code — the
   repo's `written_metrics.py` has no driver wiring generation → refs → aggregation at
   all (single hyp/ref helper functions only). Built per App. G's one-sentence spec
   ("average across all references"); `get_bleu`'s original `._verbose` return value
   (a string) was not vendored — use `.score` (the numeric BLEU).
7. **`dialtom_written` is one flat task across all 3 domains**, not 3 domain leaves —
   Table 13 reports a single overall number per model, no per-domain breakdown, unlike
   Retrospective/Prospective's Table 3.
8. **Attribute-level breakdown (Table 5) is not a separate lm-eval metric.** Each
   Retrospective doc already carries its own `state` field (Belief/Desires/.../Trust) in
   the logged sample; reproduce Table 5 by grouping `--log_samples` output on that field
   rather than declaring 6 additional per-state metrics (lm-eval's per-doc `mean`
   aggregation has no clean way to average a metric over only a state-matching subset of
   docs within one task without either NaN-skipping machinery or per-state leaf tasks —
   not worth the complexity for a metric fully recoverable from the existing log).
9. **Pre-existing data quirks are reproduced verbatim, not cleaned up** — e.g. one MI
   item's `topic` field is literally `"on on reducing recidivism"` (a duplication baked
   into the source data), which renders as "...on the topic of on on reducing
   recidivism." Confirmed present in the raw JSON, not introduced by this adapter.

## Verification performed

- **Registration:** `lm-eval validate` — all 9 tasks (6 leaves + 2 groups + `dialtom_written`) register cleanly.
- **Loader sanity:** item counts match Table 11's `# Retained` exactly for every domain × task (retro 306/354/339, prospective 136/203/184) and Written's 300; every doc has a non-empty gold letter / 3 refs.
- **Prompt fidelity:** rendered `doc_to_text` diffed by eye against the literal `benchmark.py` f-strings for all 3 tasks; no gold leakage into the prompt.
- **Metric plumbing:** MCQ extractor handles clean, punctuated, "answer is X", parenthetical, and prose-embedded letters; written scorer is monotonic (near-perfect hyp > garbage > empty-hyp guard) for all 3 metrics, with metric-name keys matching `metric_list` exactly.
- **Smoke test** (Qwen3-1.7B, `--apply_chat_template`, `--limit 3`, all 3 tasks): end-to-end OK. `acc` = 0.67 retrospective / 0.33 prospective (directionally consistent with the paper's reasoning-asymmetry finding, though n=3 is not statistically meaningful); written `bleu=6.89, rouge_l=0.33, bertscore_f1=0.47` — same order of magnitude as Table 13's comparable GPT-4o row (`9.44 / 0.31 / 0.44`). Caught and fixed the `<think>`-truncation bug described above during this pass.
