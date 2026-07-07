# OmniToM (lm-eval adapter)

Faithful lm-eval port of **OmniToM** (Bawatneh et al., 2026; arXiv:2605.26322) — a
two-stage, text-only Theory-of-Mind benchmark that makes the hidden belief structure of
a story *explicit and measurable* instead of probing endpoint QA. Stories are derived
from ToMBench (ACL 2024); **895 stories, 22,343 labeled belief propositions, 7 story
categories**.

OmniToM evaluates two linked generative tasks under **zero-shot TELeR Level-3** prompts:

- **Stage 1 — Belief Extraction:** given a story, emit all `(Actor, Belief, Order)`
  tuples as a pipe table. Scored by a **live GPT-5 semantic judge** → precision / recall /
  F1 over matched propositions, **macro-averaged over stories**.
- **Stage 2 — Belief Labeling:** given a story **and the benchmark belief table**, assign
  a 7-dimensional schema label vector to each belief. Scored by **per-dimension
  exact-match accuracy**, macro-averaged over stories.

The adapter **vendors** OmniToM's own prompt text and its *pure, deterministic* Stage-2
scorer from the read-only submodule `benchmarks/omnitom-benchmark/` (never imported into
the task, never edited). Data is read live from
`benchmarks/omnitom-benchmark/benchmark_story_belief_labels.jsonl`.

## Why Stage 1 is offline and Stage 2 is in-harness

Stage 1's scorer is a **generative LLM-as-judge (GPT-5)** in the loop — it cannot run
inside lm-eval. Stage 2's scorer is **pure and deterministic** (per-dimension exact
match), so it runs fully in-harness. Hence the split (approved design):

| | in-harness scoring? | how |
|---|---|---|
| **Stage 2** `omnitom_label` (+ 7 leaves) | ✅ yes | vendored scorer, byte-parity-diffed vs `compute_stage2_metrics` over all 895 stories |
| **Stage 1** `omnitom_extract` | ⛔ offline | `generate_until --predict_only --log_samples`, then `score_stage1_offline.py` feeds the generations through the repo's **own** GPT-5 judge |

## Tasks

**Stage 2 (labeling) — fully in-harness, faithful:**

| task | scope | metrics |
|---|---|---|
| `omnitom_label` | all 895 stories (**the paper headline**) | `overall_acc` + `acc_{order,truth_status,knowledge_access,representation,content_type,mental_source,context}` |
| `omnitom_label_ast` | Ambiguous Story Task (98) | same 8 |
| `omnitom_label_fbt` | False Belief Task (97) | same 8 |
| `omnitom_label_fpt` | Faux-pas Recognition Test (142) | same 8 |
| `omnitom_label_ht`  | Hinting Task Test (100) | same 8 |
| `omnitom_label_pst` | Persuasion Story Task (97) | same 8 |
| `omnitom_label_sit` | Scalar Implicature Test (154) | same 8 |
| `omnitom_label_sst` | Strange Story Task (207) | same 8 |

The 7 by-category leaves share the tag **`omnitom_label_by_category`** (run all with
`--tasks omnitom_label_by_category`; each is also runnable on its own). Doc = **one story**
(batched container: the whole story + belief table in one prompt, all label rows back).
Each metric is a per-story scalar meaned over stories — this equals the repo's
`safe_mean`-over-stories (macro-average over stories; no story is empty, so lm-eval's
`mean` == `safe_mean`). **`overall_acc` on `omnitom_label` is the paper's Stage-2 overall
accuracy** (macro over all 895 stories); the `acc_<dim>` are the per-dimension columns.

> There is deliberately **no group with a top-line aggregate** over the 7 leaves: an
> unweighted macro-over-categories ≠ the story-weighted overall (categories are unequal:
> 98/97/142/100/97/154/207). The faithful overall comes from `omnitom_label`; the leaves
> give the faithful by-category breakdown.

**Stage 1 (extraction) — generate-only, offline-scored:**

| task | scope | in-harness metrics |
|---|---|---|
| `omnitom_extract` | all 895 stories | `extract_exact_f1`, `extract_exact_precision`, `extract_exact_recall`, `pred_row_count` — **DIAGNOSTIC proxies only** |

The in-harness metrics are a strict **exact-string** P/R/F1 lower bound (the mock
backend's matcher), **not** the paper's Stage-1 F1. The faithful number comes from the
GPT-5 judge via `score_stage1_offline.py`.

## The system prompt is a REAL system role (faithfulness)

OmniToM's runner builds `[{system}, {user}]` and, for every model the paper evaluated
(all chat models), calls `apply_chat_template` → a **genuine system role**
(`run_replication.py:716`). To reproduce that context exactly, `doc_to_text` emits the
**user prompt only** and the system prompt is delivered as a real system turn via a
shipped run config:

```bash
# Stage 2 (labeling) — the headline in-harness run:
PYTHONIOENCODING=utf-8 lm-eval --model hf \
  --model_args pretrained=<chat-model> \
  --config lm_eval/tasks/omnitom/eval_config_label.yaml \
  --tasks omnitom_label --batch_size auto \
  --output_path out/omnitom_label --log_samples

# single category:
lm-eval ... --config .../eval_config_label.yaml --tasks omnitom_label_sst --limit 5 --log_samples
# all 7 categories:
lm-eval ... --config .../eval_config_label.yaml --tasks omnitom_label_by_category ...
```

`eval_config_{label,extract}.yaml` set `apply_chat_template: true` +
`system_instruction: <the stage's system prompt>` (both **byte-verified identical** to
`prompts_{label,extract}.PROMPT`). Equivalent to passing `--apply_chat_template
--system_instruction "<…>"` on the CLI.

> ⚠️ **Do not omit the config/flags.** Without `--system_instruction` the system prompt is
> dropped; without `--apply_chat_template` the run targets a base-model context the paper
> never used. The two stages have different system prompts, so use the matching config.

## Stage 1: run generate-only, then score offline

```bash
# 1) generate extractions in-harness (no scoring):
PYTHONIOENCODING=utf-8 lm-eval --model hf --model_args pretrained=<chat-model> \
  --config lm_eval/tasks/omnitom/eval_config_extract.yaml \
  --tasks omnitom_extract --predict_only --log_samples \
  --output_path out/omnitom_extract

# 2) score with the benchmark's OWN GPT-5 judge (needs OPENAI_API_KEY):
OPENAI_API_KEY=... python lm_eval/tasks/omnitom/score_stage1_offline.py \
  --samples out/omnitom_extract \
  --output-dir runs/omnitom_stage1
# -> runs/omnitom_stage1/metrics/stage1_overall.csv  (+ stage1_by_category.csv)
```

`score_stage1_offline.py` parses each generation with the benchmark's **own**
`parse_extraction_rows`, writes the `extraction/csv/` layout the repo expects, then
subprocesses `run_replication.py --stages judge metrics` — so the canonical GPT-5 judge
(3 few-shots) and the macro-over-stories P/R/F1 are preserved byte-for-byte. Use
`--dry-run` to only materialize the extraction CSVs (no API calls). `benchmarks/` is never
modified; `run_replication` is imported only for its pure parser/writer/path helpers.

## Decoding

Greedy, matching `run_replication.py` defaults: `do_sample: false`, `temperature: 0.0`,
`max_gen_toks: 6000`. The original sets **no stop strings** (generation runs to the token
cap / EOS); the templates add only the inert end-of-turn markers `</s>` / `<|im_end|>`,
which cannot appear inside a valid pipe table.

- **Capacity.** Belief tables are large (Ambiguous stories average ~47 beliefs × 9
  columns). Pick a smoke model whose context **exceeds** the largest prompt, and keep
  `max_gen_toks` high — under-sizing truncates the label table and the tail rows score
  wrong with no error.
- **Reasoning models.** A thinking model (e.g. Qwen3 with `<think>` blocks) can exhaust
  the budget before emitting the table → parsed rows drop and scores collapse. Raise
  `max_gen_toks` and/or disable thinking. (See memory: reasoning-model truncation.)
- `--limit` is for plumbing only — it does not distort these metrics (each story is
  scored independently), but report numbers from a full run.

## Faithfulness deviations (and why)

1. **Stage 1 scored offline** (unavoidable): its judge is a live GPT-5 model, which
   lm-eval cannot host. Reproduced faithfully via the repo's own judge+metrics
   (`score_stage1_offline.py`); the in-harness `extract_exact_*` metrics are labeled
   diagnostic proxies, **not** the paper metric.
2. **System prompt via run flag** (`--system_instruction` + `--apply_chat_template`):
   reproduces the original's real system role for chat models byte-for-byte. The
   original's *base-model* path (`system + "\n\n" + user` concatenation) is **not**
   reproduced — the paper evaluated no base models. `doc_to_text` intentionally carries
   only the user prompt.
3. **By-category as 7 independent leaves; overall from the unified task.** No group
   top-line aggregate is emitted (an unweighted macro-over-categories would not equal the
   paper's story-weighted overall).
4. **Stage-2 row matching is exact-text on `(actor, belief, occurrence)`** — brittle to
   paraphrase. This is the **original design** (`pair_indexed_rows` /
   `compute_stage2_metrics`): the model is *given* the belief table and asked to echo
   `Actor | Belief` and append labels, so a faithful model matches verbatim.
5. **Greedy decoding, `max_gen_toks: 6000`, no content stop strings** — matches
   `run_replication.py`'s defaults.

## Verification

- **Prompt parity — byte-identical.** All 895 Stage-1 **and** Stage-2 *user* prompts equal
  `build_extract_messages` / `build_label_messages` output exactly; `SYSTEM_EXTRACT` /
  `SYSTEM_LABEL` equal `prompts_{extract,label}.PROMPT` exactly (and the configs round-trip
  them byte-for-byte).
- **Stage-2 scorer parity — byte-identical (decisive).** Over all 895 stories,
  `process_results_label` reproduces the repo's `parse_label_rows` → `pair_indexed_rows` →
  per-dimension exact-match → `safe_mean` pipeline exactly (echo-gold → 1.0 on every
  dimension; perturbed predictions match `compute_stage2_metrics`'s replicated scorer to
  < 1e-12).
- **Loader sanity.** 895 stories (ids 1–895); per-category 98/97/142/100/97/154/207;
  22,343 beliefs.
- **Harness acceptance (Gate B).** All 10 tasks register; the 8 Stage-2 metrics compute;
  the `--config` system-role path works end-to-end.

Run env: `eval_env` (Python 3.13); `PYTHONIOENCODING=utf-8` on Windows.

## Provenance (all from the read-only submodule)

- Stage-1 / Stage-2 prompts ← `prompts_extract.py` / `prompts_label.py` (verbatim).
- Injected belief table ← `benchmark_prompting.py` (`actor_belief_rows`,
  `belief_table_pipe`).
- Gold recovery, table parsing, canonicalization, Stage-2 metric ←
  `run_replication.py` (`ground_truth_*_rows`, `parse_*_rows`, `canonicalize_label_*`,
  `pair_indexed_rows`, `compute_stage2_metrics`).
- Stage-1 judge + metrics (offline) ← `prompt_evaluate.py` + `run_replication.py`
  (`build_evaluation_messages`, `run_judge`, `compute_stage1_metrics`), invoked unmodified.
