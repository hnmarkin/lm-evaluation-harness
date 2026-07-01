# OpenToM (lm-eval adapter)

Faithful lm-eval port of **OpenToM** (Xu et al., ACL 2024; [arXiv:2402.06044](https://arxiv.org/abs/2402.06044)).
OpenToM presents an LLM-generated narrative about a *mover* relocating an entity (observed
or not by an *observer*) and asks first-/second-order Theory-of-Mind questions about the
physical world (location, fullness, accessibility) and the psychological world (attitude).
Free-text answers are mapped to fixed label sets and scored with **macro-averaged F1**
(the labels are imbalanced — the authors' recommended metric).

The adapter **vendors** OpenToM's own prompt assembly and answer-extraction logic from the
read-only submodule `benchmarks/OpenToM/` (never imported, never edited). Data is read from
`benchmarks/OpenToM/data/opentom.json` (13,708 questions, normal-length narratives).

## Tasks (group `opentom`)

Per-item protocol → leaf MATRIX (mirrors `hitom`/`dyntom`). The 7 leaves reproduce the
official `data/opentom_data/` split files exactly (verified counts in parentheses):

| task | genre / order | label space | n |
|---|---|---|---|
| `opentom_location_cg_fo` | coarse location, first-order  | binary Yes/No        | 1192 |
| `opentom_location_cg_so` | coarse location, second-order | binary Yes/No        | 1192 |
| `opentom_location_fg_fo` | fine location, first-order    | ternary orig/new/equal | 2384 |
| `opentom_location_fg_so` | fine location, second-order   | ternary orig/new/equal | 1192 |
| `opentom_multihop_fo`    | multihop, first-order         | ternary (fullness + accessibility) | 3576 |
| `opentom_multihop_so`    | multihop, second-order        | ternary (fullness + accessibility) | 3576 |
| `opentom_attitude`       | attitude                      | ternary pos/neu/neg  | 596 |

Coarse vs fine location is a **question-phrasing split, not a re-scoring of the same rows**:
`'locate'` questions are fine (answer is a place string), the rest (`'initial'`) are coarse
(Yes/No) — exactly the `'locate'`/`'initial'` branch in `build_prompt.py`. (The original's
`--lg` flag achieves the same split implicitly: it scores all location rows one way and the
mismatched subset falls out as `gold=None` and is dropped.)

## Metrics

Each task reports (custom `!function` aggregations consuming the whole `(gold,pred)` list —
the FANToM / `matthews_corrcoef` set-level pattern):

- **`macro_f1`**, **`acc`** — paper-faithful: corrupted predictions (the extractor cannot
  parse a valid label, `pred == -1`) are **dropped** before scoring, matching `evaluate.py`.
- **`macro_f1_strict`**, **`acc_strict`** — corrupted predictions counted as **wrong**.
- **`corrupt_rate`** — fraction of corrupted predictions (lower is better).
- `opentom_multihop_*` additionally report **`fullness_f1`** and **`accessibility_f1`** (the
  per-subgenre breakdown `evaluate.py` also prints; `macro_f1` is the combined "overall").

The group `opentom` adds an (unweighted) macro-mean across the 7 partitions as a convenience —
**not** a paper-reported number (OpenToM reports per-partition macro-F1).

## Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval --model hf --model_args pretrained=<model> \
  --tasks opentom --batch_size auto --output_path out/opentom --log_samples
# single partition:
lm-eval ... --tasks opentom_location_fg_fo --limit 5 --log_samples
```

## Faithfulness deviations (and why)

1. **By-key `plot_info` access (bug-fix vs the released evaluator).** The original
   `opentom_evaluator.py` / `get_info` positionally unpack `plot_info.values()` assuming key
   order `(mover, affected_char, eoi, original_place, move_to_place)`, but the **released**
   `opentom.json` uses `(mover, eoi, original_place, move_to_place, observer)` — there is no
   `affected_char` key (the `observer` plays that role). Copying the positional unpack would
   mis-bind `original_place`/`move_to_place` and silently corrupt **every fine-location gold**.
   We read places **by key** and use `observer` as the affected character. (In the vanilla
   prompt path this never affects the prompt: every `{coi}/{eoi}/{poi}/{second_order_statement}`
   replace in `build_prompt.py` is a no-op because the `.txt` templates contain only
   `{narrative}`/`{question}` — verified.)
2. **Full-pool scoring** (one set) instead of the paper's **5 batches × 50 sampled items →
   mean ± std**. The dossier recommends scoring the full pool (cleaner, deterministic). Macro-F1
   of the full pool differs slightly from the mean of per-batch macro-F1s.
3. **System message baked into the prompt.** The original prepends a system turn
   `"You are an expert in modeling other's mental state."` (`load_baseline_model.py`). lm-eval's
   `--system_instruction` is a run-level flag (only active with `--apply_chat_template`), so to
   keep the task self-contained and model-agnostic we prepend it as the first line of the user
   prompt instead of a dedicated system role.
4. **Greedy decoding** (`do_sample: false`, `temperature: 0.0`), vs the paper's GPT runs at
   `temperature=0` and open-model runs that stop at `\n`/`\n\n`. Stop strings: `</s>`,
   `<|im_end|>`, `\n\n`; `max_gen_toks: 128` (answers are "without any explanation").
   **Reasoning models:** the default budget/stop assume a *direct-answering* model (the paper's
   setup — GPT-3.5/4, Llama-chat, Mixtral). A thinking model (e.g. Qwen3 with `<think>` blocks)
   will exhaust 128 tokens inside its reasoning and emit no parseable label → high `corrupt_rate`.
   For such models raise `--gen_kwargs max_gen_toks=...` and/or disable thinking; the drop-policy
   `macro_f1`/`acc` then score only the parsed answers (`corrupt_rate` flags the rest).
5. **Vanilla prompting only.** CoT / SimToM / Self-Ask variants are deferred (documented
   follow-ups), as is **OpenToM-L** (`opentom_long.json`) and **mover/observer perspective**
   filtering (`--perspective`, default `all` reproduced).
6. **`macro_f1` matches `evaluate.py` exactly** (`f1_score(valid_gt, valid_pred, average='macro')`
   over corruption-filtered pairs). For multihop, `macro_f1` is the combined fullness+accessibility
   "overall" number (`evaluate.py`'s `f13`), which mixes the two ternary label semantics — faithful
   to the original. `corrupt_rate` is defined as corrupted-preds / partition-size (a clean
   interpretation; `evaluate.py` folds `gold=None` drops into its corruption denominator).

## Provenance

- Prompt templates ← `src/prompts/chatgpt_opentom_prompts/*.txt` (transcribed verbatim).
- Prompt question mods (vanilla path) ← `src/utils/build_prompt.py`.
- Answer extractors ← `src/evaluate/opentom_evaluator.py` (`check_*_answer`).
- Corpus metric ← `src/evaluate.py` (macro-F1 + accuracy + corruption rate).

Prompt fidelity was verified by diffing `doc_to_text` against the original `build_prompt`
output for every genre (byte-identical modulo the baked-in system line).
