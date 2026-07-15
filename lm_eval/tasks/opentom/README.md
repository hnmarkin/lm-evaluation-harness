# OpenToM (lm-eval adapter)

Faithful lm-eval port of **OpenToM** (Xu et al., ACL 2024; [arXiv:2402.06044](https://arxiv.org/abs/2402.06044)).
OpenToM presents an LLM-generated narrative about a *mover* relocating an entity (observed
or not by an *observer*) and asks first-/second-order Theory-of-Mind questions about the
physical world (location, fullness, accessibility) and the psychological world (attitude).
Free-text answers are mapped to fixed label sets and scored with **macro-averaged F1**
(the labels are imbalanced — the authors' recommended metric).

The adapter **vendors** OpenToM's own prompt assembly and answer-extraction logic from the
read-only submodule `benchmarks/OpenToM/` (never imported, never edited). Data is read from
`benchmarks/OpenToM/data/opentom.json` and `data/opentom_data/meta_data.json`.

**Paper protocol is baked in** (for direct comparability to a run of the original
`run_baseline.py` + `evaluate.py` on the same model). Rather than scoring all 596 narratives
as one pool, the adapter reproduces the original's sampling and aggregation exactly:

- **250-narrative subset.** `run_baseline.py`'s coarse path does `set_seed(42)` →
  `random.shuffle(list(meta_data.keys()))` → takes the first `num_batch*batch_size = 5*50`
  narratives (`sample_entries`). We replicate that shuffle byte-for-byte (verified
  `random.Random(42)` ≡ the global seed) and tag each doc with its batch index 0–4.
- **Mean ± std of per-batch metrics**, not one pooled score (`evaluate.py`: per-batch
  `f1_score`, then `np.mean`/`np.std` over the 5 batches).
- **Multihop fullness downsampled 4 → 2** per narrative, so the combined multihop metric is a
  balanced 2:2 fullness:accessibility mix (`sample_questions`). The adapter replays the original
  seed-42 NumPy draws in their exact call order (FO then SO for each shuffled narrative), retaining
  the same realized pair of fullness questions in every narrative.

## Tasks (group `opentom`)

Per-item protocol → leaf MATRIX (mirrors `hitom`/`dyntom`). The 7 leaves mirror the official
`data/opentom_data/` split files, restricted to the seed-42 250-narrative subset (verified
`n` over the subset in parentheses; each `n` splits evenly into 5 batches):

| task | genre / order | label space | n (250-subset) |
|---|---|---|---|
| `opentom_location_cg_fo` | coarse location, first-order  | binary Yes/No        | 500 |
| `opentom_location_cg_so` | coarse location, second-order | binary Yes/No        | 500 |
| `opentom_location_fg_fo` | fine location, first-order    | ternary orig/new/equal | 1000 |
| `opentom_location_fg_so` | fine location, second-order   | ternary orig/new/equal | 500 |
| `opentom_multihop_fo`    | multihop, first-order         | ternary (2 fullness + 2 accessibility / narr) | 1000 |
| `opentom_multihop_so`    | multihop, second-order        | ternary (2 fullness + 2 accessibility / narr) | 1000 |
| `opentom_attitude`       | attitude                      | ternary pos/neu/neg  | 250 |

Coarse vs fine location is a **question-phrasing split, not a re-scoring of the same rows**:
`'locate'` questions are fine (answer is a place string), the rest (`'initial'`) are coarse
(Yes/No) — exactly the `'locate'`/`'initial'` branch in `build_prompt.py`. (The original's
`--lg` flag achieves the same split implicitly: it scores all location rows one way and the
mismatched subset falls out as `gold=None` and is dropped.)

## Metrics

Each task reports (custom `!function` aggregations consuming the whole `(gold,pred,batch)` list
— the FANToM set-level pattern). Every metric is the **mean of the per-batch value over the 5
batches**, matching `evaluate.py` (per-batch `f1_score`/`accuracy_score`, then `np.mean`):

- **`macro_f1`**, **`acc`** — paper-faithful: corrupted predictions (the extractor cannot
  parse a valid label, `pred == -1`) are **dropped per batch** before scoring, matching
  `evaluate.py`. These are the numbers to compare against the original's "Average F1/Accuracy".
- **`macro_f1_std`**, **`acc_std`** — the population std of the per-batch values
  (`np.std`, ddof=0) — `evaluate.py`'s "Variance" line (compare "within std").
- **`macro_f1_strict`**, **`acc_strict`** — corrupted predictions counted as **wrong** (not a
  paper metric; a stricter companion).
- **`corrupt_rate`** — mean per-batch fraction of corrupted predictions (lower is better).
- `opentom_multihop_*` additionally report **`fullness_f1`** and **`accessibility_f1`** (the
  per-subgenre breakdown `evaluate.py` also prints; `macro_f1` is the combined "overall" `f13`,
  which is a *balanced* 2:2 mix thanks to the fullness downsampling above).

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
2. **Paper protocol reproduced** (250-narrative seed-42 subset, 5 batches × 50, mean ± std of
   per-batch macro-F1, and the exact seeded 2:2 fullness/accessibility selection). The fullness
   selection uses legacy `np.random.RandomState(42)` and always consumes draws in the original
   default-run order—FO then SO for each shuffled narrative—even when lm-eval loads just one leaf.
   This is a faithful reproduction, not a deviation; see the "Paper protocol" box above.
3. **System message baked into the prompt.** The original prepends a system turn
   `"You are an expert in modeling other's mental state."` (`load_baseline_model.py`). lm-eval's
   `--system_instruction` is a run-level flag (only active with `--apply_chat_template`), so to
   keep the task self-contained and model-agnostic we prepend it as the first line of the user
   prompt instead of a dedicated system role.
4. **Greedy decoding** (`do_sample: false`, `temperature: 0.0`), matching the paper's GPT runs
   (`temperature=0`) and open-model runs. Stop strings match the open-model (llama/mixtral)
   branch of `run_baseline.py` exactly: `\n` **and** `\n\n` (plus `</s>`, `<|im_end|>`).
   The lone `\n` matters — a multi-line answer would otherwise reach the substring extractors
   whole and mis-fire (e.g. `"Yes.\nIt's now in the basket"` → `"now"` contains `"no"` →
   corrupted). `max_gen_toks: 128` (answers are "without any explanation").
   **Reasoning models:** the default budget/stop assume a *direct-answering* model (the paper's
   setup — GPT-3.5/4, Llama-chat, Mixtral). A thinking model (e.g. Qwen3 with `<think>` blocks)
   will exhaust 128 tokens inside its reasoning and emit no parseable label → high `corrupt_rate`.
   For such models raise `--gen_kwargs max_gen_toks=...` and/or disable thinking; the drop-policy
   `macro_f1`/`acc` then score only the parsed answers (`corrupt_rate` flags the rest).
5. **Vanilla prompting only.** CoT / SimToM / Self-Ask variants are deferred (documented
   follow-ups), as is **OpenToM-L** (`opentom_long.json`) and **mover/observer perspective**
   filtering (`--perspective`, default `all` reproduced).
6. **`macro_f1`/`acc`/`corrupt_rate` match `evaluate.py` exactly** — the per-batch aggregation
   was verified byte-for-byte (<1e-9) against an independent reimplementation of `evaluate.py`'s
   mean-of-per-batch logic across all 7 leaves and both multihop subgenres. `corrupt_rate` uses
   `evaluate.py`'s denominator (all items in the batch, incl. `gold=None`), meaned over batches.

## Comparability caveat — fine-location (`opentom_location_fg_*`)

The **released** `benchmarks/OpenToM/` cannot produce fine-location numbers as shipped, so those
two leaves cannot be diffed against an unmodified original run:

- `run_baseline.py`'s fine path loads `location_fg_fo.json` / a `baseline_results/v2_results/
  sampled_keys/{model}.json` that **do not exist** (the shipped files are `*_fg_fo_new.json`; no
  `baseline_results/` dir) → it crashes. Only `--lg coarse` runs on released data (5 tasks:
  `location_cg_*`, `multihop_*`, `attitude`) — exactly the 5 leaves that *are* directly comparable.
- `evaluate.py` scores fine-location via the positional `meta_data[key]['plot_info'].values()`
  unpack, which on the released key order `(mover, eoi, original_place, move_to_place, observer)`
  binds `move_to_place = observer` (a **person's name**) → broken golds. Our by-key binding
  (deviation 1) is correct-by-intent but will therefore **not** match a buggy original run.

**Bottom line:** the 5 coarse-path leaves are directly comparable to an original `--lg coarse`
run on the same model. The 2 fine-location leaves are comparable only if the original is first
fixed (rename `*_new.json`, replace the positional unpack with by-key) — in which case our
numbers match its *intent*.

## Provenance

- Prompt templates ← `src/prompts/chatgpt_opentom_prompts/*.txt` (transcribed verbatim).
- Prompt question mods (vanilla path) ← `src/utils/build_prompt.py`.
- Answer extractors ← `src/evaluate/opentom_evaluator.py` (`check_*_answer`).
- Corpus metric ← `src/evaluate.py` (macro-F1 + accuracy + corruption rate).

Prompt fidelity was verified by diffing `doc_to_text` against the original `build_prompt`
output for every genre (byte-identical modulo the baked-in system line).
