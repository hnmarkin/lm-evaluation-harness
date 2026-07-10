# ToMBench — lm-eval adapter

Faithful port of **ToMBench** (Chen et al., ACL 2024; [arXiv:2402.15052](https://arxiv.org/abs/2402.15052)),
adapted from the read-only vendored repo at `benchmarks/ToMBench/`.

Bilingual (zh/en) multiple-choice Theory-of-Mind benchmark: 2,860 questions over 8 ToM tasks
and 31 ATOMS abilities.

---

## Architecture

| | |
|---|---|
| **doc unit** | one *(question × shuffle repeat)* — 2,860 questions × 5 repeats = **14,300 docs per task** |
| **output_type** | `generate_until` — the system prompt demands `[[X]]`, and `get_results.py` parses free text |
| **decoding** | greedy (`do_sample: false`), mirroring `run_huggingface.py` |
| **tasks** | `tombench_{zh,en}` and `tombench_{zh,en}_cot` (+ 4 `*_rawgold` parity variants) |
| **metrics** | 57 custom metrics, each with its own `aggregation: !function` |

### Why one multi-metric task, and not a leaf MATRIX

ToMBench asks **one question per prompt**, which normally implies a per-item leaf MATRIX. But its
*scoring* is cross-item in two independent ways, and neither survives being split across leaves:

1. **Majority vote over 5 shuffled repeats** (paper Sec. 4.1). Decoding is greedy, so the option
   order — which lives in the prompt — is the protocol's *only* stochastic element. `repeats: 5`
   would re-run an identical prompt and vote over five identical answers, a no-op. The five repeats
   are therefore materialised as five pre-shuffled docs, and the vote happens in the aggregation.
   **This is the faithful implementation, not a workaround.**
2. **Two overlapping views of the same docs.** The *task view* buckets by source file (8 of the 20);
   the *ability view* buckets by ATOMS ability, which cuts across files. No single leaf grain serves
   both, and the roll-up is a nested macro hierarchy that `aggregate_metric_list` (mean-only) cannot
   express.

So this adapter uses the **FANToM pattern**: `process_results` emits one payload per doc under all 57
metric names, and each metric's custom `aggregation` reads its own cell out of a memoised roll-up.

### The scoring hierarchy (verified numerically against the paper)

```
per-task acc     = micro over that task's questions      (8 task files, 2,470 questions)
task_avg         = MACRO mean of the 8 tasks             <- Table 2 "AVG."
per-ability acc  = micro over that ability's questions   (all 20 files, 2,860 questions)
category acc     = MACRO mean of that category's abilities
ability_avg      = MACRO mean of the 6 categories        <- Table 3 "AVG."
coherent_<task>  = fraction of stories answered ENTIRELY correctly   <- Fig. 4 / Table 22
coherent_avg     = MACRO mean of the 8 tasks
acc              = micro over all 2,860 questions        (auxiliary; not a paper number)
```

Both AVG columns are **macro**, not micro. Checked against Table 2 (GPT-4-1106 zh: `602.6/8 = 75.3` ✓;
micro would give 78.5) and Table 3 (Emotion zh: `531/7 = 75.9` ✓; micro would give 78.8).

Values are reported in `[0, 1]`; the paper's tables are the same numbers ×100.

---

## Usage

```bash
# System prompt is delivered via `description:` -> --apply_chat_template is REQUIRED, and
# ENFORCED: every task routes through `utils.ToMBenchTask`, which raises without the flag.
~/miniconda3/envs/eval_env/python.exe -m lm_eval run \
    --model hf --model_args pretrained=Qwen/Qwen3-1.7B,dtype=float16 \
    --tasks tombench_en --apply_chat_template --batch_size 8 \
    --output_path ./results --log_samples

# all four reported cells (zh/en x vanilla/CoT)
--tasks tombench          # the tag

# plumbing smoke (use multiples of 5 so repeat-groups stay whole)
--tasks tombench_en --limit 20 --apply_chat_template
```

Run under `eval_env` (Python 3.13); set `PYTHONIOENCODING=utf-8` on Windows.

### Reasoning models

`extract_answer` is vendored verbatim, and its last-resort branch scans **backwards** for any
`A`/`B`/`C`/`D` character. On a reasoning model that emits a `<think>` block, that returns a letter
from the *reasoning trace*, and its `[[A]]`-first ladder matches any `[[A]]` anywhere — including the
system prompt's own example if the model echoes it. Both hazards exist in the original too.

The paper evaluated non-reasoning chat models. To reproduce that condition, **disable thinking**:

```bash
--model_args pretrained=Qwen/Qwen3-1.7B,enable_thinking=False
```

Without it, a thinking model's `tombench_*` (vanilla) run is not the paper's vanilla condition, and
its answers are parsed out of a reasoning trace.

### Cost

5 repeats × 2,860 questions = 14,300 generations per task. The paper used one round for the GPT-4-*
models ("very consistent answers across different option orders"); to reproduce that, set
`try_times: 1` in `dataset_kwargs`.

`--limit N` keeps repeat-groups whole when `N` is a multiple of 5 (a doc's five repeats are adjacent),
so the majority vote stays correct — but a capped run reaches only the first task file, leaving most
`task_*` / `ability_*` / `coherent_*` metrics `NaN`. Report numbers only from a full run.

---

## Faithfulness deviations

Everything below is a deliberate, documented departure. Everything *not* listed is byte-identical to
ToMBench's own code, as enforced by `parity_check.py` (see "Verification").

### 1. The 5 shuffles are seeded per-item, not drawn from one global stream

`run_api.py` calls `random.seed(42)` once and consumes a single shuffle stream across all files in
`os.listdir()` order — which is filesystem-dependent, so the exact permutations are not reproducible
on another machine (and `eval_api.sh` passes `--task ""`, taking that path). We draw each permutation
from a `random.Random` seeded on `(file, row, repeat)`.

The *process* is identical — five uniformly random permutations per item, majority-voted — and which
five you draw is precisely the option-order noise the vote exists to average out. Only the particular
draw differs.

### 2. Absent options are `NaN`, not `null` — so `format_prompt_2` is unreachable in the original

The released JSONL encodes a missing option as bare `NaN` (pandas' export of an empty cell, and of
the literal string `"None"`). `run_api.py` gates on `d['选项C'] != None`, which is **True** for `NaN`,
so it takes the 4-choice branch and then dies on `float.replace`. **`run_api.py` cannot run the
released `data/*.jsonl` at all.**

We treat `NaN` as absent and render the 2-choice template — the code's evident intent, and what the
paper describes ("for true/false questions … the options are simply yes/no"). This affects **483 en /
484 zh items (~17% of the corpus)**: 280 in Faux-pas Recognition Test and 203 in Strange Story Task
(`"Is what Zhang Yang says true?"` → Yes/No).

### 3. Arity is gated on each language's own option column

`run_api.py` always tests the *Chinese* `选项C` regardless of `--language`. That disagrees with the
English column on exactly one corrupt row (Strange Story Task #292, whose Chinese options were
overwritten with a yes/no pair that doesn't answer its own "Why did Li Hua do this?" question, while
the English row keeps four sensible options). Gating per-language lets English keep its four real
options. One row of 2,860.

### 4. The 31 abilities are recovered by normalising the raw `ABILITY` field

The data carries **34** distinct raw ability strings; `get_results.py` buckets on the raw string and
therefore reports 34 groups, not the paper's 31. Three defects separate them:

| defect | example | fix |
|---|---|---|
| compound label | `"Belief: Location false beliefs Belief: Second-order beliefs"` | → `Belief: Second-order beliefs` |
| split ability | `"Desires influence on actions"` (76) + `"…on emotions (beliefs)"` (24) | → `Desires influence on actions/emotions` (100) |
| whitespace / casing | `"Information-knowledge links "`, `"Non-literal communication:"` | strip / re-case |

This is **proven, not guessed**: after normalisation the per-ability question counts match the paper's
Table 18 **exactly — all 31, summing to 2,860** (asserted in `loader_check.py`).

### 5. One upstream gold label is corrected (main tasks) — `*_rawgold` reproduces the bug

`Knowledge-Attention Links.jsonl` row 10 has `ANSWER == 'A. '` (with a period and a trailing space).
Predictions are always bare letters, so under `get_results.py` **that item scores wrong for every
model, on every run**. The typo is upstream — `ToMBench_release_v1_0618.xlsx`, sheet
`Knowledge-attention links`, row 12 has it too — and its origin is visible: that row's Chinese option A
literally begins `'A. 小芳说…'`, so the option's label was pasted into the answer cell.

The paper's own numbers cannot adjudicate this (its Table 21 row for that ability tops out at 55%,
never approaching the 95% ceiling the bug imposes).

* `tombench_{zh,en}[_cot]` — `fix_gold_typo: true` (default): the leading letter is taken.
* `tombench_{zh,en}[_cot]_rawgold` — `fix_gold_typo: false`: reproduces `get_results.py` exactly.

The ability has 20 items, so the choice is worth up to **5pp on `ability_knowledge_attention_links`**,
~1.25pp on `category_knowledge`, ~0.2pp on `ability_avg`, and **nothing** on any task-view metric
(Knowledge-Attention Links is one of the 12 ability-only files).

### 6. The coherent test's story ids are reconstructed from `INDEX`

There is no story-id column. `INDEX` is a question counter that restarts at 1 for each new story, so
counting restarts recovers the story blocks. This reproduces the paper's per-task story counts exactly
for **7 of the 8 tasks**, including the irregular ones:

| task | questions | stories | paper | block sizes |
|---|---|---|---|---|
| Unexpected Outcome | 300 | 100 | 100 ✓ | 100×3 |
| Scalar Implicature | 200 | 100 | 100 ✓ | 100×2 |
| Persuasion Story | 100 | 100 | 100 ✓ | 100×1 |
| False Belief | 600 | 100 | 100 ✓ | 100×6 |
| Ambiguous Story | 200 | 100 | 100 ✓ | 100×2 |
| Hinting | 103 | 93 | 93 ✓ | 83×1 + 10×2 |
| Strange Story | 407 | 201 | 201 ✓ | 197×2 + 3×3 + 1×4 |
| **Faux-pas** | 560 | **141** | **140** ✗ | 139×4 + 1×3 + 1×1 |

Persuasion Story coming out as 100 one-question stories independently corroborates the paper's aside
that "for all ToM tasks **except** the Persuasion Story Task (PST), most stories are associated with
multiple coherent questions".

Note what that sentence does *not* say: PST is **not** excluded from the coherent test. The paper plots
it in Fig. 4 and observes that "no performance drop occurs in the persuasion story task", precisely
because each PST story has exactly one question. So `coherent_pst` is identically equal to `task_pst`
by construction, and `coherent_avg` macro-averages all 8 tasks. Do not "fix" this by dropping PST.

**Faux-pas is a defect in the released file**, not a reconstruction failure: the "Xiao Wang roommate"
story lost its *"Does anyone say something inappropriate?"* question (3 questions remain), and a
"Grandpa dies…" story survives as a single orphaned question (its other three are absent — "nursing
home" occurs nowhere else in the file). 139×4 + 3 + 1 = 560. Effect on `coherent_frt`: a 141 vs 140
denominator plus one trivially-passable single-question story, i.e. **under one point**.

### 7. Known original-code quirks, reproduced deliberately

* **`str.replace`, not a prefix strip.** `_strip_label` reproduces `d['选项A'].replace("A. ", "")`
  exactly. Most option values carry no prefix (10,372 of 10,474 English values), and 752 Chinese
  values use a space-less `"A."` that the call leaves untouched. Anything "smarter" changes the prompt.
* **First-match-wins de-map.** English `Unexpected Outcome Test` rows 165/167 render
  `Angry / Thrilled / Angry / Surprise`, so both "Angry" positions map to `"A"` and no position ever
  maps to `"C"`. Reproduced. (Neither row's gold is C, so nothing is unscorable.)
* **Two-choice items keep the four-letter map.** A model answering `C`/`D` on a yes/no item de-maps to
  `""` and scores wrong rather than erroring — as in the original.
* **`extract_answer` defaults to `"A"`** when it finds no letter at all. Kept.
* **Tie-break in the vote.** `most_common_element` returns the first-seen prediction on a tie, so the
  aggregation sorts each item's payloads by `rep` before voting.

---

## Verification

`parity_check.py` imports ToMBench's **own** `prompts.py` / `run_api.py` / `get_results.py` (stubbing
`openai` and `tqdm`, which it never needs) and asserts byte-level agreement. Run from the repo root:

```bash
~/miniconda3/envs/eval_env/python.exe lm-evaluation-harness/lm_eval/tasks/tombench/parity_check.py
```

```
[1/2] prompts + de-maps: 16190 renderings byte-identical to run_api.py
      (1 skipped: the 1 zh/en arity-disagreement row, per-language gate)
[2/2] extract_answer (20k strings) + most_common_element (5k lists): identical
[+]   all 8 task YAML `description:` blocks == prompts.py system prompts
PARITY OK
```

`loader_check.py` covers the **corpus** side, which `parity_check.py` never touches — it pins every
count and mapping this adapter depends on, so a later edit to `normalize_ability` / `_story_ids` /
the arity gate, or a bump of the `benchmarks/ToMBench` submodule, fails loudly instead of quietly
reporting plausible numbers:

```bash
~/miniconda3/envs/eval_env/python.exe lm-evaluation-harness/lm_eval/tasks/tombench/loader_check.py
```

```
[1/4] 31 abilities x exact Table-18 counts (sum 2860); 8 task shapes incl. story ids
[2/4] loader: 2860x5 docs/lang, reps 0-4, 2470 task items, 483 en / 484 zh two-choice, every gold de-mappable
      gold typo Knowledge-Attention Links#10: 'A. ' -> 'A' (main) / 'A. ' (rawgold), 1 row affected
[3/4] 57 metric names == metric_list; all-correct -> 1.0, all-wrong -> 0.0
[4/4] exactly 1 zh/en arity disagreement (Strange Story Task#292); C/D always vanish together
LOADER OK
```

### Shrinking `--log_samples` output

Because scoring is cross-item, `process_results` emits the *same* payload under all 57 metric names,
and lm-eval's `example.update(metrics)` writes all 57 into every logged sample — roughly **90% of the
samples file**. `dedupe_samples.py` verifies the 57 copies really are identical (an invariant of this
pattern: if they ever diverge, the aggregations are silently reading different data) and collapses
them into one `tombench_payload` key. Measured ~4.6× smaller.

```bash
python lm-evaluation-harness/lm_eval/tasks/tombench/dedupe_samples.py results/**/samples_tombench_*.jsonl
python lm-evaluation-harness/lm_eval/tasks/tombench/dedupe_samples.py samples_tombench_zh.jsonl --check-only
```

Smoke-tested end-to-end with `Qwen/Qwen3-1.7B` (`enable_thinking=False`) on `tombench_{en,zh}` and
`tombench_en_cot` under `--apply_chat_template`. Observed the de-map doing its job: across the five
shuffles of one item the model emitted `[[D]] [[B]] [[C]] [[C]] [[B]]`, all five de-mapping to the same
canonical `A` — the model consistently chose the same option *text*.

---

## Not built (deferred)

* **Per-task or per-ability leaf tasks** for selective/diagnostic runs. The 57 metrics already surface
  every partition from one run; leaves would only help to *run* a subset.
* **`try_times: 1` task variants** for GPT-4-class models. Available today via `dataset_kwargs`.
* **The human baseline** (paper's 20 graduate students) — not reproducible from data.
