# SimpleToM (lm-eval adapter)

Faithful lm-eval port of **SimpleToM** — "SimpleToM: Exposing the Gap between Explicit ToM
Inference and Implicit ToM Application in LLMs" ([arXiv:2410.13648](https://arxiv.org/abs/2410.13648),
ICLR 2026). Data loads directly from HF `allenai/SimpleToM` (three configs: `mental-state-qa`,
`behavior-qa`, `judgment-qa`, 1,147 rows each, `test` split only) — no submodule/local copy needed,
and no custom loader: a clean HF `dataset_path`/`dataset_name` pair reads it natively.

Each story presents an information asymmetry (e.g. a bag of chips has mold Mary can't see) and is
paired with one binary-forced-choice question from one of three families: **mental-state inference**
("Is Mary likely to be aware that ...?"), **behavior prediction** ("What will Mary likely do
next?"), or **behavior judgment** ("... was that reasonable?"). The paper's headline finding is that
accuracy drops sharply aware → action → judge — the three families are reported as **separate**
numbers, not pooled.

## Two Phase-0 corrections to the benchmark's own recon dossier

1. **`generate_until` + a letter extractor, not `multiple_choice`.** The benchmark's own eval code
   (`benchmarks/SimpleToM/inference/run_inference.py` + `inference/utils.py:process_answer`)
   generates free text at `temperature=0` (greedy, `max_tokens=20` non-CoT / `500` CoT) and
   regex-parses the response for `(A)`/`(B)` — it never ranks the two option strings by
   loglikelihood. Same fork as `hitom`/`bigtom`.
2. **`action_combo` / `action_combo_faithful` are not a real reported metric — not built.** The
   repo's `basic_stats` (`inference/utils.py:151`) buckets results by `id.endswith(cat)` for
   `cats=['aware','action','judge','action_combo','action_combo_faithful']`, which looks like a
   cross-item conditional-accuracy metric joining the `_aware`+`_action` rows of the same story.
   Verified against the **live** `allenai/SimpleToM` dataset: no row's `id` ever ends in
   `_action_combo` or `_action_combo_faithful` for any of the three configs — those two branches
   of `basic_stats` are unreachable on the released data. Verified against the **full published
   paper text** (arXiv:2410.13648v2, fetched directly, not from training memory): zero occurrences
   of "combo" or "conditioned" anywhere in the paper. So `action_combo` has no defined formula to
   reproduce — it is dead code in the source repo, not a benchmark requirement. The three native
   per-category accuracies below are the complete faithful reproduction of the paper's **headline
   Table 2** result (reported as per-category accuracy with a Wald 95% CI over the 1,147-sample
   category).

   This does **not** mean the paper defines no cross-item analysis: **Figure 3** does. See
   "Deferred follow-ups".

These two findings simplify the build relative to the recon dossier's `[MATRIX, AGG-FN]` /
`CROSS-ITEM` flags: no LOADER, no RESHAPER, no cross-item aggregation — just `MATRIX` + a standard
letter-`EXTRACTOR`, the `hitom` shape.

## Architecture

- **doc = one (config, row) item**, one question per prompt (per-item protocol) → leaf MATRIX
  (`hitom`/`bigtom` style), not a batched container.
- **`dataset_path: allenai/SimpleToM`**, `dataset_name` set per leaf to the HF config
  (`mental-state-qa` / `behavior-qa` / `judgment-qa`). No `custom_dataset`/loader script.
- **`doc_to_text_vanilla` / `doc_to_text_cot`** are **vendored, byte-parity ports** of
  `inference/utils.py:220-262` (`question_prompt`, `make_mcq`) — verified by importing the
  original module directly and diffing rendered prompts over **all 3,441 rows × both variants
  (6,882 prompts): 0 mismatches**. The unused `add_either_option`/`add_neither_option` parameters
  are dropped, and `extra_question` is left unbuilt (see deviation 3 — it is *not* dead code).
- **`process_results_vanilla` / `process_results_cot`** are a vendored port of
  `inference/utils.py:169-218` (`process_answer`)'s letter-extraction cascade (exact substring
  presence checks in `(A)`/`(B)`/`(C)`/`(D)` → `(A`/`(B`/... → bare `A`/`B`/... priority order,
  including its ambiguity-nulling behavior in the 2nd/3rd tiers) plus its `re.sub` prefix-strip
  (the CoT `".*the answer is"` pattern uses `(?ims)` dotall, so it greedily strips everything up
  to the **last** occurrence). Output-parity verified against the live `process_answer` over a
  156-case adversarial grid: **0 mismatches** on `predicted`, `acc`, and `na`. The
  logprob/`probability` bookkeeping is dropped (`generate_until` returns text only, and the
  original never used it for scoring — only for logging).

  The adapter restructures the tier-2 → tier-3 cascade (the original guards tier 3 with
  `if answer_index is None`, so an ambiguous tier 2 skips it; the adapter falls through and re-runs
  tier 3). This is provably equivalent: tier 2 yields `-1` only when ≥2 of `(A`/`(B`/... are
  present, which implies ≥2 bare letters, which forces tier 3 to `-1` too. Both return raw text.
- **`description: "You are a helpful assistant."`** on every template — the original's fixed
  system message (`run_inference.py:80`), rendered as an actual system-role turn **only under
  `--apply_chat_template`** (see below).

## `--apply_chat_template` is required

Without it, `ConfigurableTask.fewshot_context` (`lm_eval/api/task.py:974-979`) builds
`Message("system", description)` with an empty `_delimiter`, and the non-chat branch does
`"".join(m.to_text())` — the system message is **glued onto the prompt with no separator**.
Verified rendering of the real task:

```
'You are a helpful assistant.Given the following story, answer the question by giving the correct answer choice, (A) or (B).\n\nStory:\n...'
```

This is upstream `maybe_delimit` behavior (`lm_eval/api/utils.py:7-17`) affecting every adapter
that uses `description`; it is documented here, not patched. **Always pass
`--apply_chat_template` for a faithful prompt.** The `gpt2` command below is a plumbing-only smoke
test and does *not* produce the paper's prompt.

## Tasks

6 leaves + 2 groups, over 2 prompting variants (`vanilla`, `cot` — mirrors the repo's own
`--use-cot` flag) × 3 question families:

- `simpletom_vanilla_aware`, `simpletom_vanilla_action`, `simpletom_vanilla_judge`
- `simpletom_cot_aware`, `simpletom_cot_action`, `simpletom_cot_judge`
- Groups `simpletom_vanilla`, `simpletom_cot` (mean of their 3 leaves) are a **convenience
  bundle for one-command runs only — not a paper-reported number**. The paper deliberately keeps
  aware/action/judge separate (that gap *is* the finding); read the three leaf numbers, not the
  group mean.

CoT is a **separate intervention** in the paper (§5.3, tested on 3 models), not the main-table
setting — `simpletom_vanilla_*` is the paper's headline Table 2 protocol. Note that
`simpletom_cot_aware` has no paper counterpart: Table 3's CoT column covers behavior and judgment
only. It is nonetheless repo-faithful, since the repo's `--use-cot` applies to every subset.

## Metrics

- **`acc`** — 1.0 if the extracted letter matches `answerKey`, else 0.0 (native `acc`,
  `weight_by_size: true` micro-average at the group level). **This is the paper-comparable
  number**, and it is byte-faithful to `process_answer`.
- **`na_frac`** — fraction of items whose extracted prediction is not `A` or `B` (so a `C`/`D` or
  an unparseable raw-text fallback both count). Same definition as the repo's
  `basic_stats` (`inference/utils.py:160`), which computes it on every run. **Repo-side
  diagnostic, not a paper-comparable number** — `na_frac` and every synonym ("no answer",
  "unparse", "malformed", "abstain", ...) appears nowhere in the paper's main body or appendices;
  Tables 2 and 3 report accuracy only.
- **`truncated_think_frac`** — fraction of generations containing `<think>` but no `</think>`.
  **Adapter-only diagnostic**, in neither the repo nor the paper. A non-zero value means
  "raise `max_gen_toks`", not "the model is bad" — see the reasoning-model note below. It does not
  affect `acc`; it only makes an otherwise invisible failure observable.

## Running

```bash
# Single leaf, non-chat smoke model. PLUMBING ONLY -- see "--apply_chat_template is required".
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=gpt2 \
  --tasks simpletom_vanilla_aware --include_path lm-evaluation-harness/lm_eval/tasks \
  --limit 5 --log_samples --output_path out/

# Full vanilla group (paper's main-table protocol), chat model
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<chat-model> \
  --tasks simpletom_vanilla --include_path lm-evaluation-harness/lm_eval/tasks \
  --apply_chat_template --batch_size auto --output_path out/ --log_samples

# CoT intervention
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<chat-model> \
  --tasks simpletom_cot --include_path lm-evaluation-harness/lm_eval/tasks \
  --apply_chat_template --batch_size auto --output_path out/ --log_samples
```

**Reasoning-capable chat models (Qwen3, DeepSeek-R1, ...) think by default**, even under the
vanilla "Respond with just ..." instruction — a model-level chat-template default the task has no
way to disable. `generation_kwargs` therefore has no `"\n\n"` stop, `max_gen_toks` is sized well
beyond the paper's own 20 (vanilla) / 500 (CoT) — **`2048` / `4096`** — and `process_results` grades
only the text after the **last** `</think>` (`utils._strip_thinking`, vendored from
`bigtom/utils.py`).

This matters more than it looks. `_strip_thinking` is a **silent no-op when `</think>` never
appears**, so a think block truncated by the token cap gets handed to the extractor whole — and the
extractor's first tier returns the *lowest letter present anywhere in the text*, not the first one
stated. A truncated ramble almost always scores as a confident `A`.

Measured on `Qwen/Qwen3-1.7B`, `simpletom_vanilla_action`, 40 docs, `--apply_chat_template`, at the
**old 512 budget**: 14/40 generations truncated mid-`<think>`. Of those 14, **12 extracted a
spurious `A`** and 2 fell back to raw text. Their `acc` was **0.357 — exactly the 5/14 whose gold
happened to be `A`.** Zero model signal, pure gold-distribution luck. The 26 that finished thinking
scored `0.846`. Raising the cap to the shipped 2048 removes the truncation entirely and lifts
reported `acc` from `0.675` to `0.825`. **`na_frac` caught only 2 of those 14 contaminated docs** —
it is not a sufficient guard, which is why `truncated_think_frac` exists.

Budgets were chosen against measured distributions (`Qwen3-1.7B`, uncapped enough that nothing
truncates):

| variant | measured on `*_action` (the longest family) | median | p90 | max | shipped cap |
|---|---|---|---|---|---|
| vanilla | tokens to `</think>` | 354 | 886 | 1213 | **2048** (1.69×) |
| cot | total generated tokens | 504 | 1037 | 1864 | **4096** (2.20×) |

Truncation rate at the old 512 cap: **35%** on `action` alone, **15.8%** averaged across all three
families (`aware` thinks least, `action` most) — the two figures are consistent, not contradictory.
By budget, across families: 512 → 15.8%, 1024 → 1.7%, 1536 → 0%, 2048 → 0%.

Watch `truncated_think_frac`: on a real run it should be `0.0`. If it is not, **raise
`max_gen_toks`** — do not revert. All of the above is from a *1.7B* model; larger reasoning models
think substantially longer, and the shipped caps have not been confirmed against one (see
"Deferred follow-ups").

**Capacity note:** a small-context smoke model (e.g. `sshleifer/tiny-gpt2`, 1024-token context)
cannot fit either shipped default (`2048` vanilla / `4096` CoT) in its own context window — the
harness raises `max tokens to generate ... must be less than model's maximum sequence length`.
Override with `--gen_kwargs max_gen_toks=100` for tiny-model plumbing smoke tests only; real runs
should use the shipped per-variant default.

## Faithfulness deviations (and why)

1. **`generate_until` + a deterministic extractor, not `multiple_choice`** — see "Two Phase-0
   corrections" above.
2. **`action_combo`/`action_combo_faithful` not built** — see "Two Phase-0 corrections" above;
   verified dead code, not a defined paper metric.
3. **`extra_question` / either / neither options dropped** — none is reachable from the released
   protocol (`run_inference.py`'s `main()` never sets them). But only two of the three are
   genuinely unused: the paper never adds an "either"/"neither" third choice anywhere.
   **`extra_question` is not dead code** — it is the mechanism behind the paper's "Patching mental
   state inference in the prompt (MS remind)" intervention (Appendix K.1), whose scores appear in
   **main-body Table 3** (§5.3) alongside the CoT column this adapter *does* build. Appendix K.1's
   example prompt is character-for-character what `question_prompt`'s `prompt_question_text` branch
   emits. Building it would mean writing new inference code rather than porting `run_inference.py`,
   so it is out of scope for a repo-faithful port. The CoT column is built because the repo exposes
   `--use-cot`; this asymmetry is deliberate.
4. **`max_gen_toks` raised (2048/4096) and `"\n\n"` stop dropped**, vs. the paper's own 20/500.
   **This is a scoring change, not just a capacity fix** — the budget interacts with the extractor's
   lowest-letter-wins first tier and moved measured `acc` from 0.675 to 0.825 on the 40-doc
   `Qwen3-1.7B` `vanilla_action` probe above. Raising it is nonetheless the correct direction: the
   paper's `max_tokens=20` is inapplicable to local reasoning models (under a 20-token cap a Qwen3
   generation has not even left its `<think>` block), and truncation *manufactures* a letter rather
   than recording a failure — the truncated docs score at their gold-`A` base rate, not at any
   measure of ability. See "Running".
5. **Greedy decoding** (`do_sample: false`, `temperature: 0.0`), matching the paper's own
   `temperature=0` main-table setting. (The paper's special-cased reasoning-model temperature
   overrides for o1/o3/GPT-5/DeepSeek-R1-via-API are a provider-API quirk in `inference/utils.py`,
   not applicable to local/hosted lm-eval models.)
6. **Convenience groups only** (`simpletom_vanilla`, `simpletom_cot`) — the paper reports the
   three per-category accuracies separately; the group mean is not a paper number (see Tasks
   above).
7. **Two diagnostic metrics added alongside `acc`** — `na_frac` (repo-defined) and
   `truncated_think_frac` (adapter-defined). Neither perturbs `acc`; see Metrics.
8. **`dataset_path` is unpinned** (no `revision`) — accepted reproducibility risk.

## Provenance

- Prompt logic ← `inference/utils.py:question_prompt`, `make_mcq` (vendored, byte-parity
  verified against the live original over all 6,882 rendered prompts).
- Answer extraction ← `inference/utils.py:process_answer` (vendored, output-parity verified
  against the live original over a 156-case adversarial grid).
- `na_frac` ← `inference/utils.py:basic_stats` (line 160).
- CoT instruction ← `inference/prompts.py:COT_PROMPT` (transcribed verbatim, equality-asserted).
- System message ← `inference/run_inference.py:80` (`"You are a helpful assistant."`).
- Reasoning-model `<think>`-stripping pattern ← `bigtom/utils.py:_strip_thinking`.

## Deferred follow-ups (build only when asked)

- **Confirm the 2048/4096 budgets on a larger reasoning model.** Every measurement above is from
  `Qwen/Qwen3-1.7B` (the 8B does not fit in 8 GB VRAM on the machine this was validated on). Run all
  six leaves on `Qwen/Qwen3-8B` with `--limit 24 --apply_chat_template --log_samples` and check that
  `truncated_think_frac == 0.0` everywhere. If it is not, raise the caps further rather than
  reverting, and record the observed tokens-to-`</think>` distribution.
- **Figure 3's first-failure distribution — paper-defined, not reproduced.** The paper *does*
  specify one cross-item analysis: across the mental state → behavior → judgment questions of a
  shared story, "we record a failure for the mistake that occurs first", bucketed as *fail at
  mental state / fail at behavior / fail at judgment / all correct*. Reproducing it needs an offline
  join of `--log_samples` across the three families on the story-id stem. Deferred by choice — but
  unlike `action_combo`, it is a real paper metric, not a reconstruction.
- **The MS-remind and SysP interventions (Table 3)** — see deviation 3.
- A `scenario_name`/severity-tier leaf breakdown (the dossier's candidate sub-partitions) — not a
  paper-reported table, so not built by default.
