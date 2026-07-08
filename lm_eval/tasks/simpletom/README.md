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
   of "combo" or "conditioned" anywhere in the paper; the only reported numbers are per-category
   accuracy with a Wald 95% CI over the 1,147-sample category. So there is no defined formula to
   reproduce — `action_combo` is dead code in the source repo, not a benchmark requirement. The
   three native per-category accuracies below **are** the complete faithful reproduction of the
   paper's headline result.

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
  original module directly and diffing rendered prompts over 120 sampled docs (40/config, both
  variants): **0 mismatches**. The unused `extra_question`/`add_either_option`/
  `add_neither_option` parameters are dropped since `run_inference.py`'s `main()` never passes
  them (dead parameters for the released protocol — confirmed binary-only, matching the dossier's
  open question).
- **`process_results_vanilla` / `process_results_cot`** are a vendored port of
  `inference/utils.py:169-218` (`process_answer`)'s letter-extraction cascade (exact substring
  presence checks in `(A)`/`(B)`/`(C)`/`(D)` → `(A`/`(B`/... → bare `A`/`B`/... priority order,
  including its ambiguity-nulling behavior in the 2nd/3rd tiers) plus its `re.sub` prefix-strip
  (the CoT `".*the answer is"` pattern uses `(?ims)` dotall, so it greedily strips everything up
  to the **last** occurrence — verified byte-identical against the original on 10 synthetic
  generations, including a multi-occurrence "the answer is" case). The logprob/`probability`
  bookkeeping is dropped (`generate_until` returns text only, and the original never used it for
  scoring — only for logging).
- **`description: "You are a helpful assistant."`** on every template — the original's fixed
  system message (`run_inference.py:80`), rendered as an actual system-role turn under
  `--apply_chat_template`.

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
setting — `simpletom_vanilla_*` is the paper's headline Table 2 protocol.

## Metrics

- **`acc`** — 1.0 if the extracted letter matches `answerKey`, else 0.0 (native `acc`,
  `weight_by_size: true` micro-average at the group level).

## Running

```bash
# Single leaf, non-chat smoke model
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
way to disable. Following the `bigtom`/`dyntom` lesson: `generation_kwargs` has no `"\n\n"` stop
and `max_gen_toks` is sized well beyond the paper's own 20 (vanilla) / 500 (CoT) — `512` /
`2048` — and `process_results` grades only the text after the **last** `</think>`
(`utils._strip_thinking`, vendored from `bigtom/utils.py`). Without this, a `<think>` block can
silently exhaust the token budget before the model reaches an answer, scoring as a spurious
"wrong" that looks like a model failure rather than a config issue.

**Capacity note:** a small-context smoke model (e.g. `sshleifer/tiny-gpt2`, 1024-token context)
cannot fit `max_gen_toks: 2048` (the CoT default) in its own context window — the harness raises
`max tokens to generate ... must be less than model's maximum sequence length` in that case.
Override with `--gen_kwargs max_gen_toks=100` for tiny-model plumbing smoke tests only; real runs
should use the shipped per-variant default.

## Faithfulness deviations (and why)

1. **`generate_until` + a deterministic extractor, not `multiple_choice`** — see "Two Phase-0
   corrections" above.
2. **`action_combo`/`action_combo_faithful` not built** — see "Two Phase-0 corrections" above;
   verified dead code, not a defined paper metric.
3. **`extra_question`/either/neither options dropped** — dead parameters in the released
   protocol (`run_inference.py`'s `main()` never sets them).
4. **`max_gen_toks` raised (512/2048) and `"\n\n"` stop dropped**, vs. the paper's own 20/500 —
   reasoning-model capacity fix, not a scoring change (see Running above).
5. **Greedy decoding** (`do_sample: false`, `temperature: 0.0`), matching the paper's own
   `temperature=0` main-table setting. (The paper's special-cased reasoning-model temperature
   overrides for o1/o3/GPT-5/DeepSeek-R1-via-API are a provider-API quirk in `inference/utils.py`,
   not applicable to local/hosted lm-eval models.)
6. **Convenience groups only** (`simpletom_vanilla`, `simpletom_cot`) — the paper reports the
   three per-category accuracies separately; the group mean is not a paper number (see Tasks
   above).

## Provenance

- Prompt logic ← `inference/utils.py:question_prompt`, `make_mcq` (vendored, byte-parity
  verified against the live original).
- Answer extraction ← `inference/utils.py:process_answer` (vendored, output-parity verified
  against the live original on synthetic generations).
- CoT instruction ← `inference/prompts.py:COT_PROMPT` (transcribed verbatim).
- System message ← `inference/run_inference.py:80` (`"You are a helpful assistant."`).
- Reasoning-model `<think>`-stripping pattern ← `bigtom/utils.py:_strip_thinking`.

## Deferred follow-ups (build only when asked)

- A `scenario_name`/severity-tier leaf breakdown (the dossier's candidate sub-partitions) — not a
  paper-reported table, so not built by default.
- An offline join of `--log_samples` from `simpletom_vanilla_aware` + `simpletom_vanilla_action`
  by story-id stem, if a user specifically wants a *reconstructed* conditional-accuracy number —
  clearly labeled as a reconstruction beyond what the paper defines, not a reproduction of it.
