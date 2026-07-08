# BigToM (lm-eval adapter)

Faithful lm-eval port of **BigToM** — "Understanding Social Reasoning in LLMs with LLMs"
([arXiv:2306.15448](https://arxiv.org/abs/2306.15448)). The submodule at
`benchmarks/procedural-evals-tom/` (read-only) is the original Stanford BigToM repo. BigToM
presents a short procedurally-generated story about an agent and asks a binary forced-choice
question about the agent's **belief**, **action**, or **desire**, crossed against whether the
agent witnessed a world-state change (true/false belief) and whether a matched no-op control
event occurred instead (true/false control).

## Why this needed a real port, not a thin wrapper

The repo ships only 200 raw multi-variant rows
(`benchmarks/procedural-evals-tom/data/bigtom/bigtom.csv`) plus an expander script
(`code/src/generate_conditions.py`) that **splices sentences out of each raw story at fixed
positions** to build the actual per-condition test items — there is no frozen, shipped set of
test items (`PIPELINE-ONLY`). `utils.py` is a line-for-line **vendored port** of that script's
branching logic (attributed inline), reimplemented as a pure function that materializes items in
memory instead of writing CSVs under `benchmarks/`.

Grading similarly reproduces the repo's own `--mcq` protocol
(`code/src/evaluate_conditions.py` + `evaluate_llm.py`): the model is shown both options as
`a)`/`b)`, generates free text, and a **correct-key-first substring check** grades it — not a
loglikelihood ranking of the two option strings. That is a `generate_until` + EXTRACTOR protocol
(the `hitom` shape), not `multiple_choice`.

## Architecture

- **doc = one (variable, init_belief, condition) item.** Per-item protocol → leaf MATRIX
  (`hitom`/`opentom` style), not a batched container.
- **`utils.load(variable, init_belief, condition)`** re-derives one 200-item cell in memory from
  the raw CSV: for each raw row, apply the vendored per-variable sentence-splice construction
  (`_forward_common`/`_backward_belief`/`_percept_to_belief`) and the matching per-condition
  branch (`_apply_condition`), then apply the repo's own **deterministic per-cell answer-order
  shuffle** (`random.Random(0)`, sequential in row order — reproduces
  `evaluate_conditions.py`'s `random.seed(0)` + `random.shuffle(answers)`, since that script
  reseeds to 0 at the start of every condition-file run).
- **`doc_to_text`** is one shared Jinja template for *all four* prompting variants below (the
  user turn is identical between vanilla and CoT in the original chat prompts — only the system
  instruction and scoring differ): `"Story: {{story}}\nQuestion: {{question_with_choices}}"`.
- **`process_results_vanilla` / `process_results_cot`** vendor `evaluate_conditions.py`'s
  correct-key-first substring check (`_grade`) and, for CoT, the exact `parse_chat_response`
  (including its off-by-one quirk when `"Answer:"` is absent). The original's rare LLM-judge
  fallback (only reached when neither letter substring is found) is replaced by a **deterministic
  value-match** against the option text (hitom/DynToM pattern); if that also fails, the item is
  bucketed as `"error"` — kept **distinct** from a graded-wrong item, not silently folded into it.
- **1-shot uses lm-eval's native fewshot mechanism**, not hand-folded text:
  `fewshot_config: {sampler: first_n, samples: !function utils.list_fewshot_samples_{vanilla,cot}}`
  supplies the repo's own **fixed** Kofi/fishing-net exemplar (never sampled from `bigtom.csv`),
  rendered through the exact same `doc_to_text`/`doc_to_target` as real docs. Under
  `--apply_chat_template --fewshot_as_multiturn` this renders as genuine separate turns —
  verified byte-for-byte against `evaluate_llm.py`'s `ONE_SHOT_CHAT`/`ONE_SHOT_CHAT_COT`.

## Tasks

100 leaves + 4 groups, over 4 prompting variants (`0shot`, `0shot_cot`, `1shot`, `1shot_cot`,
naming mirrors `evaluate_conditions.py`'s own `method` values) × 25 (variable, init_belief,
condition) cells:

**24 core cells** (group members) — `{forward_belief, forward_action, backward_belief} ×
init_belief{0,1} × condition{true_belief, false_belief, true_control, false_control}`. This
matches the scope of the paper's *own* results-formatting script
(`code/analysis/format_model_results.py`, `VARIABLES = ['belief', 'action']` × both directions),
which is the strongest available evidence for what counts as "the headline table."

**1 extra cell, standalone** — `percept_to_belief` only ever emits `init_belief=1,
condition=true_belief` (gated by `generate_conditions.py`'s own write-condition); excluded from
the paper's own formatting script, so built but **not** counted in any group average.

Leaf naming: `bigtom_<variant>_<variable>_<init_belief>_<condition>`, e.g.
`bigtom_0shot_cot_forward_belief_1_false_belief`. Groups: `bigtom_0shot`, `bigtom_0shot_cot`,
`bigtom_1shot`, `bigtom_1shot_cot` (each = mean over its 24 core leaves; **not** a paper-reported
number — the paper reports per-condition bars, not one pooled BigToM average).

`backward_desire` and the `-n 100` subset runs some paper scripts used for cost reasons are
**not reproduced** — see Deviations below.

## Metrics

- **`acc`** — 1.0 if the correct-key-first substring check (or its deterministic value-match
  fallback) finds the correct answer, else 0.0 (an unparseable "error" item counts as not-correct
  here, matching the original's always-binary grading).
- **`error_rate`** — fraction of items where *neither* the letter nor the option text could be
  found in the (post-`<think>`) generation — i.e. where the original would have called an LLM
  judge. Kept as its own diagnostic metric rather than silently merged into "wrong."

## Running

```bash
# Vanilla, non-chat smoke model
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=gpt2 \
  --tasks bigtom_0shot_forward_belief_0_true_belief --limit 5 --log_samples --output_path out/

# Full core group, chat model (system role + genuine fewshot turns)
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<chat-model> \
  --tasks bigtom_0shot,bigtom_0shot_cot,bigtom_1shot,bigtom_1shot_cot \
  --apply_chat_template --fewshot_as_multiturn --batch_size auto --output_path out/ --log_samples

# percept_to_belief (standalone, not in any group)
lm-eval run --model hf --model_args pretrained=<model> \
  --tasks bigtom_0shot_percept_to_belief_1_true_belief --apply_chat_template --log_samples
```

**Reasoning-capable chat models (Qwen3, DeepSeek-R1, ...) think by default**, even under the
vanilla "keep your answer concise" instruction — the task has no way to disable this (that's a
model-level chat-template kwarg, not a task setting). `generation_kwargs` therefore drops the
`"\n\n"` stop and sizes `max_gen_toks` generously (1024, vs. the paper's own 20-25/220) for
*all four* variants, and `process_results` grades only the text after the **last** `</think>`
(`utils._strip_thinking`). Without this, a `<think>` block silently exhausts the token budget
before the model ever reaches `"Answer:"`, and the item scores as an unparseable "error" for
reasons that look like a model failure rather than a config issue — verified directly: with the
paper-sized `max_gen_toks: 64` and a `"\n\n"` stop, Qwen3-1.7B's `error_rate` was 0.8–1.0 on
several cells (thinking never reached a conclusion in budget); after the fix, most of that
`error_rate` disappears on the same cells.

A smaller residual `error_rate` remains even after the fix, from a different cause: the system
instruction's example format is transcribed verbatim from the original repo as
`"Answer:<option>)<answer>"`, using angle brackets as placeholder markup. Qwen3-1.7B
occasionally echoes that markup literally (`"Answer:<option>)b>"`) instead of substituting a real
letter/answer, which neither the letter check nor the value-match fallback can parse. This is a
property of the *original* instruction wording interacting with a smaller/weaker model, not
something introduced here — faithful transcription means it isn't reworded to avoid this.

## Faithfulness deviations (and why)

1. **`generate_until` + a deterministic extractor, not `multiple_choice`.** The paper's own
   `--mcq` protocol shows both options and grades free-text generation by substring match — not a
   loglikelihood ranking of the two option strings (which would never show the options to the
   model at all). Mirrors `hitom`.
2. **LLM-judge fallback replaced with a deterministic value-match + a distinct `"error"` bucket**
   (not folded into "wrong"). The original only reaches its LLM judge when neither `a)` nor `b)`
   appears in the response (rare, since the format is mandated by the system instruction) — see
   Metrics above.
3. **Deterministic per-cell shuffle**, reproducing `random.seed(0)` + sequential
   `random.shuffle(answers)` per condition file. A fresh `random.Random(0)` per cell, consumed in
   raw-CSV row order, reproduces exactly what a fresh run of the original would produce.
4. **All 4 prompting variants built** (`0shot`/`1shot` × vanilla/CoT). The fixed one-shot
   exemplar is wired through lm-eval's native `fewshot_config`/`list_fewshot_samples` mechanism,
   not hand-folded into the prompt text.
5. **`backward_desire` excluded.** It exists in `generate_conditions.py`'s branch logic but the
   driving `VARIABLES` loop never includes it — dead code in the source repo itself, not
   reachable by any script in that repo either.
6. **Inherited quirk, not corrected:** `backward_belief`'s `true_control` and `false_control`
   cells are **byte-identical items** (both use `actions[1]` and answers[1]/answers[0]) — this is
   `generate_conditions.py`'s own code (both branches are identical), not a bug introduced here.
   Faithfully reproduced per the "vendor, never edit" rule; flagged so it isn't mistaken for an
   adapter bug.
7. **`parse_chat_response`'s off-by-one quirk vendored verbatim.** When `"Answer:"` is absent,
   the original's `response.find('Answer:')` returns `-1`, so it slices at index 7 instead of
   returning the empty string. Not fixed; faithfully reproduced (in practice this only matters
   when there's no letter/value match either, so the item is already bound for the "error"
   bucket).
8. **Full 200-item cells**, not the `-n 100` subsets some of the paper's own shell scripts used
   for cost on some models/conditions. The complete 200-row benchmark is the correct default for
   an lm-eval task; `-n 100` was a paper-authors' cost choice, not part of the benchmark's
   definition.
9. **Greedy decoding** (`do_sample: false`, `temperature: 0.0`), matching the paper's own
   `temperature=0` eval runs.
10. **`max_gen_toks` sized well beyond the paper's own 20-25 (vanilla) / 220 (CoT)**, and no
    `"\n\n"` stop — see the reasoning-model note under Running.
11. **Core group scope = 24 cells**, matching `format_model_results.py`'s own
    `{belief, action} × {forward, backward}` scope; `percept_to_belief` built but excluded from
    the group average since the paper's own formatting script excludes it too.

## Provenance

- Expansion logic ← `code/src/generate_conditions.py` (`generate_conditions`, vendored per-variable
  branch).
- Grading/prompt logic ← `code/src/evaluate_conditions.py` (`evaluate_condition`) and
  `code/src/evaluate_llm.py` (`EvaluateLLM`, `parse_chat_response`, `ONE_SHOT*`).
- System instructions ← `code/prompt_instructions/{evaluate,evaluate_cot_chat_model}.txt`
  (transcribed verbatim).
- Headline-table scope ← `code/analysis/format_model_results.py`.

## Deferred follow-ups (build only when asked)

- An offline `--predict_only` + real LLM-judge script for exact parity on the rare
  neither-letter-nor-value-found cases (the deterministic value-match fallback above is the v1
  in-harness answer).
- The non-chat "completion-style" prompt framing (raw string concatenation for base/non-chat
  models, as opposed to the chat `description`/turn framing used here) — not built since this
  repo's other adapters target chat/instruct models via `--apply_chat_template`.
