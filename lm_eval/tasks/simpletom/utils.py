"""SimpleToM (arXiv:2410.13648, ICLR 2026) adapter.

Data loads natively from HF `allenai/SimpleToM` (configs `mental-state-qa` / `behavior-qa` /
`judgment-qa`, each a clean `test`-split table -- no custom loader needed).

The prompt-builder and answer-extractor below are **vendored ports** (attributed inline) of the
benchmark's own eval code, `benchmarks/SimpleToM/inference/utils.py` (`question_prompt`,
`make_mcq`, `process_answer`) and `.../inference/prompts.py` (`COT_PROMPT`) -- not reconstructed
from the paper, since the repo ships the actual eval code. See README.md for the full
faithfulness-deviation list, in particular:
  - This is `generate_until` + a letter extractor, not `multiple_choice`: the original protocol
    generates text at temperature 0 and regex-parses out the answer, it never ranks the option
    strings by loglikelihood.
  - The repo's `basic_stats` also buckets by `action_combo` / `action_combo_faithful`, but no
    released id ever ends in those suffixes (verified against the live HF dataset) and the
    published paper defines no such metric (verified against the full paper text: zero
    occurrences of "combo" or "conditioned") -- dead code in the source repo, not reproduced here.
  - `basic_stats`'s other per-category output, `na_frac`, IS reproduced (same definition:
    `predicted not in ['A', 'B']`). It is a repo-side diagnostic only: the paper reports accuracy
    alone and never mentions it. Do not treat `na_frac` as a paper-comparable number.
  - The paper DOES define one cross-item analysis -- Figure 3's first-failure distribution over
    mental state -> behavior -> judgment for a shared story. It is not built here because it needs
    an offline join of `--log_samples` across the three families. Deferred, not undefined.
"""

import re


_COT_PROMPT = (
    "Think step by step to arrive at an answer. Start your response by explaining your reasoning "
    'process and end your response with "Therefore, the answer is: " followed by '
)

_LETTERS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# PROMPT-FN -- vendored port of inference/utils.py:220-262 (`make_mcq`, `question_prompt`).
# All three optional parameters are dropped, because run_inference.py's main() never passes them
# and so none is reachable from the released protocol. Two of them, `add_either_option` and
# `add_neither_option`, are genuinely unused: the paper never adds a third "either"/"neither"
# choice anywhere. `extra_question` is different -- it is the mechanism behind the paper's
# "Patching mental state inference in the prompt (MS remind)" intervention (Appendix K.1, scored
# in main-body Table 3, section 5.3). Reproducing that column would mean writing new inference
# code rather than porting run_inference.py, so it stays out of scope here -- but it is not dead
# code, and the `prompt_question_text` slot below is where it would go.
# ---------------------------------------------------------------------------


def _make_mcq(question, answer_choices):
    choice_text = " \n".join(
        f"({label}) {choice}" for label, choice in zip(_LETTERS, answer_choices)
    )
    return f"{question} \n{choice_text}"


def _question_prompt(doc, final_prompt):
    story_text = doc["story"]
    question = doc["question"]
    answer_choices = list(doc["choices"]["text"])
    prompt_choice_labels = " or ".join(["(A)", "(B)", "(C)", "(D)"][: len(answer_choices)])
    prompt_choice_labels_quoted = re.sub(r"(\(.\))", r'"\1"', prompt_choice_labels)
    prompt_question_text = ""
    prompt = f"""Given the following story, answer the question by giving the correct answer choice, {prompt_choice_labels}.

Story:
{story_text}
{prompt_question_text}
Question: {_make_mcq(question, answer_choices)}

What is the correct answer? {final_prompt} {prompt_choice_labels_quoted}
"""
    return prompt


def doc_to_text_vanilla(doc):
    return _question_prompt(doc, "Respond with just")


def doc_to_text_cot(doc):
    return _question_prompt(doc, _COT_PROMPT)


# ---------------------------------------------------------------------------
# EXTRACTOR / METRIC-FN -- vendored port of inference/utils.py:169-218 (`process_answer`),
# minus the logprob-probability bookkeeping (lm-eval's `generate_until` returns text only, and
# the original never used `probability` for scoring -- only for logging).
# ---------------------------------------------------------------------------


def _strip_thinking(text):
    """Reasoning-capable chat models (Qwen3, DeepSeek-R1, ...) prepend a <think> block by
    default, something that did not exist when SimpleToM's own grading code was written and
    that the task has no way to disable (a model-level chat-template default, not a task
    setting). Grade only the text after the LAST </think> (see README's reasoning-model note;
    same fix as bigtom/utils.py:_strip_thinking)."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx != -1 else text


def _strip_answer_prefix(text, answer_prefix):
    # Vendored verbatim: `(?ims)` makes `.` match newlines (dotall), so for the CoT prefix
    # ".*the answer is" this greedily strips everything up to and including the LAST
    # occurrence of "the answer is", leaving only the text that follows it.
    return re.sub(f"(?ims){answer_prefix}:?", "", text).strip()


def _extract_predicted_letter(text):
    for idx, marker in enumerate(["(A)", "(B)", "(C)", "(D)"]):
        if marker in text:
            return ["A", "B", "C", "D"][idx]
    answer_index = None
    for idx, marker in enumerate(["(A", "(B", "(C", "(D"]):
        if marker in text:
            answer_index = idx if answer_index is None else -1
    if answer_index is not None and answer_index >= 0:
        return ["A", "B", "C", "D"][answer_index]
    answer_index = None
    for idx, marker in enumerate(["A", "B", "C", "D"]):
        if marker in text:
            answer_index = idx if answer_index is None else -1
    if answer_index is not None and answer_index >= 0:
        return ["A", "B", "C", "D"][answer_index]
    return text


def _truncated_think(generation):
    """Diagnostic tripwire, NOT part of the original protocol.

    `_strip_thinking` is a silent no-op when `</think>` never appears, so a reasoning trace that
    exhausts `max_gen_toks` mid-`<think>` gets fed to `_extract_predicted_letter` whole. That
    extractor's first tier scans `(A)`->`(B)`->`(C)`->`(D)` and returns the LOWEST letter present
    anywhere in the text, so a truncated ramble almost always scores as a confident "A". Neither
    `acc` nor `na_frac` can see this. A non-zero `truncated_think_frac` means "raise
    `max_gen_toks`", not "the model is bad".

    Assumes the model emits its own opening `<think>` (Qwen3, DeepSeek-R1 chat templates do). A
    template that pre-fills `<think>` into the prompt would leave neither tag in the generation
    and this would read 0.0.
    """
    return 1.0 if ("<think>" in generation and "</think>" not in generation) else 0.0


def _score(doc, generation, answer_prefix):
    text = _strip_thinking(generation)
    text = _strip_answer_prefix(text, answer_prefix)
    pred = _extract_predicted_letter(text)
    gold = str(doc["answerKey"]).strip().upper()
    return {
        # `acc` is byte-faithful to process_answer; the two metrics below are additive only.
        "acc": 1.0 if pred == gold else 0.0,
        # basic_stats (inference/utils.py:160): `na = 1 if predicted not in ['A', 'B'] else 0`,
        # so an extracted "C"/"D" or an unparseable raw-text fallback both count as no-answer.
        "na_frac": 0.0 if pred in ("A", "B") else 1.0,
        "truncated_think_frac": _truncated_think(generation),
    }


def process_results_vanilla(doc, results):
    return _score(doc, results[0], answer_prefix="answer")


def process_results_cot(doc, results):
    return _score(doc, results[0], answer_prefix=".*the answer is")
