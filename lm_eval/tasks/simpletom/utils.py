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
"""

import re


_COT_PROMPT = (
    "Think step by step to arrive at an answer. Start your response by explaining your reasoning "
    'process and end your response with "Therefore, the answer is: " followed by '
)

_LETTERS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# PROMPT-FN -- vendored port of inference/utils.py:220-262 (`make_mcq`, `question_prompt`).
# `extra_question`/`add_either_option`/`add_neither_option` are dropped: run_inference.py's
# main() never passes them (all default off), so they are dead parameters for the released
# protocol.
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


def _score(doc, generation, answer_prefix):
    text = _strip_thinking(generation)
    text = _strip_answer_prefix(text, answer_prefix)
    pred = _extract_predicted_letter(text)
    gold = str(doc["answerKey"]).strip().upper()
    return {"acc": 1.0 if pred == gold else 0.0}


def process_results_vanilla(doc, results):
    return _score(doc, results[0], answer_prefix="answer")


def process_results_cot(doc, results):
    return _score(doc, results[0], answer_prefix=".*the answer is")
