"""lm-eval adapter for RecToM (Li, Shi, Deng; arXiv:2511.22275).

RecToM ships static JSON data plus OpenAI-style evaluation scripts under the
read-only benchmark submodule. This adapter vendors the pure prompt/extraction
and exact set-match scoring behavior without importing or editing the benchmark.
"""

import functools
import json
import re
from pathlib import Path

import datasets


_TASK_ORDER = [
    "fine_intent_rec",
    "coarse_intent_rec",
    "belief_rec",
    "fine_intent_seeker",
    "coarse_intent_seeker",
    "desire_seeker",
    "prediction_rec",
    "judgement_rec",
    "prediction_seeker",
    "judgement_seeker",
]

_TASKS = {
    "fine_intent_rec": {
        "file": "1_intent_rec.json",
        "choice_key": "choice",
        "answer_key": "answer_fine",
        "option_count": 10,
        "multi_label": True,
        "label": "Fine Intention (Rec)",
    },
    "coarse_intent_rec": {
        "file": "1_coarse_intent_rec.json",
        "choice_key": "choice",
        "answer_key": "answer_coarse",
        "option_count": 5,
        "multi_label": True,
        "label": "Coarse Intention (Rec)",
    },
    "belief_rec": {
        "file": "8_belief_rec_2_com.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 7,
        "multi_label": False,
        "label": "Belief (Rec)",
    },
    "fine_intent_seeker": {
        "file": "2_intent_seeker.json",
        "choice_key": "choices",
        "answer_key": "answer_fine",
        "option_count": 16,
        "multi_label": True,
        "label": "Fine Intention (Seek)",
    },
    "coarse_intent_seeker": {
        "file": "2_coarse_intent_seeker.json",
        "choice_key": "choices",
        "answer_key": "answer_coarse",
        "option_count": 4,
        "multi_label": True,
        "label": "Coarse Intention (Seek)",
    },
    "desire_seeker": {
        "file": "7_desire_seeker_com.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 2,
        "multi_label": False,
        "label": "Desire (Seek)",
    },
    "prediction_rec": {
        "file": "3_pred_rec.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 5,
        "multi_label": True,
        "label": "Prediction (Rec)",
    },
    "judgement_rec": {
        "file": "5_reverse_judge_rec.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 2,
        "multi_label": False,
        "label": "Judgement (Rec)",
    },
    "prediction_seeker": {
        "file": "4_pred_seeker.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 4,
        "multi_label": True,
        "label": "Prediction (Seek)",
    },
    "judgement_seeker": {
        "file": "6_judge_seeker.json",
        "choice_key": "choices",
        "answer_key": "answer",
        "option_count": 2,
        "multi_label": False,
        "label": "Judgement (Seek)",
    },
}

_COT_SYSTEM = (
    "Here is a movie recommendation dialogue, there are two agents, the RECOMMENDER "
    "and the SEEKER. The RECOMMENDER is trying to recommend movies to SEEKER. Think "
    "step by step to answer the quesiton, but limit yourself to no more than 3 steps."
)

_COT_MULTI_SHOT = """

Ending with "The answer is X",where X is a combination of letters from choices (e.g., AB, ACD).
Do not use any other format for the ending.
Multiple selections are valid and expected when appropriate.
"""

_COT_SINGLE_SHOT = """

            Ending with "The answer is X", where X is one of the option from choices.
            Do not use any other format for the ending.
    """


def _repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "benchmarks" / "RecToM" / "data").exists():
            return parent
    raise FileNotFoundError("benchmarks/RecToM/data not found from task directory")


def _data_dir():
    return _repo_root() / "benchmarks" / "RecToM" / "data"


@functools.lru_cache(maxsize=None)
def _read_json(filename):
    path = _data_dir() / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _letters_for_count(n):
    return [chr(ord("A") + i) for i in range(n)]


def _choice_lines(raw_choices):
    if isinstance(raw_choices, list):
        lines = [str(choice) for choice in raw_choices]
        letters = []
        for line in lines:
            match = re.match(r"\s*([A-Z])\s*:", line)
            if match is None:
                raise ValueError(f"Cannot parse choice letter from {line!r}")
            letters.append(match.group(1))
        return lines, letters
    if isinstance(raw_choices, dict):
        letters = [str(key) for key in raw_choices.keys()]
        lines = [f"{key}: {value}" for key, value in raw_choices.items()]
        return lines, letters
    raise TypeError(f"Unsupported choices object: {type(raw_choices)!r}")


def _normalise_doc(row, task_id, spec):
    lines, letters = _choice_lines(row[spec["choice_key"]])
    expected = _letters_for_count(spec["option_count"])
    if letters != expected:
        raise ValueError(
            f"{task_id}: expected choices {expected}, found {letters} for "
            f"dialogue_id={row.get('dialogue_id')}"
        )

    gold = [str(answer).strip().upper() for answer in row[spec["answer_key"]]]
    bad = sorted(set(gold) - set(letters))
    if bad:
        raise ValueError(
            f"{task_id}: gold letters {bad} outside choices {letters} for "
            f"dialogue_id={row.get('dialogue_id')}"
        )
    gold = sorted(set(gold))

    return {
        "dialogue_id": int(row["dialogue_id"]),
        "utterance_pos": int(row["utterance_pos"]),
        "utterance_context": str(row["utterance_context"]),
        "question": str(row["question"]),
        "choices_text": "\n".join(lines),
        "letters": "".join(letters),
        "answer_range": f"A-{letters[-1]}",
        "gold_letters": gold,
        "answer": "".join(gold),
        "task_id": task_id,
        "task_label": spec["label"],
        "multi_label": bool(spec["multi_label"]),
    }


def load(task_id=None, **kwargs):
    """Load one RecToM paper question family."""
    if task_id not in _TASKS:
        known = ", ".join(_TASK_ORDER)
        raise ValueError(f"Unknown RecToM task_id {task_id!r}; expected one of: {known}")
    spec = _TASKS[task_id]
    docs = [_normalise_doc(row, task_id, spec) for row in _read_json(spec["file"])]
    return {"train": datasets.Dataset.from_list(docs)}


def _direct_system_prompt(doc):
    letters = doc["letters"]
    last = letters[-1]
    if doc["multi_label"]:
        example = "ADE" if len(letters) >= 5 else "ACD"
        return (
            "You are an expert in dialogue analysis. Given a dialogue and a "
            "multiple-choice question, respond ONLY with the letter(s) of the "
            f"correct choice(s) from A-{last}. Do not include any other text, "
            "punctuation, explanation, or whitespace. Example valid outputs: "
            f"'A', 'BC', '{example}'."
        )

    if letters == "AB":
        option_phrase = "A or B"
        examples = "'A', 'B'"
    else:
        option_phrase = f"from A-{last}"
        examples = "'A', 'D'"
    return (
        "You are an expert in dialogue analysis. Given a dialogue and a question, "
        f"respond ONLY with the letter of the correct choice {option_phrase}. "
        "Do not include any other text, punctuation, explanation, or whitespace. "
        f"Example valid outputs: {examples}."
    )


def _base_user_prompt(doc):
    return (
        "\nDiallogue History:\n"
        + doc["utterance_context"]
        + "\nQuestion:\n"
        + doc["question"]
        + "\nChoices:\n"
        + doc["choices_text"]
        + "\nAnswer:"
    )


def doc_to_text(doc):
    return _direct_system_prompt(doc) + "\n" + _base_user_prompt(doc)


def doc_to_text_cot(doc):
    shot = _COT_MULTI_SHOT if doc["multi_label"] else _COT_SINGLE_SHOT
    return (
        _COT_SYSTEM
        + "\n"
        + shot
        + "\nDiallogue History:\n"
        + doc["utterance_context"]
        + "\nQuestion:\n"
        + doc["question"]
        + "\nChoices:\n"
        + doc["choices_text"]
        + "\nAnswer: Let's think step by step."
    )


def _extract_direct(response, valid_letters):
    cleaned = re.sub(r"[^A-Za-z]", "", response)
    candidates = sorted(set(cleaned))
    if any(candidate not in valid_letters for candidate in candidates):
        return []
    return candidates


def _extract_cot(response, answer_range):
    start, end = answer_range.split("-")
    valid_letters = "".join(
        chr(code) for code in range(ord(start.upper()), ord(end.upper()) + 1)
    )
    lead_pattern = r"\b(?:answer\s*(?:is|:)|the\s+answer\s+is)\b"
    answer_block_pattern = r"[^{}]*((?:[{}]|\\boxed\{{[{}]}})+)".format(
        valid_letters, valid_letters, valid_letters
    )
    full_pattern = r"(?i){}{}".format(lead_pattern, answer_block_pattern)
    matches = re.findall(full_pattern, response)

    letters = []
    for match in matches:
        found = re.findall(
            r"\\boxed\{{([{}])}}|([{}])".format(valid_letters, valid_letters),
            match,
            re.IGNORECASE,
        )
        letters.extend(left.upper() or right.upper() for left, right in found)
    return sorted(set(letters))


def _score(doc, pred):
    return {"acc": 1.0 if sorted(doc["gold_letters"]) == sorted(pred) else 0.0}


def process_results(doc, results):
    pred = _extract_direct(results[0], set(doc["letters"]))
    return _score(doc, pred)


def process_results_cot(doc, results):
    pred = _extract_cot(results[0], doc["answer_range"])
    return _score(doc, pred)

