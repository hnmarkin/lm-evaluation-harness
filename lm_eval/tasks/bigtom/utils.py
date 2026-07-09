"""BigToM (Understanding Social Reasoning in LLMs with LLMs, arXiv:2306.15448) adapter.

The repo (benchmarks/procedural-evals-tom/, read-only) ships only 200 raw multi-variant
rows (data/bigtom/bigtom.csv) plus an expander script
(code/src/generate_conditions.py) that splices sentences out of each raw story to build
per-condition test items. There is no frozen, shipped set of test items. The functions
below are a **vendored, line-for-line port** of generate_conditions.py's branching logic
(attributed inline), reimplemented as a pure function instead of a CSV-writing script, so
items are materialized in memory instead of under benchmarks/.

Faithfulness notes (see README.md for the full list):
  - Grading reproduces evaluate_conditions.py's own `--mcq` protocol: the model is shown
    both options as a)/b), generates free text, and a correct-key-first substring check
    grades it. The original's rare LLM-judge fallback (only when neither letter substring
    is found) is replaced by a deterministic value-match against the option text; if that
    also fails, the item is bucketed as "error" (see `_grade`), not silently marked wrong.
  - The per-item a)/b) order is reshuffled deterministically with `random.Random(0)` in
    row order, reproducing evaluate_conditions.py's own `random.seed(0)` + sequential
    `random.shuffle(answers)` (that script reseeds to 0 at the start of every condition
    file, so this is exactly what a fresh run of the original would produce).
  - backward_belief's `true_control` and `false_control` cells are byte-identical items
    (both use `actions[1]` and answers[1]/answers[0]) -- this is generate_conditions.py's
    own code, not a bug introduced here; preserved as-is.
  - `backward_desire` exists in generate_conditions.py's branch logic but is never reached
    (the driving `VARIABLES` loop omits it) -- excluded here as dead code.
"""

import csv
import random
from pathlib import Path

import datasets

# Column order of benchmarks/procedural-evals-tom/data/bigtom/bigtom.csv (semicolon-CSV,
# no header, 19 columns; only the first 17 are used -- the trailing "auto";"0" columns are
# generation-provenance metadata, unused at eval time). Verbatim from
# generate_conditions.py's `list_var`.
_LIST_VAR = [
    "Story",
    "Aware of event",
    "Not Aware of event",
    "Action aware",
    "Action not aware",
    "Belief Question",
    "Desire Question",
    "Action Question",
    "Belief Answer Aware",
    "Desire Answer Aware",
    "Action Answer Aware",
    "Belief Answer not Aware",
    "Desire Answer not Aware",
    "Action Answer not Aware",
    "Random Event",
    "Aware of random event",
    "Not aware of random event",
]

# Fixed one-shot exemplar, transcribed verbatim from evaluate_llm.py's `ONE_SHOT` /
# `ONE_SHOT_CHAT` / `ONE_SHOT_CHAT_COT` constants. Always the same item, never sampled
# from bigtom.csv -- used only as the lm-eval fewshot exemplar (see
# `list_fewshot_samples_vanilla` / `list_fewshot_samples_cot`).
_ONE_SHOT = {
    "story": (
        "Kofi is a fisherman from a small village in Ghana. He wants to catch enough "
        "fish today to provide for his family and sell the surplus at the market. Kofi "
        "repaired his fishing net last night. While Kofi is away from his boat, a group "
        "of monkeys comes and plays with the fishing net, tearing it apart. Kofi does "
        "not see the monkeys damaging his fishing net."
    ),
    "question": (
        "Does Kofi believe his fishing net is in good condition or torn apart?\n"
        "Choose one of the following:\n"
        "a)Kofi believes his fishing net is in good condition.\n"
        "b)Kofi believes his fishing net is torn apart."
    ),
    "answer": "a)Kofi believes his fishing net is in good condition.",
    "thought": (
        "Let's think step by step:\n"
        "1) Kofi repaired his fishing net last night. So last night he believes that "
        "his net is fixed.\n"
        "2) While Kofi is away from his boat, a group of monkeys comes and plays with "
        "the fishing net, tearing it apart.\n"
        "3) Kofi does not see the monkeys damaging his fishing net. So, his belief "
        "about his net stays the same. He thinks that it is fixed.\n"
        "4) Does Kofi believe his fishing net is in good condition or torn apart?\n"
        "5) Kofi believes his fishing net is in good condition."
    ),
}

_raw_rows_cache = None


def _data_file():
    # Resolve THIS repo's own procedural-evals-tom submodule copy, relative to this
    # file, so the task is portable across clones/machines (no hardcoded absolute path).
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks" / "procedural-evals-tom" / "data" / "bigtom" / "bigtom.csv"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "bigtom.csv not found under any parent's benchmarks/ directory"
    )


def _raw_rows():
    """Cache-once parse of the 200-row raw CSV (read-only benchmarks/ source)."""
    global _raw_rows_cache
    if _raw_rows_cache is None:
        with open(_data_file(), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)
        _raw_rows_cache = [
            {_LIST_VAR[i]: row[i] for i in range(len(_LIST_VAR))} for row in rows
        ]
    return _raw_rows_cache


# ---------------------------------------------------------------------------
# RESHAPER: vendored port of generate_conditions.py's per-variable / per-condition
# sentence-splice expansion. Each raw story is `.split(".")`-indexed at fixed
# positions [0..4] (5 sentences: setup, want, percept-action, belief-state, event) --
# this brittle, position-dependent construction IS the benchmark; ported verbatim.
# ---------------------------------------------------------------------------


def _forward_common(d, init_belief, question_key, answer_keys):
    """Shared construction for forward_belief / forward_action (identical in the
    original apart from which question/answer columns are read)."""
    question = d[question_key]
    answers = [d[answer_keys[0]], d[answer_keys[1]]]
    awareness = [d["Aware of event"], d["Not Aware of event"]]
    awareness_random = [d["Aware of random event"], d["Not aware of random event"]]
    parts = d["Story"].split(".")
    if init_belief == 0:
        story = parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[4] + "."
        story_control = ".".join(parts[:3] + [" " + d["Random Event"]])
    else:
        story = (
            parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[3] + "." + parts[4] + "."
        )
        story_control = ".".join(parts[:4] + [" " + d["Random Event"]])
    return question, answers, awareness, awareness_random, story, story_control


def _backward_belief(d, init_belief):
    question = d["Belief Question"]
    answers = [d["Belief Answer Aware"], d["Belief Answer not Aware"]]
    awareness = [d["Aware of event"], d["Not Aware of event"]]
    actions = [d["Action aware"], d["Action not aware"]]
    awareness_random = [d["Aware of random event"], d["Not aware of random event"]]
    parts = d["Story"].split(".")
    if init_belief == 0:
        story = parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[4] + "."
        # generate_conditions.py re-splits `story` (not the original `parts`) here;
        # kept verbatim even though the result is equivalent, in case any story's
        # spliced sentence set ever contains an internal period.
        story_parts = story.split(".")
        story_control = ".".join(story_parts[:3] + [" " + d["Random Event"]])
    else:
        story = (
            parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[3] + "." + parts[4] + "."
        )
        story_parts = story.split(".")
        story_control = ".".join(story_parts[:4] + [" " + d["Random Event"]])
    return question, answers, awareness, actions, awareness_random, story, story_control


def _apply_condition(variable, condition, story, story_control, answers, awareness,
                      awareness_random, actions=None):
    if condition == "true_belief":
        if variable == "backward_belief":
            final_story = f"{story} {actions[0]}"
        else:
            final_story = f"{story} {awareness[0]}"
        true_ans, wrong_ans = answers[0], answers[1]
    elif condition == "false_belief":
        if variable == "backward_belief":
            final_story = f"{story} {actions[1]}"
        else:
            final_story = f"{story} {awareness[1]}"
        true_ans, wrong_ans = answers[1], answers[0]
    elif condition == "true_control":
        # backward_belief: both true_control and false_control use actions[1] --
        # byte-identical items, inherited from generate_conditions.py itself (see
        # module docstring); not corrected here.
        if variable == "backward_belief":
            final_story = f"{story_control} {actions[1]}"
        else:
            final_story = f"{story_control} {awareness_random[0]}"
        true_ans, wrong_ans = answers[1], answers[0]
    elif condition == "false_control":
        if variable == "backward_belief":
            final_story = f"{story_control} {actions[1]}"
        else:
            final_story = f"{story_control} {awareness_random[1]}"
        true_ans, wrong_ans = answers[1], answers[0]
    else:
        raise ValueError(f"unsupported condition {condition!r}")
    return final_story, true_ans, wrong_ans


def _percept_to_belief(d, init_belief):
    question = d["Belief Question"]
    answers = [d["Belief Answer Aware"], d["Belief Answer not Aware"]]
    parts = d["Story"].split(".")
    if init_belief == 1:
        story = parts[0] + "." + parts[1] + "." + parts[2] + "."
    else:
        story = d["Story"]
    return story, question, answers[0], answers[1]


_FORWARD_KEYS = {
    "forward_belief": ("Belief Question", ("Belief Answer Aware", "Belief Answer not Aware")),
    "forward_action": ("Action Question", ("Action Answer Aware", "Action Answer not Aware")),
}

_CORE_VARIABLES = ("forward_belief", "forward_action", "backward_belief")
_CONDITIONS = ("true_belief", "false_belief", "true_control", "false_control")

_PAIR_KIND_TO_CONDITIONS = {
    "belief": ("true_belief", "false_belief"),
    "control": ("true_control", "false_control"),
}


def _pair_kind_for_condition(condition):
    if condition in ("true_belief", "false_belief"):
        return "belief"
    if condition in ("true_control", "false_control"):
        return "control"
    raise ValueError(f"unsupported condition {condition!r}")


def _pair_side_for_condition(condition):
    if condition in ("true_belief", "true_control"):
        return "true"
    if condition in ("false_belief", "false_control"):
        return "false"
    raise ValueError(f"unsupported condition {condition!r}")


def _expand_row(d, variable, init_belief, condition):
    if variable == "percept_to_belief":
        if init_belief != 1 or condition != "true_belief":
            raise ValueError(
                "percept_to_belief is only defined for init_belief=1, condition=true_belief "
                "(generate_conditions.py's own write-gate)"
            )
        story, question, true_ans, wrong_ans = _percept_to_belief(d, init_belief)
        return story, question, true_ans, wrong_ans
    if variable in _FORWARD_KEYS:
        qkey, akeys = _FORWARD_KEYS[variable]
        question, answers, awareness, awareness_random, story, story_control = _forward_common(
            d, init_belief, qkey, akeys
        )
        final_story, true_ans, wrong_ans = _apply_condition(
            variable, condition, story, story_control, answers, awareness, awareness_random
        )
        return final_story, question, true_ans, wrong_ans
    if variable == "backward_belief":
        question, answers, awareness, actions, awareness_random, story, story_control = (
            _backward_belief(d, init_belief)
        )
        final_story, true_ans, wrong_ans = _apply_condition(
            variable, condition, story, story_control, answers, awareness, awareness_random,
            actions=actions,
        )
        return final_story, question, true_ans, wrong_ans
    raise ValueError(f"unsupported variable {variable!r}")


# ---------------------------------------------------------------------------
# LOADER
# ---------------------------------------------------------------------------


def _materialize_cell(variable, init_belief, condition):
    init_belief = int(init_belief)
    rows = _raw_rows()
    rng = random.Random(0)
    docs = []
    pair_kind = _pair_kind_for_condition(condition)
    for idx, d in enumerate(rows):
        story, question, true_ans, wrong_ans = _expand_row(d, variable, init_belief, condition)
        pair = [true_ans, wrong_ans]
        rng.shuffle(pair)
        gold_letter = "a" if pair[0] == true_ans else "b"
        other_letter = "b" if gold_letter == "a" else "a"
        docs.append(
            {
                "id": f"{variable}_{init_belief}_{condition}_{idx}",
                "raw_idx": idx,
                "variable": variable,
                "init_belief": init_belief,
                "condition": condition,
                "pair_kind": pair_kind,
                "story": story,
                # Verbatim format from evaluate_conditions.py: "a)<opt>" / "b)<opt>",
                # no space after the paren.
                "question_with_choices": (
                    f"{question}\nChoose one of the following:\na){pair[0]}\nb){pair[1]}"
                ),
                "answer_target": f"Answer: {gold_letter}){true_ans}",
                "true_text": true_ans,
                "wrong_text": wrong_ans,
                "gold_letter": gold_letter,
                "other_letter": other_letter,
            }
        )
    return docs


def load(variable=None, init_belief=None, condition=None, **kwargs):
    """Materialize one (variable, init_belief, condition) cell (200 items) in memory.

    Reproduces evaluate_conditions.py's per-cell deterministic answer-order shuffle:
    that script does `random.seed(0)` once at import time and then
    `random.shuffle(answers)` per row in sequential row order; since each condition
    file is processed by its own process invocation, a fresh `random.Random(0)`
    shuffled in row order reproduces it exactly for a full 200-row run.
    """
    docs = _materialize_cell(variable, init_belief, condition)
    return {"train": datasets.Dataset.from_list(docs)}


def load_paired(variable=None, init_belief=None, pair_kind=None, **kwargs):
    """Materialize a paired true/false belief or control cell for `tb_fb_acc`.

    The original paper's `TB & FB` contingency is a cross-item metric: a false-side
    item is credited only when the raw-row-matched true-side item is also correct.
    lm-eval cannot compute that from separate leaves, so paired leaves contain both
    sides with stable `raw_idx` metadata and aggregate them inside one task.
    """
    if pair_kind not in _PAIR_KIND_TO_CONDITIONS:
        raise ValueError(f"unsupported pair_kind {pair_kind!r}")

    true_condition, false_condition = _PAIR_KIND_TO_CONDITIONS[pair_kind]
    true_docs = _materialize_cell(variable, init_belief, true_condition)
    false_docs = _materialize_cell(variable, init_belief, false_condition)
    if len(true_docs) != len(false_docs):
        raise ValueError(
            f"paired cells differ in length: {true_condition}={len(true_docs)}, "
            f"{false_condition}={len(false_docs)}"
        )

    docs = []
    for true_doc, false_doc in zip(true_docs, false_docs, strict=True):
        if true_doc["raw_idx"] != false_doc["raw_idx"]:
            raise ValueError(
                f"paired cells are not aligned at raw_idx {true_doc['raw_idx']} / "
                f"{false_doc['raw_idx']}"
            )
        docs.extend([true_doc, false_doc])
    return {"train": datasets.Dataset.from_list(docs)}


def load_variable(variable=None, **kwargs):
    """Materialize all core BigToM cells for one variable.

    This keeps the registered task surface small (`variant x variable`) while
    preserving the old per-cell granularity through metric names emitted from
    `process_results_variable_*`.
    """
    if variable not in _CORE_VARIABLES:
        raise ValueError(f"unsupported core variable {variable!r}")

    docs = []
    for init_belief in (0, 1):
        for condition in _CONDITIONS:
            docs.extend(_materialize_cell(variable, init_belief, condition))
    return {"train": datasets.Dataset.from_list(docs)}


# ---------------------------------------------------------------------------
# Fixed fewshot exemplar (lm-eval's native list_fewshot_samples mechanism; see
# fewshot_config in the 1shot templates). Rendered through the SAME doc_to_text /
# doc_to_target as real docs, so it must carry the same fields.
# ---------------------------------------------------------------------------


def list_fewshot_samples_vanilla():
    return [
        {
            "story": _ONE_SHOT["story"],
            "question_with_choices": _ONE_SHOT["question"],
            "answer_target": f"Answer: {_ONE_SHOT['answer']}",
        }
    ]


def list_fewshot_samples_cot():
    return [
        {
            "story": _ONE_SHOT["story"],
            "question_with_choices": _ONE_SHOT["question"],
            "answer_target": f"Thought: {_ONE_SHOT['thought']}\nAnswer: {_ONE_SHOT['answer']}",
        }
    ]


# ---------------------------------------------------------------------------
# EXTRACTOR / METRIC-FN
# ---------------------------------------------------------------------------


def _strip_thinking(text):
    """Reasoning-capable chat models (Qwen3, DeepSeek-R1, ...) prepend a <think>
    block by default -- something that did not exist when BigToM's own grading
    code was written, and the task instructions have no way to disable it (that's
    a model-level chat-template kwarg, not a task setting). Grade only the text
    after the LAST </think>, so mid-thought mentions of both options (or a
    truncated think block that never got to a real answer) can't spuriously
    satisfy the correct-key-first substring check below."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx != -1 else text


def _grade(check_text, gold_letter, other_letter, true_text, wrong_text):
    """Correct-key-first substring check, vendored from evaluate_conditions.py:
    `if answer_key in predicted_answer_parsed.lower(): True elif negative_answer_key
    in ...: False else: <LLM judge>`. The LLM-judge fallback is replaced here by a
    deterministic value-match against the option text (hitom/DynToM pattern); if
    neither the letter nor the value is found, the item is an "error" -- kept
    distinct from a graded-wrong item rather than silently folded into it."""
    low = check_text.lower()
    if f"{gold_letter})" in low:
        return "correct"
    if f"{other_letter})" in low:
        return "incorrect"
    if true_text.lower() in low:
        return "correct"
    if wrong_text.lower() in low:
        return "incorrect"
    return "error"


def _to_metrics(grade):
    return {"acc": 1.0 if grade == "correct" else 0.0, "error_rate": 1.0 if grade == "error" else 0.0}


def _cell_suffix(doc):
    return f"{int(doc['init_belief'])}_{doc['condition']}"


def _score_vanilla(doc, results):
    text = _strip_thinking(results[0])
    return _grade(text, doc["gold_letter"], doc["other_letter"], doc["true_text"], doc["wrong_text"])


def process_results_vanilla(doc, results):
    return _to_metrics(_score_vanilla(doc, results))


def _parse_chat_response(response):
    """Vendored verbatim from evaluate_llm.py:parse_chat_response, INCLUDING its
    off-by-one quirk when "Answer:" is absent: `str.find` returns -1, so this then
    slices at index 7 (`-1 + 8`) instead of returning the empty string. Not fixed;
    faithfully reproduced."""
    answer_idx = response.find("Answer:")
    return response[answer_idx + 8:].strip()


def _score_cot(doc, results):
    parsed = _parse_chat_response(_strip_thinking(results[0]))
    return _grade(parsed, doc["gold_letter"], doc["other_letter"], doc["true_text"], doc["wrong_text"])


def process_results_cot(doc, results):
    return _to_metrics(_score_cot(doc, results))


def _tb_fb_payload(doc, grade):
    return {
        "raw_idx": int(doc["raw_idx"]),
        "variable": doc["variable"],
        "init_belief": int(doc["init_belief"]),
        "pair_kind": doc["pair_kind"],
        "condition": doc["condition"],
        "acc": 1.0 if grade == "correct" else 0.0,
    }


def _to_paired_metrics(doc, grade):
    metrics = _to_metrics(grade)
    metrics["tb_fb_acc"] = _tb_fb_payload(doc, grade)
    return metrics


def process_results_paired_vanilla(doc, results):
    return _to_paired_metrics(doc, _score_vanilla(doc, results))


def process_results_paired_cot(doc, results):
    return _to_paired_metrics(doc, _score_cot(doc, results))


def _to_variable_metrics(doc, grade):
    base = _to_metrics(grade)
    suffix = _cell_suffix(doc)
    base[f"acc_{suffix}"] = base["acc"]
    base[f"error_rate_{suffix}"] = base["error_rate"]
    base[f"tb_fb_acc_{int(doc['init_belief'])}_{doc['pair_kind']}"] = _tb_fb_payload(doc, grade)
    return base


def process_results_variable_vanilla(doc, results):
    return _to_variable_metrics(doc, _score_vanilla(doc, results))


def process_results_variable_cot(doc, results):
    return _to_variable_metrics(doc, _score_cot(doc, results))


def agg_tb_fb_acc(items):
    """Aggregate BigToM's paper metric: P(true-side correct AND false-side correct)."""
    pairs = {}
    for item in items:
        key = (
            item["variable"],
            int(item["init_belief"]),
            item["pair_kind"],
            int(item["raw_idx"]),
        )
        side = _pair_side_for_condition(item["condition"])
        pairs.setdefault(key, {})[side] = float(item["acc"])

    scores = [
        1.0 if pair["true"] == 1.0 and pair["false"] == 1.0 else 0.0
        for pair in pairs.values()
        if "true" in pair and "false" in pair
    ]
    return sum(scores) / len(scores) if scores else float("nan")
