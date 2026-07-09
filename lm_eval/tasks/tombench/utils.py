"""ToMBench (ACL 2024, arXiv:2402.15052) -- lm-eval adapter utilities.

Vendored, with attribution, from `benchmarks/ToMBench/` (read-only; never imported):

  * ``prompts.py``     -> the 2-/4-choice user templates (the four *system* prompts live
                          verbatim in each task YAML's ``description:``).
  * ``run_api.py``     -> ``format_prompt_4`` / ``format_prompt_2``: option de-prefixing
                          (``str.replace``, not lstrip), the per-run option shuffle, and
                          the shuffled->canonical letter map (first-match-wins).
  * ``get_results.py`` -> ``extract_answer`` (the ``[[X]]`` -> ``[X]`` -> last-A-D ladder)
                          and ``most_common_element`` (the majority vote).

Protocol (paper Sec. 4.1): every question is asked ``try_times`` (=5) times with the
answer options shuffled each time; the most frequently selected option, de-mapped back to
its canonical letter, is the final answer. Decoding is greedy (``do_sample=False``), so the
*only* source of variation across the 5 runs is the option order -- and that lives in the
prompt. `repeats: 5` would therefore re-run an identical prompt and vote over 5 identical
answers. The 5 runs are instead materialised as 5 pre-shuffled docs, and the vote happens
in a custom ``aggregation``. That is the faithful implementation, not a workaround.

Scoring hierarchy, verified numerically against the paper's Tables 2/3/21:
  per-task acc     = micro over that task's questions   (8 task files, 2,470 questions)
  task_avg         = MACRO mean of the 8 tasks          (Table 2 "AVG.")
  per-ability acc  = micro over that ability's questions (all 20 files, 2,860 questions)
  category acc     = MACRO mean of that category's abilities
  ability_avg      = MACRO mean of the 6 categories     (Table 3 "AVG.")
  coherent_<task>  = fraction of stories whose questions are ALL correct (Fig. 4/Table 22)
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import datasets

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

def _data_dir():
    """Resolve this repo's own ToMBench submodule copy relative to this file."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks/ToMBench/data"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("benchmarks/ToMBench/data not found under any parent directory")


# Raw column names. The released JSONL mixes Chinese and English keys, and two of
# them embed a literal newline -- access them only through these constants.
_K_ABILITY = "能力\nABILITY"
_K_INDEX = "序号\nINDEX"
_K_ANSWER = "答案\nANSWER"
_ZH = ("故事", "问题", "选项A", "选项B", "选项C", "选项D")
_EN = ("STORY", "QUESTION", "OPTION-A", "OPTION-B", "OPTION-C", "OPTION-D")


# ---------------------------------------------------------------------------
# User prompt templates -- verbatim from benchmarks/ToMBench/prompts.py
# (the system prompts live in each task YAML's `description:`)
# ---------------------------------------------------------------------------

UserEvaluatePrompt4Choices_zh = """[故事]
{story}

[问题]
{question}

[答案选项]
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}"""

UserEvaluatePrompt2Choices_zh = """[故事]
{story}

[问题]
{question}

[答案选项]
A. {choice_a}
B. {choice_b}"""

UserEvaluatePrompt4Choices_en = """[Story]
{story}

[Question]
{question}

[Candidate Answers]
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}"""

UserEvaluatePrompt2Choices_en = """[Story]
{story}

[Question]
{question}

[Candidate Answers]
A. {choice_a}
B. {choice_b}"""


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

# The 8 task-view files (paper Table 2, in its column order) -> paper abbreviation.
# The other 12 files carry the "extra samples" for abilities not covered by any task
# (paper Sec. 3.1); they contribute to the ability view ONLY.
TASK_FILES = {
    "Unexpected Outcome Test": "uot",
    "Scalar Implicature Test": "sit",
    "Persuasion Story Task": "pst",
    "False Belief Task": "fbt",
    "Ambiguous Story Task": "ast",
    "Hinting Task Test": "ht",
    "Strange Story Task": "sst",
    "Faux-pas Recognition Test": "frt",
}
TASK_ABBREVS = list(TASK_FILES.values())

# The 31 ATOMS abilities, grouped into the 6 categories (paper Table 18 order).
CATEGORIES = {
    "Emotion": [
        "Emotion: Typical emotional reactions",
        "Emotion: Atypical emotional reactions",
        "Emotion: Discrepant emotions",
        "Emotion: Mixed emotions",
        "Emotion: Hidden emotions",
        "Emotion: Moral emotions",
        "Emotion: Emotion regulation",
    ],
    "Desire": [
        "Desire: Multiple desires",
        "Desire: Desires influence on actions/emotions",
        "Desire: Desire-action contradiction",
        "Desire: Discrepant desires",
    ],
    "Intention": [
        "Intention: Discrepant intentions",
        "Intention: Prediction of actions",
        "Intention: Intentions explanations",
        "Intention: Completion of failed actions",
    ],
    "Knowledge": [
        "Knowledge: Knowledge-pretend play links",
        "Knowledge: Percepts-knowledge links",
        "Knowledge: Information-knowledge links",
        "Knowledge: Knowledge-attention links",
    ],
    "Belief": [
        "Belief: Content false beliefs",
        "Belief: Location false beliefs",
        "Belief: Identity false beliefs",
        "Belief: Second-order beliefs",
        "Belief: Beliefs based action/emotions",
        "Belief: Sequence false beliefs",
    ],
    "Non-Literal Communication": [
        "Non-Literal Communication: Irony/Sarcasm",
        "Non-Literal Communication: Egocentric lies",
        "Non-Literal Communication: White lies",
        "Non-Literal Communication: Involuntary lies",
        "Non-Literal Communication: Humor",
        "Non-Literal Communication: Faux pas",
    ],
}
ABILITIES = [a for abs_ in CATEGORIES.values() for a in abs_]
_ABILITY_TO_CATEGORY = {a: c for c, abs_ in CATEGORIES.items() for a in abs_}


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


ABILITY_SLUGS = {a: _slug(a) for a in ABILITIES}
CATEGORY_SLUGS = {c: _slug(c) for c in CATEGORIES}


def normalize_ability(raw):
    """Map a raw ABILITY string onto one of the paper's 31 abilities.

    The released data carries 34 distinct raw strings. Three defects separate them from
    the paper's 31, and correcting all three reproduces Table 18's per-ability question
    counts EXACTLY (all 31, summing to 2,860) -- which is what proves this mapping:

      1. compound labels ("Belief: Location false beliefs Belief: Second-order beliefs")
         denote Second-order beliefs alone: the plain Location/Content buckets already
         hold exactly the 200 questions Table 18 assigns them, and the two compound
         groups hold exactly the 200 it assigns Second-order.
      2. "Desires influence on actions" (76) + "Desires influence on emotions (beliefs)"
         (24) are one ability in the paper: "Desires influence on actions/emotions" (100).
      3. whitespace ("Information-knowledge links ") and casing ("Non-literal
         communication:") drift.

    `get_results.py` does none of this and therefore buckets into 34 raw groups.
    """
    a = " ".join(raw.split())
    if "Second-order beliefs" in a:
        return "Belief: Second-order beliefs"
    if a.startswith("Desire: Desires influence on"):
        return "Desire: Desires influence on actions/emotions"
    if a.startswith("Non-literal communication:"):
        a = "Non-Literal Communication:" + a[len("Non-literal communication:"):]
    if a not in _ABILITY_TO_CATEGORY:
        raise ValueError(f"unrecognised ToMBench ability: {raw!r}")
    return a


# ---------------------------------------------------------------------------
# Vendored from benchmarks/ToMBench/get_results.py -- do not "improve"
# ---------------------------------------------------------------------------

def most_common_element(lst):
    """Verbatim from get_results.py. On a tie, `max` returns the first key in dict
    insertion order, i.e. the earliest-seen prediction -- so callers must feed the
    predictions in run order (rep 0..N-1)."""
    element_freq = {}
    for item in lst:
        element_freq[item] = element_freq.get(item, 0) + 1
    most_common = max(element_freq, key=element_freq.get)
    return most_common


def extract_answer(text):
    """Verbatim from get_results.py: [[X]] -> [X] -> last A-D character -> default "A".

    Kept byte-for-byte, including the default-to-"A" on no match. Note the final loop
    walks the text backwards, so on a chain-of-thought response it returns the LAST
    A/B/C/D character anywhere in the trace.
    """
    if "[[A]]" in text:
        return "A"
    elif "[[B]]" in text:
        return "B"
    elif "[[C]]" in text:
        return "C"
    elif "[[D]]" in text:
        return "D"
    elif "[A]" in text:
        return "A"
    elif "[B]" in text:
        return "B"
    elif "[C]" in text:
        return "C"
    elif "[D]" in text:
        return "D"
    else:
        for i in range(len(text) - 1, -1, -1):
            if text[i] == "A":
                return "A"
            elif text[i] == "B":
                return "B"
            elif text[i] == "C":
                return "C"
            elif text[i] == "D":
                return "D"
    return "A"


# ---------------------------------------------------------------------------
# Vendored from benchmarks/ToMBench/run_api.py
# ---------------------------------------------------------------------------

def _missing(v):
    """The released JSONL encodes an absent option as NaN, not null.

    Two distinct source conditions collapse onto NaN via pandas: a genuinely empty cell
    (the 280 two-choice Faux-pas items) and the literal string "None" (one corrupt Strange
    Story row). run_api.py tests `d['选项C'] != None`, which is True for NaN, so it takes
    the 4-choice branch and then crashes on `float.replace`. Treating NaN as absent is the
    code's evident intent and matches the paper ("for true/false questions ... the options
    are simply yes/no").
    """
    return v is None or (isinstance(v, float) and math.isnan(v))


def _strip_label(value, letter):
    """`d['选项A'].replace("A. ", "")` -- a replace-ALL, not a prefix strip.

    Most option values carry no "A. " prefix at all (10,372 of 10,474 English values), and
    752 Chinese values use a space-less "A." that this call leaves untouched. Reproducing
    `str.replace` exactly is what keeps the rendered prompt byte-identical.
    """
    return value.replace(f"{letter}. ", "")


def _shuffle_and_map(canon, rng):
    """Shuffle the options and build the shuffled-letter -> canonical-letter map.

    Mirrors run_api.py's if/elif chain, including its first-match-wins behaviour on
    duplicate option texts (English "Unexpected Outcome Test" rows 165/167 both render
    Angry/Thrilled/Angry/Surprise, so no position ever maps to "C"), and its
    initialisation of every map to {"A":"","B":"","C":"","D":""} -- so a model answering
    "C" or "D" on a two-choice item de-maps to "" and scores wrong rather than erroring.
    """
    shuffled = list(canon)
    rng.shuffle(shuffled)
    letter_map = {"A": "", "B": "", "C": "", "D": ""}
    for pos, letter in enumerate("ABCD"[: len(canon)]):
        for j, canonical in enumerate(canon):
            if shuffled[pos] == canonical:
                letter_map[letter] = "ABCD"[j]
                break
    return shuffled, letter_map


# ---------------------------------------------------------------------------
# Gold + story reconstruction
# ---------------------------------------------------------------------------

def _normalize_gold(raw, fix_gold_typo):
    """`Knowledge-Attention Links` row 10 has ANSWER == 'A. ' (upstream typo, present in
    ToMBench_release_v1_0618.xlsx too). Predictions are always bare letters, so under
    get_results.py that item can never be scored correct. With `fix_gold_typo` (the default
    for the main tasks) we take the leading letter; the `*_rawgold` variants set it False
    to reproduce the original scorer exactly."""
    if raw in ("A", "B", "C", "D"):
        return raw
    if not fix_gold_typo:
        return raw
    for ch in raw:
        if ch in "ABCD":
            return ch
    raise ValueError(f"ungoldable ANSWER: {raw!r}")


def _story_ids(rows):
    """There is no story-id column. INDEX is a question counter that restarts at 1 for each
    new story, so counting restarts recovers the story blocks. This reproduces the paper's
    per-task story counts exactly for 7 of the 8 tasks -- including the irregular ones
    (Hinting 93 = 83 singles + 10 doubles; Strange Story 201 = 197 + 3 + 1).

    Faux-pas yields 141 vs the paper's 140 because the released file lost 4 rows: the
    "Xiao Wang roommate" story is missing its "Does anyone say something inappropriate?"
    question (3 questions), and a "Grandpa dies..." story survives as a single orphaned
    question. 139*4 + 3 + 1 = 560. Effect on `coherent_frt` is under one point.
    """
    ids, sid, prev = [], -1, None
    for row in rows:
        idx = row[_K_INDEX]
        if prev is None or idx <= prev:
            sid += 1
        ids.append(sid)
        prev = idx
    return ids


# ---------------------------------------------------------------------------
# LOADER
# ---------------------------------------------------------------------------

_CACHE = {}


def load(language="en", try_times=5, fix_gold_typo=True, **kwargs):
    """Build one doc per (question x shuffle repeat).

    The 5 repeats are pre-expanded here because the shuffle -- the protocol's only
    stochastic element under greedy decoding -- lives in the prompt. Each repeat draws its
    permutation from a `random.Random` seeded on (file, row, repeat), so a run is
    reproducible on any machine. The original seeds `random` once globally and consumes a
    single stream across files in `os.listdir()` order, which is filesystem-dependent and
    so not portable; the process (5 uniformly random permutations per item, majority vote)
    is identical, only the particular draw differs.
    """
    key = (language, try_times, fix_gold_typo)
    if key in _CACHE:
        return {"train": _CACHE[key]}

    if language not in ("zh", "en"):
        raise ValueError(f"language must be 'zh' or 'en', got {language!r}")
    story_k, question_k, *option_ks = _ZH if language == "zh" else _EN
    tmpl4 = UserEvaluatePrompt4Choices_zh if language == "zh" else UserEvaluatePrompt4Choices_en
    tmpl2 = UserEvaluatePrompt2Choices_zh if language == "zh" else UserEvaluatePrompt2Choices_en

    docs = []
    for path in sorted(_data_dir().glob("*.jsonl")):
        stem = path.stem
        task = TASK_FILES.get(stem, "")
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        stories = _story_ids(rows)

        for row_idx, (row, story) in enumerate(zip(rows, stories)):
            ability = normalize_ability(row[_K_ABILITY])
            gold = _normalize_gold(row[_K_ANSWER], fix_gold_typo)

            # Arity is decided by THIS language's own option-C cell. run_api.py always
            # tests the Chinese column, which disagrees with the English one on exactly
            # one corrupt row (Strange Story #292, whose Chinese options were overwritten
            # with a yes/no pair); gating per-language lets English keep its 4 real options.
            values = [row[k] for k in option_ks]
            n_choices = 2 if _missing(values[2]) else 4
            canon = [_strip_label(v, L) for v, L in zip(values[:n_choices], "ABCD")]
            tmpl = tmpl4 if n_choices == 4 else tmpl2

            for rep in range(try_times):
                rng = random.Random(f"{stem}|{row_idx}|{rep}")
                shuffled, letter_map = _shuffle_and_map(canon, rng)
                fields = dict(story=row[story_k], question=row[question_k])
                for L, choice in zip("abcd", shuffled):
                    fields[f"choice_{L}"] = choice
                docs.append(
                    {
                        "input_text": tmpl.format(**fields),
                        "target": gold,
                        "item_id": f"{stem}#{row_idx}",
                        "story_id": f"{stem}#{story}",
                        "task": task,
                        "ability": ability,
                        "gold": gold,
                        "opt_map": json.dumps(letter_map),
                        "rep": rep,
                    }
                )

    ds = datasets.Dataset.from_list(docs)
    _CACHE[key] = ds
    return {"train": ds}


# ---------------------------------------------------------------------------
# Metric surface
# ---------------------------------------------------------------------------

METRIC_NAMES = (
    ["acc"]
    + [f"task_{t}" for t in TASK_ABBREVS]
    + ["task_avg"]
    + [f"ability_{ABILITY_SLUGS[a]}" for a in ABILITIES]
    + [f"category_{CATEGORY_SLUGS[c]}" for c in CATEGORIES]
    + ["ability_avg"]
    + [f"coherent_{t}" for t in TASK_ABBREVS]
    + ["coherent_avg"]
)
assert len(METRIC_NAMES) == 57, len(METRIC_NAMES)


def process_results(doc, results):
    """EXTRACTOR: parse the letter, de-map it to the canonical letter. The vote itself is
    cross-doc, so emit a payload under every metric name and let the aggregations do it."""
    letter_map = json.loads(doc["opt_map"])
    predicted = letter_map.get(extract_answer(results[0]), "")
    payload = {
        "item_id": doc["item_id"],
        "story_id": doc["story_id"],
        "task": doc["task"],
        "ability": doc["ability"],
        "gold": doc["gold"],
        "rep": doc["rep"],
        "pred": predicted,
    }
    return {name: payload for name in METRIC_NAMES}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


_REPORT_CACHE = {}


def _report(payloads):
    """AGG-FN: majority-vote each item, then roll up the paper's nested hierarchy.

    Memoised on the identity of the payload list -- lm-eval hands the *same* list object to
    all 57 aggregations, so the vote and roll-up run once per task, not 57 times.
    """
    cache_key = (id(payloads), len(payloads))
    if cache_key in _REPORT_CACHE:
        return _REPORT_CACHE[cache_key]

    by_item = defaultdict(list)
    for payload in payloads:
        by_item[payload["item_id"]].append(payload)

    correct, meta = {}, {}
    for item_id, runs in by_item.items():
        runs.sort(key=lambda p: p["rep"])  # tie-break order must be run order
        vote = most_common_element([p["pred"] for p in runs])
        correct[item_id] = 1.0 if vote == runs[0]["gold"] else 0.0
        meta[item_id] = runs[0]

    report = {"acc": _mean(correct.values())}

    # Task view: micro within a task, macro across the 8. Only the 8 task files count.
    task_acc = {
        t: _mean(correct[i] for i in correct if meta[i]["task"] == t) for t in TASK_ABBREVS
    }
    for t in TASK_ABBREVS:
        report[f"task_{t}"] = task_acc[t]
    report["task_avg"] = _mean(v for v in task_acc.values() if not math.isnan(v))

    # Ability view: micro within an ability, macro across a category's abilities,
    # macro across the 6 categories. Spans all 20 files.
    ability_acc = {
        a: _mean(correct[i] for i in correct if meta[i]["ability"] == a) for a in ABILITIES
    }
    for a in ABILITIES:
        report[f"ability_{ABILITY_SLUGS[a]}"] = ability_acc[a]
    category_acc = {}
    for category, members in CATEGORIES.items():
        category_acc[category] = _mean(
            ability_acc[a] for a in members if not math.isnan(ability_acc[a])
        )
        report[f"category_{CATEGORY_SLUGS[category]}"] = category_acc[category]
    report["ability_avg"] = _mean(v for v in category_acc.values() if not math.isnan(v))

    # Coherent test: a story fails if ANY of its questions is wrong. Task view only --
    # the paper defines it over task-oriented performance.
    story_items = defaultdict(list)
    for item_id in correct:
        if meta[item_id]["task"]:
            story_items[(meta[item_id]["task"], meta[item_id]["story_id"])].append(item_id)
    coherent_acc = {}
    for t in TASK_ABBREVS:
        stories = [ids for (task, _), ids in story_items.items() if task == t]
        coherent_acc[t] = _mean(
            1.0 if all(correct[i] == 1.0 for i in ids) else 0.0 for ids in stories
        )
        report[f"coherent_{t}"] = coherent_acc[t]
    report["coherent_avg"] = _mean(v for v in coherent_acc.values() if not math.isnan(v))

    _REPORT_CACHE.clear()
    _REPORT_CACHE[cache_key] = report
    return report


def _make_agg(metric_name):
    def _agg(items):
        return _report(items)[metric_name]

    return _agg


for _name in METRIC_NAMES:
    globals()["agg_" + _name] = _make_agg(_name)
