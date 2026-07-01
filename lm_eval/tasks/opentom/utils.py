"""lm-eval adapter for OpenToM (Xu et al., ACL 2024; arXiv:2402.06044).

Faithfulness over convenience: this file **vendors** OpenToM's own prompt
assembly and answer-extraction logic from the read-only submodule under
`benchmarks/OpenToM/` (never imported). Provenance of each vendored block is
noted inline:

  - prompt templates  ← benchmarks/OpenToM/src/prompts/chatgpt_opentom_prompts/*.txt
                         (transcribed verbatim; every {coi}/{eoi}/{poi}/
                         {second_order_statement} replace in build_prompt.py is a
                         no-op because the .txt templates contain only
                         {narrative}/{question} — verified).
  - prompt question mods ← benchmarks/OpenToM/src/utils/build_prompt.py
                           (the vanilla, non-CoT/SimToM/Self-Ask path).
  - answer extractors ← benchmarks/OpenToM/src/evaluate/opentom_evaluator.py
                        (check_*_answer, compute_lexical_overlap, remove_determinant).
  - corpus metric     ← benchmarks/OpenToM/src/evaluate.py
                        (macro-F1 + accuracy over (gold,pred), corruption rate).

Architecture (see outputs/recon/OpenToM.md and outputs/references/OpenToM-build-notes.md):
  doc = ONE (narrative, question) pair  (OpenToM prompts each question separately
        — a per-item protocol → leaf MATRIX, mirroring hitom/dyntom).
  load(genre, order, granularity)  filters opentom.json to one of the 7 official
        opentom_data/ partitions and tags each doc with `_genre`.
  doc_to_text   reconstructs the vanilla build_prompt branch for the doc's genre.
  process_results  dispatches on `_genre` to the vendored extractor, emitting the
        integer (gold, pred) labels as a per-doc payload under every metric name.
  aggregation   (custom !function per metric, the FANToM pattern) consumes the
        whole (gold,pred) list and computes set-level macro-F1 / accuracy.

Two scoring policies are emitted side by side (user choice — "emit both"):
  * `macro_f1` / `acc`  — paper-faithful: corrupted preds (pred == -1) dropped
                          before scoring (evaluate.py behaviour); `corrupt_rate`
                          reports how many were dropped.
  * `macro_f1_strict` / `acc_strict` — stricter: corrupted preds counted as wrong.

Key faithfulness fix (documented in README): plot_info places are read BY KEY,
never via the original's positional `plot_info.values()` unpack. The released
opentom.json key order is (mover, eoi, original_place, move_to_place, observer),
which does NOT match the (mover, affected_char, eoi, original_place, move_to_place)
order the positional unpack assumes — so copying `.values()` would mis-bind
original_place/move_to_place and silently corrupt every fine-location gold.

v1 scope: normal-length opentom.json, vanilla prompting, perspective='all'.
Deferred (documented follow-ups): OpenToM-L, CoT/SimToM/Self-Ask, mover/observer
perspective filtering.
"""

import functools
import json
from pathlib import Path

import datasets
from sklearn.metrics import accuracy_score, f1_score

# Vendored verbatim from load_baseline_model.py (chatgpt_prefix system message).
_SYSTEM = "You are an expert in modeling other's mental state."

# Prompt templates transcribed verbatim from
# src/prompts/chatgpt_opentom_prompts/*.txt (post-strip — build_prompt strips).
_T_LOCATION_CG = (
    "Read and comprehend the following short story. Then, answer the question that follows.\n\n"
    "{narrative}\n\n"
    'Question: {question} Answer the question with "Yes" or "No". Do not give any explanation.'
)
# location_fg / multihop_fullness / multihop_accessibility templates are identical.
_T_NO_EXPLANATION = (
    "Read and comprehend the following short story. Then, answer the question that follows.\n\n"
    "{narrative}\n\n"
    "Question: {question} Answer the question without any explanation."
)
_T_ATTITUDE = (
    "Read and comprehend the following short story. Then, answer the question that follows.\n\n"
    "{narrative}\n\n"
    "Question: {question}"
)

_CORRUPTED = -1            # opentom_evaluator's corrupted-prediction sentinel
_STRICT_WRONG = 99         # never a valid gold in any partition → always counted wrong


# ---------------------------------------------------------------------------
# LOADER — read opentom.json once, filter to one official partition.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _read_opentom():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks" / "OpenToM" / "data" / "opentom.json"
        if cand.exists():
            return json.load(open(cand, encoding="utf-8"))
    raise FileNotFoundError(
        "opentom.json not found under any parent's benchmarks/OpenToM/data/ directory"
    )


def load(genre=None, order=None, granularity=None, **kwargs):
    """Filter opentom.json to one partition and tag each doc with `_genre`.

    Partitions mirror the official opentom_data/ split files:
      genre=location, granularity=coarse|fine, order=fo|so  (Loc_coarse / Loc_fine)
      genre=multihop, order=fo|so                           (fullness + accessibility)
      genre=attitude
    Coarse vs fine location is a question-phrasing split, not a re-scoring of the
    same rows: 'locate' questions are fine (place string), the rest are coarse
    ('initial', Yes/No) — exactly build_prompt.py's 'locate'/'initial' branch.
    """
    type_suffix = {"fo": "-fo", "so": "-so"}.get(order)
    docs = []
    for d in _read_opentom():
        q = d["question"]
        t = q["type"]
        qtext = q["question"]
        pi = d["plot_info"]

        if genre == "location":
            if type_suffix is None or t != "location" + type_suffix:
                continue
            is_fine = "locate" in qtext
            if granularity == "coarse":
                if is_fine:
                    continue
                _genre = "location_cg"
            elif granularity == "fine":
                if not is_fine:
                    continue
                _genre = "location_fg"
            else:
                continue
        elif genre == "multihop":
            if type_suffix is None or t != "multihop" + type_suffix:
                continue
            _genre = "fullness" if "fullness" in qtext else "accessibility"
        elif genre == "attitude":
            if t != "attitude":
                continue
            _genre = "attitude"
        else:
            continue

        docs.append({
            "narrative": d["narrative"],
            "question": qtext,
            "answer": str(q["answer"]),
            "_genre": _genre,
            # BY-KEY access (never plot_info.values()) — see module docstring.
            "original_place": str(pi.get("original_place", "")),
            "move_to_place": str(pi.get("move_to_place", "")),
        })
    return {"train": datasets.Dataset.from_list(docs)}


# ---------------------------------------------------------------------------
# PROMPT — reconstruct the vanilla build_prompt branch for the doc's genre.
# ---------------------------------------------------------------------------

def doc_to_text(doc):
    genre = doc["_genre"]
    narrative = doc["narrative"]
    question = doc["question"]

    if genre == "location_cg":
        body = _T_LOCATION_CG.replace("{narrative}", narrative).replace("{question}", question)
    elif genre == "location_fg":
        body = _T_NO_EXPLANATION.replace("{narrative}", narrative).replace("{question}", question)
    elif genre == "fullness":
        q = f'{question} Answer with "more full", "equally full", or "less full".'
        body = _T_NO_EXPLANATION.replace("{narrative}", narrative).replace("{question}", q)
    elif genre == "accessibility":
        q = f'{question} Answer with "more accessible", "equally accessible", or "less accessible".'
        body = _T_NO_EXPLANATION.replace("{narrative}", narrative).replace("{question}", q)
    elif genre == "attitude":
        q = question.split("?")[0].strip() + ", assuming that you observed the action?"
        q = q + ' Answer with "positive", "neutral", or "negative".'
        body = _T_ATTITUDE.replace("{narrative}", narrative).replace("{question}", q)
        body = body.strip() + " Answer without any explanation."
    else:
        raise ValueError(f"unknown genre: {genre!r}")

    # System message baked in (load_baseline_model.py prepends it as a system turn;
    # we put it in the user turn so it is always present and model-agnostic).
    return _SYSTEM + "\n\n" + body


# ---------------------------------------------------------------------------
# EXTRACTORS — vendored verbatim from opentom_evaluator.py (self → module funcs).
# Each returns [gold_label, pred_label]; pred_label == -1 means corrupted.
# ---------------------------------------------------------------------------

def _remove_determinant(word):
    for det in ("a", "an", "the"):
        if word.startswith(det):
            return word[len(det):].strip()
    return word


def _compute_lexical_overlap(pred, location):
    pred = pred.lower().replace("_", " ").replace("'s", "")
    location = location.lower().replace("_", " ").replace("'s", "")
    score = 0
    pred = pred.replace(".", "").split()
    location = location.split()
    visited_word = []
    for word in pred:
        if word in location and word not in visited_word:
            score += 1
            visited_word.append(word)
    return score / len(location)


def _check_answer_for_fg_location(prediction, answer, original_place, move_to_place):
    answer = _remove_determinant(answer).lower()
    original_place = _remove_determinant(original_place).lower()
    move_to_place = _remove_determinant(move_to_place).lower()

    gt_label, pred_label = None, None
    original_place_score = _compute_lexical_overlap(prediction, original_place)
    move_to_place_score = _compute_lexical_overlap(prediction, move_to_place)

    if original_place_score == move_to_place_score:
        pred_label = 3
    if original_place_score > move_to_place_score:
        pred_label = 1
    elif original_place_score < move_to_place_score:
        pred_label = 2

    if original_place == answer:
        gt_label = 1
    elif move_to_place == answer:
        gt_label = 2

    return [gt_label, pred_label]


def _check_answer_for_cg_location(prediction, answer):
    prediction = prediction.lower()
    answer = answer.lower()
    if "no" in prediction and "yes" not in prediction:
        pred_label = 0
    elif "yes" in prediction and "no" not in prediction:
        pred_label = 1
    else:
        pred_label = -1

    if "no" in answer:
        gt_label = 0
    elif "yes" in answer:
        gt_label = 1
    else:
        gt_label = None
    return [gt_label, pred_label]


def _check_fullness_answer(prediction, answer):
    prediction = prediction.replace(".", "").lower()
    less_full_answer_list = ["less full", "emptier", "more empty"]
    more_full_answer_list = ["more full", "fuller"]

    pred_label, gt_label = None, None
    for less_full_ans in less_full_answer_list:
        if less_full_ans in prediction:
            pred_label = 1
    if not pred_label:
        for more_full_ans in more_full_answer_list:
            if more_full_ans in prediction:
                pred_label = 2
    if not pred_label:
        if "equally full" in prediction:
            pred_label = 3
    if not pred_label:
        pred_label = -1  # corrupted

    if answer == "less full":
        gt_label = 1
    elif answer == "more full":
        gt_label = 2
    elif answer == "equally full":
        gt_label = 3
    return [gt_label, pred_label]


def _check_accessibility_answer(prediction, answer):
    prediction = prediction.replace(".", "").lower()
    if "more accessible" in prediction:
        pred_label = 1
    elif "less accessible" in prediction:
        pred_label = 2
    elif "equally accessible" in prediction:
        pred_label = 3
    else:
        pred_label = -1  # corrupted

    if answer == "more accessible":
        gt_label = 1
    elif answer == "less accessible":
        gt_label = 2
    else:
        gt_label = 3
    return [gt_label, pred_label]


def _check_attitude_answer(prediction, answer):
    prediction = prediction.lower()
    answer = answer.lower()
    answer_map = {"a": "positive", "b": "neutral", "c": "negative"}
    prediction_token = prediction.split("\n\n")[-1].split(":")[-1].split(".")[0].strip().lower()

    gt_label, pred_label = None, None
    if answer == "positive":
        gt_label = 1
    elif answer == "negative":
        gt_label = 2
    else:
        gt_label = 3

    try:
        prediction = answer_map[prediction_token]
        if prediction == "positive":
            pred_label = 1
        elif prediction == "negative":
            pred_label = 2
        else:
            pred_label = 3
    except KeyError:
        if "positive" in prediction_token and "negative" in prediction_token:
            pred_label = -1
        elif "positive" in prediction_token and "neutral" in prediction_token:
            pred_label = -1
        elif "neutral" in prediction_token and "negative" in prediction_token:
            pred_label = -1
        elif "positive" in prediction_token:
            pred_label = 1
        elif "negative" in prediction_token:
            pred_label = 2
        elif "neutral" in prediction_token:
            pred_label = 3
        else:
            pred_label = -1
    return [gt_label, pred_label]


# ---------------------------------------------------------------------------
# process_results — score one QA, emit the (gold,pred) payload under every metric.
# ---------------------------------------------------------------------------

def _score(doc, results):
    genre = doc["_genre"]
    prediction = results[0].strip()
    answer = doc["answer"].strip()

    if genre == "location_cg":
        gold, pred = _check_answer_for_cg_location(prediction, answer)
    elif genre == "location_fg":
        gold, pred = _check_answer_for_fg_location(
            prediction, answer, doc["original_place"], doc["move_to_place"]
        )
    elif genre == "fullness":
        gold, pred = _check_fullness_answer(prediction, answer)
    elif genre == "accessibility":
        # opentom_evaluator collapses '|'-joined accessibility golds to "equally accessible".
        if "|" in answer:
            answer = "equally accessible"
        gold, pred = _check_accessibility_answer(prediction, answer)
    elif genre == "attitude":
        gold, pred = _check_attitude_answer(prediction, answer)
    else:
        raise ValueError(f"unknown genre: {genre!r}")

    return {"gold": gold, "pred": pred, "genre": genre}


_BASE_METRICS = ["macro_f1", "macro_f1_strict", "acc", "acc_strict", "corrupt_rate"]
_MULTIHOP_METRICS = _BASE_METRICS + ["fullness_f1", "accessibility_f1"]


def process_results(doc, results):
    payload = _score(doc, results)
    return {m: payload for m in _BASE_METRICS}


def process_results_multihop(doc, results):
    payload = _score(doc, results)
    return {m: payload for m in _MULTIHOP_METRICS}


# ---------------------------------------------------------------------------
# AGGREGATIONS — set-level macro-F1 / accuracy over the emitted (gold,pred) list
# (the FANToM custom-!function pattern). Docs with gold None are always dropped
# (no ground truth — matches evaluate.py's `valid_gt != None` filter).
# ---------------------------------------------------------------------------

def _pairs(items, strict):
    """Return (golds, preds). Corrupted preds (-1) are dropped (paper-faithful)
    unless strict=True, in which case they are mapped to a sentinel that is
    guaranteed wrong."""
    golds, preds = [], []
    for it in items:
        g, p = it["gold"], it["pred"]
        if g is None:
            continue
        if p == _CORRUPTED:
            if strict:
                golds.append(g)
                preds.append(_STRICT_WRONG)
            # else: drop
        else:
            golds.append(g)
            preds.append(p)
    return golds, preds


def _macro_f1(items, strict=False):
    golds, preds = _pairs(items, strict)
    if not golds:
        return float("nan")
    if strict:
        # Average over the true classes only so the wrong-sentinel does not add a
        # phantom class; corrupted preds simply count as misclassifications.
        return f1_score(golds, preds, average="macro", labels=sorted(set(golds)))
    # Paper-faithful: evaluate.py calls f1_score(valid_gt, valid_pred, average='macro').
    return f1_score(golds, preds, average="macro")


def _acc(items, strict=False):
    golds, preds = _pairs(items, strict)
    if not golds:
        return float("nan")
    return accuracy_score(golds, preds)


def _corrupt_rate(items):
    if not items:
        return float("nan")
    return sum(1 for it in items if it["pred"] == _CORRUPTED) / len(items)


def _subgenre_f1(items, subgenre):
    return _macro_f1([it for it in items if it["genre"] == subgenre], strict=False)


def agg_macro_f1(items):
    return _macro_f1(items, strict=False)


def agg_macro_f1_strict(items):
    return _macro_f1(items, strict=True)


def agg_acc(items):
    return _acc(items, strict=False)


def agg_acc_strict(items):
    return _acc(items, strict=True)


def agg_corrupt_rate(items):
    return _corrupt_rate(items)


def agg_fullness_f1(items):
    return _subgenre_f1(items, "fullness")


def agg_accessibility_f1(items):
    return _subgenre_f1(items, "accessibility")
