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

PAPER PROTOCOL (baked in — for direct comparability to run_baseline.py/evaluate.py):
  * Subset: not all 596 narratives. run_baseline.py's coarse path samples the
    seed-42 shuffle of `list(meta_data.keys())` and takes the first
    num_batch*batch_size = 5*50 = 250 narratives, chunked into 5 batches of 50.
    We replicate that subset+batch grouping exactly (`_subset`), tagging each doc
    with `_batch` (0-4).  See `_subset`.
  * Aggregation: the paper reports mean +/- std of PER-BATCH macro-F1 / accuracy
    over the 5 batches (evaluate.py: per-batch f1_score, then np.mean/np.std) —
    NOT one pooled score.  Our aggregations group by `_batch`, score each batch,
    and return the batch mean (`macro_f1`/`acc`) or batch std (`*_std`).
  * Multihop fullness downsampling: each narrative has 4 fullness + 2 accessibility
    multihop questions; run_baseline.sample_questions drops 2 fullness so the
    combined ("overall") metric is a balanced 2:2 mix.  We keep the first 2 fullness
    per narrative (deterministic; the specific pair is within the paper's own std).
  * Generation: open-model runs stop at ['\n','\n\n'] (run_baseline.py); the
    template's `until` includes both.

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
import random
from pathlib import Path

import datasets
import numpy as np
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


@functools.lru_cache(maxsize=1)
def _read_meta():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks" / "OpenToM" / "data" / "opentom_data" / "meta_data.json"
        if cand.exists():
            return json.load(open(cand, encoding="utf-8"))
    raise FileNotFoundError(
        "meta_data.json not found under any parent's benchmarks/OpenToM/data/opentom_data/"
    )


# Paper protocol constants (run_baseline.py defaults).
_SEED = 42
_NUM_BATCH = 5
_BATCH_SIZE = 50
_FULLNESS_KEEP = 2   # sample_questions drops 2 of the 4 fullness Qs -> keep 2 (balanced 2:2)


@functools.lru_cache(maxsize=1)
def _subset():
    """Replicate run_baseline.py's coarse-path sampling exactly.

    `set_seed(42)` then `random.shuffle(list(meta_data.keys()))`, then the first
    num_batch*batch_size keys chunked into batches of batch_size (sample_entries +
    the while-loop that stops after num_batch batches).  `random.Random(42)` is
    verified byte-identical to the global `random.seed(42); random.shuffle`.

    Returns:
      key2batch:  {narrative_key: batch_idx (0.._NUM_BATCH-1)} for the 250 subset.
      narr2key:   {narrative_text: narrative_key} (bijection over the 596 narratives)
                  — opentom.json rows carry no key, so we map by narrative text.
    """
    meta = _read_meta()
    keys = list(meta.keys())
    random.Random(_SEED).shuffle(keys)
    subset_keys = keys[: _NUM_BATCH * _BATCH_SIZE]
    key2batch = {k: i // _BATCH_SIZE for i, k in enumerate(subset_keys)}

    narr2key = {}
    for k, v in meta.items():
        narr2key.setdefault(v["narrative"], k)
    return key2batch, narr2key


def load(genre=None, order=None, granularity=None, **kwargs):
    """Filter opentom.json to one partition, restrict to the paper's 250-narrative
    seed-42 subset, downsample multihop fullness to 2, and tag each doc with
    `_genre` and `_batch`.

    Partitions mirror the official opentom_data/ split files:
      genre=location, granularity=coarse|fine, order=fo|so  (Loc_coarse / Loc_fine)
      genre=multihop, order=fo|so                           (fullness + accessibility)
      genre=attitude
    Coarse vs fine location is a question-phrasing split, not a re-scoring of the
    same rows: 'locate' questions are fine (place string), the rest are coarse
    ('initial', Yes/No) — exactly build_prompt.py's 'locate'/'initial' branch.
    """
    key2batch, narr2key = _subset()
    type_suffix = {"fo": "-fo", "so": "-so"}.get(order)
    docs = []
    fullness_kept = {}   # narrative_key -> count of fullness Qs kept (downsample to 2)
    for d in _read_opentom():
        narrative = d["narrative"]
        key = narr2key.get(narrative)
        if key is None or key not in key2batch:
            continue                        # not in the paper's 250-narrative subset
        batch = key2batch[key]

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
            if "fullness" in qtext:
                _genre = "fullness"
                # Downsample: keep only the first _FULLNESS_KEEP fullness Qs per
                # narrative (run_baseline.sample_questions drops the other 2).
                c = fullness_kept.get(key, 0)
                if c >= _FULLNESS_KEEP:
                    continue
                fullness_kept[key] = c + 1
            else:
                _genre = "accessibility"
        elif genre == "attitude":
            if t != "attitude":
                continue
            _genre = "attitude"
        else:
            continue

        docs.append({
            "narrative": narrative,
            "question": qtext,
            "answer": str(q["answer"]),
            "_genre": _genre,
            "_batch": batch,
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

    return {"gold": gold, "pred": pred, "genre": genre, "batch": doc["_batch"]}


_BASE_METRICS = [
    "macro_f1", "macro_f1_std", "macro_f1_strict",
    "acc", "acc_std", "acc_strict",
    "corrupt_rate",
]
_MULTIHOP_METRICS = _BASE_METRICS + ["fullness_f1", "accessibility_f1"]


def process_results(doc, results):
    payload = _score(doc, results)
    return {m: payload for m in _BASE_METRICS}


def process_results_multihop(doc, results):
    payload = _score(doc, results)
    return {m: payload for m in _MULTIHOP_METRICS}


# ---------------------------------------------------------------------------
# AGGREGATIONS — PER-BATCH macro-F1 / accuracy, then mean (or std) over batches
# (evaluate.py's protocol: per-batch f1_score, then np.mean/np.std over 5 batches).
# Docs with gold None are always dropped (no ground truth — matches evaluate.py's
# `valid_gt != None` filter). Corrupted preds (-1) are dropped per batch for the
# paper-faithful metrics (`macro_f1`/`acc`); `corrupt_rate` reports the fraction.
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


def _by_batch(items):
    """Group emitted payloads by their `_batch` index (insertion order preserved)."""
    batches = {}
    for it in items:
        batches.setdefault(it["batch"], []).append(it)
    return batches


def _perbatch(items, score_fn):
    """Apply score_fn to each batch's items; return the list of per-batch scores,
    skipping batches with no scorable items (score_fn returns None)."""
    out = []
    for _, bitems in sorted(_by_batch(items).items()):
        s = score_fn(bitems)
        if s is not None:
            out.append(s)
    return out


def _batch_macro_f1(bitems, strict=False):
    golds, preds = _pairs(bitems, strict)
    if not golds:
        return None
    if strict:
        # Average over the true classes only so the wrong-sentinel does not add a
        # phantom class; corrupted preds simply count as misclassifications.
        return f1_score(golds, preds, average="macro", labels=sorted(set(golds)))
    # Paper-faithful: evaluate.py calls f1_score(valid_gt, valid_pred, average='macro').
    return f1_score(golds, preds, average="macro")


def _batch_acc(bitems, strict=False):
    golds, preds = _pairs(bitems, strict)
    if not golds:
        return None
    return accuracy_score(golds, preds)


def _batch_corrupt(bitems):
    if not bitems:
        return None
    # Denominator = all items in the batch (incl. gold=None), matching evaluate.py's
    # `(len(pred_list) - len(valid_pred)) / len(pred_list)`.
    return sum(1 for it in bitems if it["pred"] == _CORRUPTED) / len(bitems)


def _mean(scores):
    return float(np.mean(scores)) if scores else float("nan")


def _std(scores):
    # evaluate.py reports np.std (population, ddof=0), printed as "Variance".
    return float(np.std(scores)) if scores else float("nan")


def agg_macro_f1(items):
    return _mean(_perbatch(items, lambda b: _batch_macro_f1(b, strict=False)))


def agg_macro_f1_std(items):
    return _std(_perbatch(items, lambda b: _batch_macro_f1(b, strict=False)))


def agg_macro_f1_strict(items):
    return _mean(_perbatch(items, lambda b: _batch_macro_f1(b, strict=True)))


def agg_acc(items):
    return _mean(_perbatch(items, lambda b: _batch_acc(b, strict=False)))


def agg_acc_std(items):
    return _std(_perbatch(items, lambda b: _batch_acc(b, strict=False)))


def agg_acc_strict(items):
    return _mean(_perbatch(items, lambda b: _batch_acc(b, strict=True)))


def agg_corrupt_rate(items):
    return _mean(_perbatch(items, _batch_corrupt))


def agg_fullness_f1(items):
    sub = [it for it in items if it["genre"] == "fullness"]
    return _mean(_perbatch(sub, lambda b: _batch_macro_f1(b, strict=False)))


def agg_accessibility_f1(items):
    sub = [it for it in items if it["genre"] == "accessibility"]
    return _mean(_perbatch(sub, lambda b: _batch_macro_f1(b, strict=False)))
