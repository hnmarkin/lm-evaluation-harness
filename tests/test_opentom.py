import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from lm_eval.tasks.opentom import utils as opentom_utils


def test_fullness_selection_matches_original_seeded_call_sequence():
    selection = opentom_utils._fullness_selection()
    key2batch, _ = opentom_utils._subset()
    records = []

    for key in key2batch:
        for order in ("fo", "so"):
            kept = tuple(sorted(selection[(key, f"multihop-{order}")]))
            # Released OpenToM orders its four fullness questions before the two
            # accessibility questions. Include both removed and retained indices
            # in the digest so any RNG-call-order drift is immediately visible.
            removed = tuple(idx for idx in range(4) if idx not in kept)
            records.append(f"{key}|{order}|{removed}|{kept}")

    digest = hashlib.sha256("\n".join(records).encode()).hexdigest()
    assert digest == "3b2d622ec4ad47b36487dea0a87f1504618973d184f43f75e1666acdbf18e52b"


def test_multihop_loaders_reproduce_original_fullness_gold_distributions():
    expected = {
        "fo": Counter({"less full": 196, "more full": 180, "equally full": 124}),
        "so": Counter({"equally full": 288, "less full": 112, "more full": 100}),
    }

    for order in ("fo", "so"):
        docs = opentom_utils.load(genre="multihop", order=order)["train"]
        fullness = [doc for doc in docs if doc["_genre"] == "fullness"]
        accessibility = [doc for doc in docs if doc["_genre"] == "accessibility"]

        assert len(docs) == 1000
        assert len(fullness) == 500
        assert len(accessibility) == 500
        assert Counter(doc["answer"] for doc in fullness) == expected[order]


def test_multihop_selected_questions_match_original_source_sampling():
    root = next(
        parent
        for parent in Path(opentom_utils.__file__).resolve().parents
        if (parent / "benchmarks" / "OpenToM").exists()
    )
    data_dir = root / "benchmarks" / "OpenToM" / "data" / "opentom_data"
    meta = json.loads((data_dir / "meta_data.json").read_text(encoding="utf-8"))
    source = {
        order: json.loads(
            (data_dir / f"multihop_{order}.json").read_text(encoding="utf-8")
        )
        for order in ("fo", "so")
    }

    keys = list(meta)
    random.Random(42).shuffle(keys)
    keys = keys[:250]
    rng = np.random.RandomState(42)
    expected = {"fo": [], "so": []}
    for key in keys:
        for order in ("fo", "so"):
            questions = source[order][key]
            fullness = [
                idx
                for idx, question in enumerate(questions)
                if "fullness" in question["question"]
            ]
            removed = set(rng.choice(fullness, 2, replace=False).tolist())
            expected[order].extend(
                (meta[key]["narrative"], question["question"])
                for idx, question in enumerate(questions)
                if idx not in removed
            )

    for order in ("fo", "so"):
        docs = opentom_utils.load(genre="multihop", order=order)["train"]
        actual = [(doc["narrative"], doc["question"]) for doc in docs]
        assert Counter(actual) == Counter(expected[order])
