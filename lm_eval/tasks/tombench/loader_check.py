"""Loader regression check: the corpus-side invariants `parity_check.py` does NOT cover.

`parity_check.py` proves the *prompt/parse* surface is byte-identical to ToMBench's own
code. It never calls `load()`, `normalize_ability()`, `_story_ids()`, `_normalize_gold()`
or `_report()` -- so the whole *corpus* side of the adapter was, until now, unguarded.
Every number below was derived from the released data and cross-checked against the paper;
they are pinned here so that a later "simplification" of the ability mapping, a bump of the
`benchmarks/ToMBench` submodule, or a change to the arity gate fails loudly instead of
quietly reporting plausible-looking scores.

`benchmarks/` is strictly read-only. Run from the repo root:

    ~/miniconda3/envs/eval_env/python.exe \
        lm-evaluation-harness/lm_eval/tasks/tombench/loader_check.py
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils as U  # noqa: E402


# --- Pinned expectations ----------------------------------------------------

# Per-ability question counts after `normalize_ability` collapses the released data's 34
# raw strings onto the paper's 31 abilities (paper Table 18). The exact-match across 31
# independent buckets summing to 2,860 is what proves the mapping: assigning the compound
# "... Belief: Second-order beliefs" labels to *both* abilities would sum to 2,960, and
# assigning them to Content/Location would leave Second-order empty.
EXPECTED_ABILITY_COUNTS = {
    "Emotion: Typical emotional reactions": 100,
    "Emotion: Atypical emotional reactions": 100,
    "Emotion: Discrepant emotions": 40,
    "Emotion: Mixed emotions": 40,
    "Emotion: Hidden emotions": 80,
    "Emotion: Moral emotions": 40,
    "Emotion: Emotion regulation": 20,
    "Desire: Multiple desires": 20,
    "Desire: Desires influence on actions/emotions": 100,
    "Desire: Desire-action contradiction": 40,
    "Desire: Discrepant desires": 20,
    "Intention: Discrepant intentions": 40,
    "Intention: Prediction of actions": 20,
    "Intention: Intentions explanations": 260,
    "Intention: Completion of failed actions": 20,
    "Knowledge: Knowledge-pretend play links": 30,
    "Knowledge: Percepts-knowledge links": 40,
    "Knowledge: Information-knowledge links": 200,
    "Knowledge: Knowledge-attention links": 20,
    "Belief: Content false beliefs": 200,
    "Belief: Location false beliefs": 200,
    "Belief: Identity false beliefs": 40,
    "Belief: Second-order beliefs": 200,
    "Belief: Beliefs based action/emotions": 142,
    "Belief: Sequence false beliefs": 100,
    "Non-Literal Communication: Irony/Sarcasm": 26,
    "Non-Literal Communication: Egocentric lies": 40,
    "Non-Literal Communication: White lies": 40,
    "Non-Literal Communication: Involuntary lies": 42,
    "Non-Literal Communication: Humor": 40,
    "Non-Literal Communication: Faux pas": 560,
}

# (questions, stories) per task file. Story ids are reconstructed from `INDEX` restarts.
# Matches the paper for 7 of 8; Faux-pas gives 141 vs 140 because the released file lost
# 4 rows (see README "Faithfulness deviations" #6).
EXPECTED_TASK_SHAPE = {
    "Ambiguous Story Task": (200, 100),
    "False Belief Task": (600, 100),
    "Faux-pas Recognition Test": (560, 141),
    "Hinting Task Test": (103, 93),
    "Persuasion Story Task": (100, 100),
    "Scalar Implicature Test": (200, 100),
    "Strange Story Task": (407, 201),
    "Unexpected Outcome Test": (300, 100),
}

N_ITEMS = 2860
N_TASK_ITEMS = 2470          # the 8 task files; the other 12 are ability-only
N_TWO_CHOICE = {"en": 483, "zh": 484}  # differ on the one arity-disagreement row
GOLD_TYPO_ITEM = "Knowledge-Attention Links#10"


def _fail(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --- 1. raw data -> the paper's taxonomy ------------------------------------


def check_taxonomy():
    counts = Counter()
    shapes = {}
    for path in sorted(U._data_dir().glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        for row in rows:
            counts[U.normalize_ability(row[U._K_ABILITY])] += 1
        if path.stem in U.TASK_FILES:
            shapes[path.stem] = (len(rows), len(set(U._story_ids(rows))))

    _fail(sum(counts.values()) == N_ITEMS, f"total rows {sum(counts.values())} != {N_ITEMS}")
    _fail(len(counts) == 31, f"{len(counts)} ability buckets, expected 31")
    _fail(
        dict(counts) == EXPECTED_ABILITY_COUNTS,
        "per-ability counts drifted from paper Table 18:\n"
        + "\n".join(
            f"  {a}: got {counts.get(a)}, want {n}"
            for a, n in EXPECTED_ABILITY_COUNTS.items()
            if counts.get(a) != n
        ),
    )
    _fail(set(counts) == set(U.ABILITIES), "normalized abilities != utils.ABILITIES")
    _fail(
        shapes == EXPECTED_TASK_SHAPE,
        f"task question/story shape drifted: {shapes} != {EXPECTED_TASK_SHAPE}",
    )
    print(f"[1/4] 31 abilities x exact Table-18 counts (sum {N_ITEMS}); 8 task shapes incl. story ids")


# --- 2. loader shape, arity gate, gold recovery -----------------------------


def check_loader():
    for language in ("en", "zh"):
        ds = U.load(language=language, try_times=5)["train"]
        _fail(len(ds) == N_ITEMS * 5, f"{language}: {len(ds)} docs != {N_ITEMS * 5}")

        reps, arity, task_items = defaultdict(list), {}, set()
        for doc in ds:
            reps[doc["item_id"]].append(doc["rep"])
            letter_map = json.loads(doc["opt_map"])
            arity[doc["item_id"]] = sum(1 for v in letter_map.values() if v)
            # Every gold must be reachable through the shuffled->canonical de-map, or the
            # item is unscorable for every model on that repeat.
            _fail(
                doc["gold"] in set(letter_map.values()),
                f"{language}: gold {doc['gold']!r} unreachable in {doc['item_id']} rep {doc['rep']}",
            )
            if doc["task"]:
                task_items.add(doc["item_id"])

        _fail(len(reps) == N_ITEMS, f"{language}: {len(reps)} items != {N_ITEMS}")
        _fail(
            all(sorted(v) == [0, 1, 2, 3, 4] for v in reps.values()),
            f"{language}: some item does not have exactly reps 0..4",
        )
        _fail(len(task_items) == N_TASK_ITEMS, f"{language}: {len(task_items)} task items != {N_TASK_ITEMS}")

        two = sum(1 for n in arity.values() if n == 2)
        _fail(set(arity.values()) <= {2, 4}, f"{language}: arity not in {{2,4}}")
        _fail(two == N_TWO_CHOICE[language], f"{language}: {two} two-choice items != {N_TWO_CHOICE[language]}")

        one = U.load(language=language, try_times=1)["train"]
        _fail(len(one) == N_ITEMS, f"{language}: try_times=1 gave {len(one)} docs != {N_ITEMS}")

    # The upstream gold typo, and the `*_rawgold` variants that reproduce it verbatim.
    fixed = {d["item_id"]: d["gold"] for d in U.load(language="en", try_times=1)["train"]}
    raw = {
        d["item_id"]: d["gold"]
        for d in U.load(language="en", try_times=1, fix_gold_typo=False)["train"]
    }
    _fail(fixed[GOLD_TYPO_ITEM] == "A", f"fix_gold_typo did not repair {GOLD_TYPO_ITEM}")
    _fail(raw[GOLD_TYPO_ITEM] == "A. ", f"rawgold did not preserve the {GOLD_TYPO_ITEM} typo")
    _fail(
        sum(1 for k in fixed if fixed[k] != raw[k]) == 1,
        "fix_gold_typo changed more than the one known row",
    )
    print(f"[2/4] loader: {N_ITEMS}x5 docs/lang, reps 0-4, {N_TASK_ITEMS} task items, "
          f"{N_TWO_CHOICE['en']} en / {N_TWO_CHOICE['zh']} zh two-choice, every gold de-mappable")
    print(f"      gold typo {GOLD_TYPO_ITEM}: 'A. ' -> 'A' (main) / 'A. ' (rawgold), 1 row affected")


# --- 3. metric surface ------------------------------------------------------


def check_metrics():
    yaml_text = (HERE / "_tombench_template_yaml").read_text(encoding="utf-8")
    declared = re.findall(r"\{metric: ([A-Za-z0-9_]+),", yaml_text)
    _fail(
        declared == list(U.METRIC_NAMES),
        "metric_list in _tombench_template_yaml != utils.METRIC_NAMES",
    )
    _fail(len(declared) == 57, f"{len(declared)} metrics, expected 57")

    ds = U.load(language="en", try_times=5)["train"]
    for label, want in (("all-correct", 1.0), ("all-wrong", 0.0)):
        raw = defaultdict(list)
        for doc in ds:
            letter_map = json.loads(doc["opt_map"])
            inverse = {canon: shuffled for shuffled, canon in letter_map.items() if canon}
            if label == "all-correct":
                letter = inverse[doc["gold"]]
            else:
                letter = inverse[next(c for c in inverse if c != doc["gold"])]
            for name, payload in U.process_results(doc, [f"[[{letter}]]"]).items():
                raw[name].append(payload)
        # lm-eval hands each metric its own list object; mirror that exactly.
        _fail(len({id(v) for v in raw.values()}) == 57, "expected 57 distinct payload lists")
        got = {name: getattr(U, "agg_" + name)(raw[name]) for name in U.METRIC_NAMES}
        bad = {k: v for k, v in got.items() if v != want}
        _fail(not bad, f"{label}: metrics != {want}: {bad}")
    print("[3/4] 57 metric names == metric_list; all-correct -> 1.0, all-wrong -> 0.0")


# --- 4. the one zh/en arity disagreement ------------------------------------


def check_arity_disagreement():
    def missing(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    disagree = []
    for path in sorted(U._data_dir().glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                if missing(row["选项C"]) != missing(row["OPTION-C"]):
                    disagree.append(f"{path.stem}#{i}")
                # C and D must vanish together, or `_strip_label` would crash on a NaN.
                _fail(
                    missing(row["选项C"]) == missing(row["选项D"])
                    and missing(row["OPTION-C"]) == missing(row["OPTION-D"]),
                    f"{path.stem}#{i}: OPTION-C/D missingness disagrees",
                )
    _fail(
        disagree == ["Strange Story Task#292"],
        f"zh/en arity disagreements changed: {disagree}",
    )
    print("[4/4] exactly 1 zh/en arity disagreement (Strange Story Task#292); C/D always vanish together")


def main():
    check_taxonomy()
    check_loader()
    check_metrics()
    check_arity_disagreement()
    print("\nLOADER OK")


if __name__ == "__main__":
    main()
