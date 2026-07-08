"""lm-eval adapter for OmniToM (Bawatneh et al., 2026; arXiv:2605.26322).

OmniToM reconstructs the full multi-actor belief structure of a story in two
linked generative stages, evaluated **zero-shot under TELeR Level-3 prompts**:

  * Stage 1 (Extraction) -- given a story, emit all (Actor, Belief, Order) tuples
    as a pipe table.  Scored by a **live GPT-5 semantic judge** (precision / recall /
    F1 over matched propositions, macro-averaged over stories).
  * Stage 2 (Labeling) -- given a story and the benchmark belief table, assign a
    7-dimensional schema label vector to each belief.  Scored by **per-dimension
    exact-match accuracy**, macro-averaged over stories.

Faithfulness over convenience.  This file **vendors** OmniToM's own prompt text and
its *pure, deterministic* Stage-2 scorer from the read-only submodule
`benchmarks/omnitom-benchmark/` (never imported, never edited).  Provenance is noted
inline at each vendored block:

  - Stage-1 prompt       <- prompts_extract.py       (PROMPT/OUTRO, build_extract_messages)
  - Stage-2 prompt       <- prompts_label.py         (PROMPT/OUTRO, build_label_messages)
  - belief-table render  <- benchmark_prompting.py   (actor_belief_rows, belief_table_pipe)
  - gold recovery        <- run_replication.py       (ground_truth_{label,belief}_rows)
  - table parsing        <- run_replication.py       (parse_pipe_table/{label,extraction}_rows,
                                                      canonicalize_label_*, LABEL_VALUE_ALIASES)
  - Stage-2 metric       <- run_replication.py       (pair_indexed_rows, compute_stage2_metrics)
  - Stage-1 proxy        <- run_replication.py       (exact_matchcount_rows, coerce_matchcount)

ARCHITECTURE (see outputs/recon/omnitom-benchmark.md + this task's README):
  doc  = ONE story (batched container -- the whole story, all its beliefs, in one
         prompt; many rows back).  895 stories, ids 1..895.
  load(category=...)  reads benchmark_story_belief_labels.jsonl and (optionally)
         filters to one of the 7 ToMBench story categories for the by-category leaves.
  doc_to_text_label / doc_to_text_extract  reconstruct the stage's TELeR-L3 *user*
         prompt byte-for-byte (the SYSTEM prompt lives in each task's `description`
         field and is delivered as a real system role under --apply_chat_template; see
         the README).
  process_results_label   parses the model's label table and emits 8 per-story
         scalars (overall_acc + acc_<dim> x7); each YAML metric means them over
         stories (== the repo's safe_mean-over-stories == macro over stories).
  process_results_extract emits only clearly-named DIAGNOSTIC proxies -- the faithful
         Stage-1 F1 needs the GPT-5 judge and is produced offline by
         score_stage1_offline.py.

WHY STAGE 1 IS OFFLINE: its scorer is a generative LLM-as-judge (GPT-5) in the loop,
which cannot run inside lm-eval.  Stage 1 runs generate_until + --predict_only
--log_samples; score_stage1_offline.py feeds the generations through the repo's OWN
run_replication.py judge+metrics so the canonical judge is preserved byte-for-byte.
"""

import functools
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import datasets

# ---------------------------------------------------------------------------
# Constants vendored verbatim from run_replication.py
# ---------------------------------------------------------------------------

EXTRACTION_COLUMNS = ["actor", "belief", "order"]

# The seven schema dimensions scored in Stage 2 (run_replication.py:58).
LABEL_DIMENSIONS = [
    "order",
    "truth-status",
    "knowledge-access",
    "representation",
    "content_type",
    "mental-source",
    "context",
]

# Map each hyphenated dimension to a safe lm-eval metric name (no hyphens).
_DIM_TO_METRIC = {
    "order": "acc_order",
    "truth-status": "acc_truth_status",
    "knowledge-access": "acc_knowledge_access",
    "representation": "acc_representation",
    "content_type": "acc_content_type",
    "mental-source": "acc_mental_source",
    "context": "acc_context",
}

# JSONL label key -> output/dimension key (run_replication.py:68).
JSONL_LABEL_KEY_TO_OUTPUT = {
    "order": "order",
    "truth_status": "truth-status",
    "knowledge_access": "knowledge-access",
    "representation": "representation",
    "content_type": "content_type",
    "mental_source": "mental-source",
    "context": "context",
}

# Value canonicalization tables (run_replication.py:95).
LABEL_VALUE_ALIASES = {
    "truth-status": {
        "true": "True",
        "false": "False",
        "unknown": "Unknown",
    },
    "knowledge-access": {
        "private": "Private",
        "shared": "Shared",
        "public": "Public",
        "unknown": "Unknown",
    },
    "representation": {
        "explicit": "Explicit",
        "implicit": "Implicit",
    },
    "content_type": {
        "location": "Location",
        "contents/physicalstate": "Contents/Physical State",
        "contentsphysicalstate": "Contents/Physical State",
        "contents / physical state": "Contents/Physical State",
        "physicalstatecontents": "Contents/Physical State",
        "physical state": "Contents/Physical State",
        "identity/relation": "Identity/Relation",
        "identityrelation": "Identity/Relation",
        "relation": "Identity/Relation",
        "identity": "Identity/Relation",
        "epistemic": "Epistemic",
        "desire/intention": "Desire/Intention",
        "desireintention": "Desire/Intention",
        "emotion": "Emotion",
        "trait/value": "Trait/Value",
        "traitvalue": "Trait/Value",
        "action/event": "Action/Event",
        "actionevent": "Action/Event",
        "action": "Action/Event",
        "event": "Action/Event",
    },
    "mental-source": {
        "narration": "Narration",
        "perception": "Perception",
        "memory": "Memory",
        "testimony": "Testimony",
        "inference": "Inference",
        "imagination": "Imagination",
        "unknown": "Unknown",
    },
    "context": {
        "deceptive": "Deceptive",
        "temporal": "Temporal",
        "counterfactual": "Counterfactual",
        "neutral": "Neutral",
    },
}

# ---------------------------------------------------------------------------
# Pure helpers vendored verbatim from run_replication.py (deterministic, stdlib)
# ---------------------------------------------------------------------------

import re  # noqa: E402  (kept next to the vendored regex helpers)


def normalize_space(value):
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value):
    text = normalize_space(value).lower()
    text = text.strip("\"'")
    return text


def safe_mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(mean(values))


def parse_pipe_table(text):
    raw_lines = (text or "").splitlines()
    if not raw_lines:
        return []

    def split_row(line):
        cleaned = line.strip()
        if cleaned.startswith("|"):
            cleaned = cleaned[1:]
        if cleaned.endswith("|"):
            cleaned = cleaned[:-1]
        return [cell.strip() for cell in cleaned.split("|")]

    def is_separator(cells):
        if not cells:
            return False
        return all((not cell) or bool(re.fullmatch(r":?-{3,}:?", cell)) for cell in cells)

    rows = []
    for line in raw_lines:
        if "|" not in line:
            continue
        cells = split_row(line)
        if is_separator(cells):
            continue
        rows.append(cells)
    return rows


def canonicalize_label_value(column, value):
    text = normalize_space(value)
    if not text:
        return ""

    if column == "order":
        match = re.search(r"(\d+)", text)
        return match.group(1) if match else text

    lookup = LABEL_VALUE_ALIASES.get(column, {})
    key = re.sub(r"[^a-z0-9/ ]+", "", text.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in lookup:
        return lookup[key]

    squashed = key.replace(" ", "")
    if squashed in lookup:
        return lookup[squashed]

    return text


def canonicalize_label_row(row):
    out = {
        "actor": normalize_space(row.get("actor", "")),
        "belief": normalize_space(row.get("belief", "")),
    }
    for column in LABEL_DIMENSIONS:
        out[column] = canonicalize_label_value(column, row.get(column, ""))
    return out


def parse_extraction_rows(text):
    rows = parse_pipe_table(text)
    if not rows:
        return []
    parsed = []
    for index, cells in enumerate(rows):
        if index == 0 and len(cells) >= 3 and normalize_key(cells[0]) == "actor":
            continue
        if len(cells) < 3:
            continue
        parsed.append(
            {
                "actor": normalize_space(cells[0]),
                "belief": normalize_space(cells[1]).rstrip("."),
                "order": normalize_space(cells[2]),
            }
        )
    return parsed


def parse_label_rows(text):
    rows = parse_pipe_table(text)
    if not rows:
        return []
    parsed = []
    for index, cells in enumerate(rows):
        if index == 0 and len(cells) >= 2 and normalize_key(cells[0]) == "actor":
            continue
        if len(cells) < 9:
            continue
        parsed.append(
            canonicalize_label_row(
                {
                    "actor": cells[0],
                    "belief": normalize_space(cells[1]).rstrip("."),
                    "order": cells[2],
                    "truth-status": cells[3],
                    "knowledge-access": cells[4],
                    "representation": cells[5],
                    "content_type": cells[6],
                    "mental-source": cells[7],
                    "context": cells[8],
                }
            )
        )
    return parsed


def pair_indexed_rows(rows):
    """Key rows by (norm(actor), norm(belief), occurrence-index) (run_replication.py:1080)."""
    seen = Counter()
    indexed = {}
    for row in rows:
        key_base = (normalize_key(row.get("actor", "")), normalize_key(row.get("belief", "")))
        pair_idx = seen[key_base]
        seen[key_base] += 1
        indexed[(key_base[0], key_base[1], pair_idx)] = row
    return indexed


def coerce_matchcount(value):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def exact_matchcount_rows(left_rows, right_rows):
    """Exact-string (per-actor) match counts (run_replication.py:521, the mock backend's
    matcher).  Used ONLY for the Stage-1 diagnostic proxy -- a strict lower bound on the
    GPT-5 judge's semantic matching, NOT the paper metric."""
    right_counter = Counter(
        (normalize_key(r.get("actor", "")), normalize_key(r.get("belief", ""))) for r in right_rows
    )
    output = []
    for row in left_rows:
        key = (normalize_key(row.get("actor", "")), normalize_key(row.get("belief", "")))
        output.append(
            {
                "Actor": normalize_space(row.get("actor", "")),
                "Belief": normalize_space(row.get("belief", "")),
                "MatchCount": str(right_counter.get(key, 0)),
            }
        )
    return output


# ---------------------------------------------------------------------------
# Gold recovery (run_replication.py:202/215) + belief-table render
# (benchmark_prompting.py:48/58) -- vendored verbatim.
# ---------------------------------------------------------------------------

def ground_truth_belief_rows(record):
    rows = []
    for belief in record.get("beliefs", []):
        rows.append(
            {
                "actor": normalize_space(belief.get("actor", "")),
                "belief": normalize_space(belief.get("belief", "")),
                "order": normalize_space(belief.get("labels", {}).get("order", "")),
            }
        )
    return rows


def ground_truth_label_rows(record):
    rows = []
    for belief in record.get("beliefs", []):
        labels = belief.get("labels", {})
        row = {
            "actor": normalize_space(belief.get("actor", "")),
            "belief": normalize_space(belief.get("belief", "")),
        }
        for json_key, output_key in JSONL_LABEL_KEY_TO_OUTPUT.items():
            row[output_key] = normalize_space(labels.get(json_key, ""))
        rows.append(canonicalize_label_row(row))
    return rows


def _actor_belief_rows(beliefs):
    """benchmark_prompting.actor_belief_rows -- note: raw .strip(), NOT normalize_space
    (must match to reproduce the injected Stage-2 belief table byte-for-byte)."""
    rows = []
    for belief in beliefs:
        actor = str(belief.get("actor", "")).strip()
        text = str(belief.get("belief", "")).strip()
        if actor and text:
            rows.append({"actor": actor, "belief": text})
    return rows


def belief_table_pipe(beliefs):
    """benchmark_prompting.belief_table_pipe -- the 'Actor | Belief' table injected
    into the Stage-2 prompt."""
    lines = ["Actor | Belief"]
    for row in _actor_belief_rows(beliefs):
        lines.append(f"{row['actor']} | {row['belief']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt text vendored verbatim from prompts_extract.py / prompts_label.py.
# doc_to_text emits the USER prompt only; the SYSTEM prompt below is copied verbatim into
# each task YAML's `description` field, which lm-eval delivers as a real system role under
# --apply_chat_template (see the YAML comments / README). These constants remain the
# single source of truth checked against the YAML by the parity test.
# ---------------------------------------------------------------------------

# prompts_extract.PROMPT (Stage-1 system) -- exposed for the parity test + config.
SYSTEM_EXTRACT = (
    "You are a Theory of Mind expert whose task is to extract multi-order actor beliefs "
    "from the narrative and output a table with columns Actor, Belief, and Order by "
    "performing the following steps. A belief is a minimal proposition expressing what an "
    "actor takes to be true.\n"
    "1. Identify narrated events and states that the story presents as facts, and record "
    "them as world-level beliefs attributed to the special actor 'world' (order 0).\n"
    "2. Identify all actors, including characters or groups, who appear in the narrative "
    "and are capable of holding beliefs.\n"
    "3. For each actor, extract beliefs about the narrated events or states of the world, "
    "and record them as first-order beliefs (order 1).\n"
    "4. For each actor, extract beliefs about other actors’ beliefs, applying this "
    "notion recursively for nested beliefs, and record them as higher-order beliefs "
    "(order 2 or higher)."
)

_OUTRO_EXTRACT = "Now output only the table."

# prompts_label.PROMPT (Stage-2 system) -- exposed for the parity test + config.
SYSTEM_LABEL = (
    "You are a Theory of Mind expert whose task is to label a table of actor beliefs, "
    "given a narrative, by assigning a label from each of the following closed sets—"
    "Order (0/1/2/3), Truth-Status (True/False/Unknown), Knowledge-Access "
    "(Private/Shared/Public), Representation (Explicit/Implicit), Content Type (Location, "
    "Contents/Physical State, Identity/Relation, Epistemic, Desire/Intention, Emotion, "
    "Trait/Value, Action/Event), Mental-Source (Narration, Perception, Memory, Testimony, "
    "Inference, Imagination, Unknown), and Context (Deceptive, Temporal, Counterfactual, "
    "Neutral)—and outputting only a table with columns Actor and Belief, followed by "
    "one column for each labeling set.\n\n"
    "In this context, a belief is a minimal proposition expressing what an actor takes to "
    "be true about the world or about another actor’s mental state. Label each belief "
    "in the provided table by assigning values for the following dimensions, using the "
    "narrative as evidence:\n\n"
    "1. Determine the Order of the belief, which captures the depth of belief reasoning:\n"
    "   - Order 0: Narrator- or world-level facts that anchor the story’s ground "
    "truth and are not held by any actor.\n"
    "   - Order 1: First-order beliefs (A believes p).\n"
    "   - Order 2: Second-order beliefs (A believes B believes p).\n"
    "   - Order 3: Higher-order recursive beliefs (A believes B believes C believes p).\n\n"
    "2. Determine the Truth-Status of the belief relative to the narrative:\n"
    "   - True if the belief is verified or entailed by the narration.\n"
    "   - False if the belief is contradicted by the narration.\n"
    "   - Unknown if the narrative does not provide sufficient evidence.\n\n"
    "3. Determine the Knowledge-Access of the belief by assessing who could realistically "
    "know it in the story world:\n"
    "   - Private if the belief is held internally without evidence others know it.\n"
    "   - Shared if it is mutually known within a subgroup through explicit acknowledgment "
    "or obvious mutual awareness.\n"
    "   - Public if it is common ground across all actors (announced, jointly witnessed, "
    "or mutually known to be mutually known).\n\n"
    "4. Determine the Representation of the belief:\n"
    "   - Explicit if the belief is directly stated, spoken, or narrated as a mental "
    "state.\n"
    "   - Implicit if the belief must be inferred from actions, perception, or context.\n\n"
    "5. Determine the Content Type by identifying what the proposition is about:\n"
    "   - Use Action/Event for happenings; Desire/Intention for plans or goals.\n"
    "   - Use Location when the proposition concerns where an entity is or was, even if it "
    "involves a container\n"
    "   - Use Contents / Physical State only when the belief concerns what a container "
    "holds or an object’s condition.\n"
    "   - Use Epistemic for beliefs about beliefs, knowledge, attention, or awareness..\n\n"
    "6. Determine the Mental-Source of the belief, indicating how it was acquired:\n"
    "   - Narration (Order 0 only), Perception, Memory, Testimony, Inference, Imagination, "
    "or Unknown.\n\n"
    "7. Determine the Context of the belief:\n"
    "   - Deceptive if shaped by lying, omission, or misdirection.\n"
    "   - Temporal if the belief is outdated or reflects recall of a prior true state.\n"
    "     - Temporal + False indicates an outdated false belief.\n"
    "     - Temporal + True indicates accurate recall of a past fact.\n"
    "   - Counterfactual if the belief occurs in a hypothetical or pretense frame.\n"
    "   - Neutral if none apply."
)

_OUTRO_LABEL = "Now output only the completed table for the provided story."


# ---------------------------------------------------------------------------
# LOADER
# ---------------------------------------------------------------------------

_JSONL_NAME = "benchmark_story_belief_labels.jsonl"


def _dataset_path():
    """Locate benchmark_story_belief_labels.jsonl under a parent's
    benchmarks/omnitom-benchmark/ (read-only submodule; never copied)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "benchmarks" / "omnitom-benchmark" / _JSONL_NAME
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"{_JSONL_NAME} not found under any parent's benchmarks/omnitom-benchmark/"
    )


@functools.lru_cache(maxsize=1)
def _load_records():
    records = []
    with _dataset_path().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: int(r["story_id"]))
    return records


def _build_docs(category=None):
    docs = []
    for record in _load_records():
        if category is not None and record.get("story_category") != category:
            continue
        beliefs = record.get("beliefs", [])
        docs.append(
            {
                "story_id": int(record["story_id"]),
                "story_category": record.get("story_category", ""),
                "story": str(record.get("story", "")).strip(),
                "belief_table": belief_table_pipe(beliefs),
                # Nested lists JSON-stringified to avoid Arrow schema issues (dyntom pattern).
                "gold_labels": json.dumps(ground_truth_label_rows(record), ensure_ascii=False),
                "gold_beliefs": json.dumps(ground_truth_belief_rows(record), ensure_ascii=False),
            }
        )
    return docs


def load(category=None, **kwargs):
    """custom_dataset hook.  `category` (via dataset_kwargs) selects one of the 7
    ToMBench story categories for the by-category leaf tasks; None = all 895 stories."""
    return {"train": datasets.Dataset.from_list(_build_docs(category))}


# ---------------------------------------------------------------------------
# PROMPT FUNCTIONS (user prompt only; system prompt via each task's `description`)
# ---------------------------------------------------------------------------

def doc_to_text_label(doc):
    """Byte-matches build_label_messages(...)[1] (the user prompt)."""
    return (
        "Given the story narrative and the belief table below, label each belief and "
        "output only the completed pipe-separated labels table.\n\n"
        + "Story Narrative:\n"
        + doc["story"]
        + "\n\n"
        + "Belief table:\n"
        + doc["belief_table"]
        + "\n\n"
        + _OUTRO_LABEL
    )


def doc_to_text_extract(doc):
    """Byte-matches build_extract_messages(...)[1] (the user prompt)."""
    return (
        "Given the story narrative below, extract multi-order actor beliefs and output "
        "only the pipe-separated table with header: Actor | Belief | Order.\n\n"
        + "Story Narrative:\n"
        + doc["story"]
        + "\n\n"
        + _OUTRO_EXTRACT
    )


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def process_results_label(doc, results):
    """Stage-2 per-story per-dimension exact-match (compute_stage2_metrics:1158).

    Emits 8 per-story scalars: overall_acc + acc_<dim> x7.  Each YAML metric means
    these over stories, which equals the repo's safe_mean over per-story values
    (= macro-average over stories).  No story is empty, so lm-eval's mean == safe_mean.
    """
    gen = results[0]
    pred_rows = parse_label_rows(gen)
    gold_rows = json.loads(doc["gold_labels"])

    pred_index = pair_indexed_rows(pred_rows)
    gt_index = pair_indexed_rows(gold_rows)

    total = len(gt_index)
    dim_scores = {}
    for dimension in LABEL_DIMENSIONS:
        correct = 0
        for key, gt_row in gt_index.items():
            pred_row = pred_index.get(key)
            if pred_row and pred_row.get(dimension, "") and pred_row.get(dimension, "") == gt_row.get(dimension, ""):
                correct += 1
        dim_scores[dimension] = (correct / total) if total else 0.0

    out = {"overall_acc": safe_mean(dim_scores.values())}
    for dimension in LABEL_DIMENSIONS:
        out[_DIM_TO_METRIC[dimension]] = dim_scores[dimension]
    return out


def process_results_extract(doc, results):
    """Stage-1 DIAGNOSTIC proxies only (NOT the paper metric).

    The faithful Stage-1 F1 needs the GPT-5 judge and is produced offline by
    score_stage1_offline.py.  Here we emit a strict exact-string P/R/F1 lower bound
    (mirroring the mock backend's exact_matchcount_rows) plus the parsed row count, so
    the task is runnable/plumbing-testable even without --predict_only.
    """
    gen = results[0]
    pred_rows = parse_extraction_rows(gen)
    gold_rows = json.loads(doc["gold_beliefs"])

    pred_eval = exact_matchcount_rows(pred_rows, gold_rows)
    gt_eval = exact_matchcount_rows(gold_rows, pred_rows)
    pred_total = len(pred_eval)
    gt_total = len(gt_eval)
    pred_matched = sum(coerce_matchcount(r["MatchCount"]) > 0 for r in pred_eval)
    gt_matched = sum(coerce_matchcount(r["MatchCount"]) > 0 for r in gt_eval)
    precision = (pred_matched / pred_total) if pred_total else 0.0
    recall = (gt_matched / gt_total) if gt_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "extract_exact_f1": f1,
        "extract_exact_precision": precision,
        "extract_exact_recall": recall,
        "pred_row_count": float(len(pred_rows)),
    }
