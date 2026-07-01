"""lm-eval adapter for DialToM (Yadav, Achananuparp, Jiang & Lim, 2026; arXiv:2604.20443).

v1 scope = the paper's own "core" (its two MCQ tasks, per Section 3.3: "the DialToM
benchmark consists of two distinct multiple-choice QA tasks") plus the Written
inference task (Appendix G "Supplementary Semantic Verification"). NOT built here
(deferred; each is a distinct pipeline, not a static reproducible task):
Prospective-Easy (Table 6, distractor-difficulty ablation), Counterfactual-NOTA
(Table 7, needs a live GPT-4o counterfactual-generation pass), Teacher-Student
reasoning injection (Table 8, replays one frozen historical Gemini-3-Pro trace),
Dialogue Context-Degradation (Appendix F), and LLM-as-a-Judge semantic scoring
(Table 14, needs a live judge-model call; the paper itself calls it "qualitative
explorations rather than primary empirical claims").

Faithfulness over convenience: prompts below are transcribed verbatim from the
f-strings in the read-only submodule `benchmarks/DialToM/benchmark.py` (never
imported — never edited). Provenance:

  - retrospective prompt  <- benchmark.py: retrospective()
  - prospective prompt    <- benchmark.py: prospective() (exp='normal' branch only)
  - written prompt        <- benchmark.py: written()
  - written NLP scorers   <- benchmark.py / written_metrics.py: get_bleu/get_rouge/get_bertscore
    (reconstructed with a multi-reference driver — the original ships no such driver;
    the paper's Appendix G says only "we report the average similarity scores across
    all references to handle multiple gold references", i.e. mean-over-3-refs).

Gate A (output_type): the paper's formalism (Sec 3.3, argmax over loglikelihood)
reads as `multiple_choice`; the actual run code (JSON-schema-constrained generation +
exact-match, reasoning traces permitted for thinking models) reads as `generate_until`
+ letter extractor -- the Hi-ToM pattern. Built as `generate_until` (matches what
actually produced the paper's headline numbers).

Deviation notes (see README.md for the full list):
  - Retrospective options are NOT re-shuffled at prompt-render time. The verified data
    files already carry a fixed, valid A-D option order + correct_option letter (the
    runtime `np.random.shuffle` in benchmark.py is an *additional* re-randomization on
    top of that, sequentially consumed across the whole eval loop -- reproducing it
    exactly buys nothing since the per-doc content is unaffected, only the label).
    This is the same "ship is already eval-ready; do not re-shuffle" call the dyntom
    adapter made.
  - Prospective options (correct_action + 3 distractors) have no such frozen order, so
    each doc gets a per-item deterministic shuffle seeded on the item's own `id`
    (`random.Random(item_id)`), not the original's sequentially-consumed global
    `np.random` stream. Reproducible across our own re-runs; not letter-identical to
    any specific paper run (which nothing except that run itself would be, in either
    case, since a fresh shuffle happens every invocation of the original script).
  - Prospective prompt keeps the original's literal (buggy) hardcoded "client" in
    "Mental State profile ... of client during the conversation" even for
    ESC/PFG domains where the recipient is Seeker/Persuadee -- transcribed verbatim
    from benchmark.py, not fixed, per "never invent/correct prompt wording."
"""

import functools
import json
import random
import re
from pathlib import Path

import datasets

# ---------------------------------------------------------------------------
# Shared constants (vendored from benchmarks/DialToM/utils.py)
# ---------------------------------------------------------------------------

AGENT1 = {"MI": "Counselor", "PFG": "Persuader", "ESC": "Supporter"}
AGENT2 = {"MI": "Client", "PFG": "Persuadee", "ESC": "Seeker"}

STEER = {
    "Belief": "I believe",
    "Desires": "I want",
    "Intentions": "I will",
    "Emotions": "I feel",
    "Knowledge": "I know",
    "Trust": "I view the {agent1} as",
}
MENTAL_STATES = ["Belief", "Desires", "Intentions", "Emotions", "Knowledge", "Trust"]
OPTION_LETTERS = ["A", "B", "C", "D"]


def _data_dir():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks" / "DialToM" / "data"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "benchmarks/DialToM/data not found under any parent of this file"
    )


def _agent_replace(text, domain):
    a1, a2 = AGENT1[domain], AGENT2[domain]
    return (
        text.replace("agent 1", a1)
        .replace("agent 2", a2)
        .replace("agent1", a1)
        .replace("agent2", a2)
    )


# ---------------------------------------------------------------------------
# RETROSPECTIVE — LOADER + PROMPT-FN
# ---------------------------------------------------------------------------

def load_retrospective(domain=None, **kwargs):
    domain = domain or kwargs.get("domain")
    path = _data_dir() / f"{domain}_retrospective_verified.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    docs = []
    for item in raw:
        state = item["state"]
        opts = item["options"][state]
        docs.append(
            {
                "id": item["id"],
                "domain": domain,
                "state": state,
                "ctx": "\n".join(item["ctx"]),
                "topic": item["topic"],
                "task_desc": item["task_desc"],
                "choice_a": opts["A"],
                "choice_b": opts["B"],
                "choice_c": opts["C"],
                "choice_d": opts["D"],
                "gold_letter": item["correct_option"][state],
            }
        )
    return {"train": datasets.Dataset.from_list(docs)}


def doc_to_text_retrospective(doc):
    options = (
        f"A: {doc['choice_a']}\nB: {doc['choice_b']}\n"
        f"C: {doc['choice_c']}\nD: {doc['choice_d']}"
    )
    agent1, agent2 = AGENT1[doc["domain"]], AGENT2[doc["domain"]]
    return f"""You are an expert in Theory of Mind reasoning.

Task:
You will be provided with conversation between two agents {agent1} and {agent2} engaging in a {doc['task_desc']} session on the topic of {doc['topic']}.

Your goal is to correctly infer {agent2}'s {doc['state']} state, based on the above conversation. You will be provided with a set of options, and you need to choose the most appropriate one that reflects the {doc['state']} state.

The correct option must be consistent with the provided conversation context.

Conversation Context:
{doc['ctx']}

Mental State Options:
{options}

Instruction
Output only the letter of the correct option (e.g., "A", "B", "C", or "D"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE. ONLY OUTPUT THE OPTION AND NOTHING ELSE. YOUR OUTPUT SHOULD STRICTLY BE ONE OF A, B, C, or D.

Answer:"""


# ---------------------------------------------------------------------------
# PROSPECTIVE — LOADER + PROMPT-FN (context withheld; per-item deterministic shuffle)
# ---------------------------------------------------------------------------

def _shuffle_action_options(item_id, correct_text, distractor_texts):
    opts = [correct_text] + list(distractor_texts)
    order = list(range(len(opts)))
    random.Random(item_id).shuffle(order)
    shuffled = [opts[i] for i in order]
    gold_letter = OPTION_LETTERS[order.index(0)]
    return shuffled, gold_letter


def load_prospective(domain=None, **kwargs):
    domain = domain or kwargs.get("domain")
    path = _data_dir() / f"{domain}_prospective_verified.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    docs = []
    for item in raw:
        mental_states_block = "\n".join(
            f"{s}: {item['options'][s][item['correct_option'][s]]}" for s in MENTAL_STATES
        )
        shuffled, gold_letter = _shuffle_action_options(
            item["id"], item["correct_action"], item["distractors"]
        )
        shuffled = [_agent_replace(t, domain) for t in shuffled]
        docs.append(
            {
                "id": item["id"],
                "domain": domain,
                "state": item["state"],
                "mental_states_block": mental_states_block,
                "choice_a": shuffled[0],
                "choice_b": shuffled[1],
                "choice_c": shuffled[2],
                "choice_d": shuffled[3],
                "gold_letter": gold_letter,
            }
        )
    return {"train": datasets.Dataset.from_list(docs)}


def doc_to_text_prospective(doc):
    options = (
        f"A: {doc['choice_a']}\nB: {doc['choice_b']}\n"
        f"C: {doc['choice_c']}\nD: {doc['choice_d']}"
    )
    agent2 = AGENT2[doc["domain"]]
    return f"""You are an expert in Theory of Mind reasoning.

Task:
You will be provided with internal Mental State profile (Belief, Desire, Intention, Emotion, Knowledge, Trust) of client during the conversation.

Your goal is to identify which of the candidate conversation segments is the most plausible continuation of this conversation.
The correct option must be consistent with the provided Mental States of the client.

Mental state of {agent2}
{doc['mental_states_block']}

Candidate Conversation Segments
{options}

Instruction
Output only the letter of the correct option (e.g., "A", "B", "C", or "D"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE.

Answer:"""


# ---------------------------------------------------------------------------
# EXTRACTOR + METRIC-FN — shared by retrospective & prospective (4-way letter MCQ)
# ---------------------------------------------------------------------------

_LETTER_PATTERNS = [
    r"^\s*\(?([ABCD])\)?\s*$",
    r"^\s*\(?([ABCD])[\.\):]",
    r"answer\s*(?:is|:)\s*\(?([ABCD])\b",
    r"\(([ABCD])\)",
    r"\b([ABCD])\b",
]


def _extract_letter(gen):
    """Reasoning models (Gemini 3 Pro/GPT-5 in the paper; local models like Qwen3 here)
    interleave a <think>...</think> preamble ahead of the actual answer -- option
    letters mentioned mid-reasoning must not be mistaken for the final choice. If a
    </think> close tag is present, only the text after the LAST one is searched; within
    that, the LAST match of each pattern wins (a reasoning trace may restate letters
    before settling on one)."""
    text = gen.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if not text:
        return None
    for pat in _LETTER_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def process_results_mcq(doc, results):
    pred = _extract_letter(results[0])
    return {"acc": 1.0 if pred == doc["gold_letter"] else 0.0}


# ---------------------------------------------------------------------------
# WRITTEN — LOADER (join written_inference.csv refs x3 with combined_written_data.json)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_written_docs():
    import pandas as pd

    refs_df = pd.read_csv(_data_dir() / "written_inference.csv", dtype=str)
    combined = json.loads(
        (_data_dir() / "combined_written_data.json").read_text(encoding="utf8")
    )

    grouped = {}
    for _, row in refs_df.iterrows():
        key = (row["dataset"], row["mental_state"], row["sub_id"])
        grouped.setdefault(key, []).append(row["inferences"])

    docs = []
    for (domain, state, sub_id), refs in grouped.items():
        if len(refs) != 3:
            raise ValueError(f"expected 3 refs for {domain}/{state}/{sub_id}, got {len(refs)}")
        item = combined[domain][sub_id]
        ctx = _agent_replace("\n".join(item["ctx"]), domain)
        steer = STEER[state]
        if state == "Trust":
            steer = steer.format(agent1=AGENT1[domain])
        docs.append(
            {
                "domain": domain,
                "state": state,
                "sub_id": sub_id,
                "ctx": ctx,
                "steer": steer,
                "refs": refs,
            }
        )
    return docs


def load_written(**kwargs):
    return {"train": datasets.Dataset.from_list(_load_written_docs())}


def doc_to_text_written(doc):
    agent1, agent2 = AGENT1[doc["domain"]], AGENT2[doc["domain"]]
    return f"""You are an expert in Theory of Mind reasoning.

Task:
You will be provided with a context of a conversation between {agent1} and {agent2}.

Your goal is to accurately infer the {doc['state']} mental state of the {agent2} in one line.
The correct option must be consistent with the provided conversational context.

Conversation Context:
{doc['ctx']}

Instruction
Output only a line inference. Your response should always start with "{doc['steer']}". Do not add explanations or other verbosity. STRICTLY FOLLOW THIS FORMAT AND OUTPUT ONLY ONE LINE.

Answer:"""


# ---------------------------------------------------------------------------
# WRITTEN — METRIC-FN (mean-over-3-refs BLEU / ROUGE-L / BERTScore-F1)
# Appendix G: "we report the average similarity scores across all references
# to handle multiple gold references."
# ---------------------------------------------------------------------------

_BLEU = None
_ROUGE = None


def _bleu_metric():
    global _BLEU
    if _BLEU is None:
        from sacrebleu.metrics import BLEU

        _BLEU = BLEU()
    return _BLEU


def _rouge_metric():
    global _ROUGE
    if _ROUGE is None:
        from rouge import Rouge

        _ROUGE = Rouge()
    return _ROUGE


@functools.lru_cache(maxsize=1)
def _bertscorer():
    """Lazy RoBERTa-large singleton (~1.4GB), same vendoring pattern as the fantom
    adapter's embedder: loaded once, only when a written-task doc is actually scored."""
    import bert_score

    return bert_score.BERTScorer(lang="en", rescale_with_baseline=True)


def process_results_written(doc, results):
    hyp = results[0].strip()
    if "</think>" in hyp:
        hyp = hyp.rsplit("</think>", 1)[-1].strip()
    refs = doc["refs"]

    if not hyp:
        return {"bleu": 0.0, "rouge_l": 0.0, "bertscore_f1": 0.0}

    bleu_scorer = _bleu_metric()
    bleu = sum(bleu_scorer.sentence_score(hyp, [r]).score for r in refs) / len(refs)

    rouge_scorer = _rouge_metric()
    rouge_l = sum(
        rouge_scorer.get_scores(hyp, r)[0]["rouge-l"]["f"] for r in refs
    ) / len(refs)

    scorer = _bertscorer()
    _, _, f1 = scorer.score([hyp] * len(refs), refs)
    bertscore_f1 = f1.mean().item()

    return {"bleu": bleu, "rouge_l": rouge_l, "bertscore_f1": bertscore_f1}
