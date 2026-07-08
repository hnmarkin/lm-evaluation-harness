"""lm-eval adapter utilities for SoNLI (SocialNLI; arXiv:2510.05458).

The read-only benchmark lives under benchmarks/SoNLI. This adapter exposes:

* sonli: a static, direct scalar plausibility task over eval.json.
* sonli_supporting / sonli_opposing: stage-1 counterfactual explanation prompts.
* sonli_judge: stage-2 judge scoring over a materialized JSONL of generated
  explanations, with Bayes posterior Pearson/MAE aggregation.

The full paper protocol is multi-stage, so the stage handoff is prepared offline
from lm-eval --log_samples outputs. All prompt/scoring fragments below are
vendored from the benchmark's pure Python prompt/eval code, not imported.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import datasets


_DIRECT_METRICS = ("pearson", "mae", "parse_rate")
_JUDGE_METRICS = ("pearson", "mae", "parse_rate", "paired_rate")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "benchmarks" / "SoNLI").exists():
            return parent
    raise FileNotFoundError("Could not find benchmarks/SoNLI from sonli utils.py")


@functools.lru_cache(maxsize=1)
def _read_eval() -> list[dict[str, Any]]:
    path = _repo_root() / "benchmarks" / "SoNLI" / "datasets" / "socialnli" / "eval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _base_docs() -> list[dict[str, Any]]:
    docs = []
    for raw in _read_eval():
        human = _as_float(raw["human_annotated_score"])
        docs.append(
            {
                "uuid": raw["uuid"],
                "dialogue": raw["dialogue"],
                "question": raw["question"],
                "inference": raw["inference"],
                "classification": raw.get("classification", ""),
                "inference_type": raw.get("inference_type", ""),
                "model": raw.get("model", ""),
                "human_score": human,
                "target": f"{human:.6f}",
                "human_annotated_explanation": raw.get("human_annotated_explanation", ""),
                "supporting_explanation": raw.get("supporting_explanation", ""),
                "opposing_explanation": raw.get("opposing_explanation", ""),
                "supporting_explanation_score": _as_float(raw.get("supporting_explanation_score", 0.0)),
                "opposing_explanation_score": _as_float(raw.get("opposing_explanation_score", 0.0)),
                "counterfactual_score": _as_float(raw.get("counterfactual_score", 0.0)),
            }
        )
    return docs


def load(
    classification: str | None = None,
    inference_type: str | None = None,
    model: str | None = None,
    limit: int | str | None = None,
    **kwargs,
):
    docs = _base_docs()
    if classification:
        docs = [d for d in docs if d["classification"] == classification]
    if inference_type:
        docs = [d for d in docs if d["inference_type"] == inference_type]
    if model:
        docs = [d for d in docs if d["model"] == model]
    if limit:
        docs = docs[: int(limit)]
    return {"train": datasets.Dataset.from_list(docs)}


def _direct_prompt(doc: dict[str, Any]) -> str:
    return (
        "You are evaluating a social inference about a dialogue transcript.\n"
        "Read the dialogue, the motivating question, and the inference statement. "
        "Assign the probability that the inference is true given only the dialogue.\n"
        "Use a number from 0.0 to 1.0, where 0.0 means false or impossible, "
        "0.5 means equally likely true or false, and 1.0 means certainly true.\n"
        "Return exactly one line in this form: SCORE: <number>\n\n"
        "DIALOGUE:\n"
        f"{doc['dialogue']}\n\n"
        "QUESTION:\n"
        f"{doc['question']}\n\n"
        "INFERENCE:\n"
        f"{doc['inference']}\n\n"
        "SCORE:"
    )


def doc_to_text(doc: dict[str, Any]) -> str:
    return _direct_prompt(doc)


# Vendored from src/prompts/supporting_explanations_no_q.py.
def doc_to_text_supporting(doc: dict[str, Any]) -> str:
    return f"""<role> You are an AI assistant specializing in creating concise, evidence-based explanations to support inferences. </role>

<task> Create an explanation that directly supports the inference. Find and cite specific evidence that directly supports the inference. Focus on relevant details that prove the inference is true. If there is no evidence that supports the inference, simply state that there is no evidence that supports the inference. Do not repeat the inference or dialogue in the explanation. </task>

<tone> The explanation should be simple, concise and declarative. It should be a single sentence that directly supports the inference. </tone>

<format> Write your thoughts in <think> </think> tags. Do not include any additional text, explanations, or formatting in the answer. </format>

<dialogue>
{doc['dialogue']}
</dialogue>

<inference> {doc['inference']} </inference>

<think>
"""


# Vendored from src/prompts/opposing_explanations_no_q.py.
def doc_to_text_opposing(doc: dict[str, Any]) -> str:
    return f"""<role> You are an AI assistant specializing in creating concise, evidence-based explanations to prove inferences false. </role>

<task> Create an explanation that proves that the inference is false. Find and cite specific evidence that directly contradicts the inference. Focus on relevant details that prove the inference is false. If there is no evidence that contradicts the inference, simply state that there is no evidence that contradicts the inference. Do not repeat the inference or dialogue in the explanation. </task>

<tone> The explanation should be simple, concise and declarative. It should be a single sentence that opposes the inference. </tone>

<format> Write your thoughts in <think> </think> tags. Do not include any additional text, explanations, or formatting in the answer. </format>

<dialogue>
{doc['dialogue']}
</dialogue>

<inference> {doc['inference']} </inference>

<think>
"""


# Vendored from src/prompts/judge.py.
def _scoring_prompt(explanation: str, inference: str) -> str:
    return f"""You are a reasoning system that analyzes the likelihood of complex events given information about hypothetical scenarios. 

You are given a description of a fictional scenario and a hypothesis about that scenario that may or may not be true. Given the situation, you will first score the likelihood that this hypothesis is true, on a scale from 0 to 10, using the following rubric as guidance:

0 (virtually impossible): Essentially no way the hypothesis could possibly be true, given the evidence. Less likely than being struck by lightning.
1 (unlikely): The hypothesis is unlikely, but definitely not impossible.
2 (possible): The hypothesis could be true given the evidence, but there is better chance that it is false. Less likely than drawing a card of the suit of clubs from a standard card deck.
3 (reasonable chance): You would not be more than mildly surprised that the hypothesis is true. About one thirds chance.
4 (a bit less than even-odds): Slightly below fifty-fifty probability. You would not bet more than a small sum that the hypothesis is false.
5 (fifty-fifty): Given the information about the situation, there is approximately equal chance that the hypothesis is true vs. the hypothesis is false. As likely as a fair coin landing on heads.
6 (a bit more than even-odds): Slightly above fifty-fifty probability. You would not bet more than a small sum that the hypothesis is true.
7 (probable): Likely, but you would still not be overly surprised if the hypothesis turned out to be false.
8 (quite likely): About as likely as *not* rolling a "2" with a six-sided die.
9 (extremely likely): Quite certain. You would bet a large amount of money on the hypothesis being true.
10 (practically certain): You cannot imagine a scenario in which the hypothesis is not true, given the situational evidence.

Label your initial prediction with (0), and label your updated predictions with the evidence number it corresponds to. Write your enumerated explanations and probability scores, and nothing else.

Here is a first example:

ORIGINAL DESCRIPTION: There were puddles in the street and dark clouds hung overhead. The Mississippi flag was visible on a nearby car.

HYPOTHESIS: A tornado rolled through a town in Mississippi.

EXPLANATION: It is more likely this was just a regular rainstorm than a tornado. While it might be in Mississippi, there's not enough evidence to claim a tornado occurred.

SCORE: 1

Here is a second example:

ORIGINAL DESCRIPTION: There is a large crowd of people gathered before a lit-up stage at night.

HYPOTHESIS: The band Blur performed at Coachella 2024.

EXPLANATION: While a big nighttime show with a large crowd could describe many concerts or festivals, there is no direct evidence that this specific event is Coachella or that the band on stage is Blur.

SCORE: 2

That is the end of the examples. Now, it's time for you to assign probabilities to a new fictional scenario:

ORIGINAL DESCRIPTION: {explanation}

HYPOTHESIS: {inference}

EXPLANATION:
"""


def load_judge(data_file: str | None = None, limit: int | str | None = None, **kwargs):
    data_file = data_file or os.environ.get("SONLI_JUDGE_DATA")
    if not data_file:
        raise FileNotFoundError(
            "sonli_judge needs dataset_kwargs.data_file or SONLI_JUDGE_DATA "
            "pointing at a judge-input JSONL produced by offline_score.py prepare."
        )

    path = Path(data_file)
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"SoNLI judge input file not found: {path}")

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if limit:
        rows = rows[: int(limit)]
    return {"train": datasets.Dataset.from_list(rows)}


def doc_to_text_judge(doc: dict[str, Any]) -> str:
    return _scoring_prompt(doc["explanation"], doc["inference"])


def _after_think(text: str) -> str:
    if "</think>" in text.lower():
        match = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))[-1]
        return text[match.end() :]
    return text


def extract_direct_score(text: str) -> float | None:
    """Parse a direct plausibility score normalized to [0, 1]."""
    region = _after_think(str(text or "")).strip()
    patterns = [
        r"SCORE\s*:\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*%?",
        r"(?:probability|plausibility|rating|score)[^\d+-]{0,30}([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*%?",
        r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*%",
        r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, region, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = float(match.group(1))
            if "%" in match.group(0):
                value = value / 100.0
            if 0.0 <= value <= 1.0:
                return value
            if 1.0 < value <= 100.0 and "%" in match.group(0):
                return value / 100.0
            return None
    return None


# Vendored from experiment_one.py, with prints removed.
def parse_judge_score(judge_response_content: str) -> int | None:
    if not judge_response_content:
        return None

    content = re.sub(
        r"</?(think|answer)>", "", str(judge_response_content), flags=re.IGNORECASE
    ).strip()
    match = re.search(r"SCORE:\s*(\d+)", content, re.MULTILINE | re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return score if 0 <= score <= 10 else None

    numbers = re.findall(r"\b(\d+)\b", content)
    for num_str in reversed(numbers):
        score = int(num_str)
        if 0 <= score <= 10:
            return score
    return None


# Vendored from experiment_one.py.
def calculate_final_bayes_score(s_plus: float, s_minus: float) -> float:
    s_plus = max(0.0, min(1.0, s_plus))
    s_minus = max(0.0, min(1.0, s_minus))
    numerator = s_plus * (1 - s_minus)
    denominator = numerator + ((1 - s_plus) * s_minus)
    if denominator == 0:
        return 0.5
    return numerator / denominator


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    raw = results[0] if results else ""
    pred = extract_direct_score(raw)
    payload = {
        "uuid": doc["uuid"],
        "gold": float(doc["human_score"]),
        "pred": pred,
        "parsed": pred is not None,
    }
    return {metric: payload for metric in _DIRECT_METRICS}


def process_results_explanation(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    raw = results[0] if results else ""
    return {"nonempty_rate": 1.0 if str(raw).strip() else 0.0}


def process_results_judge(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    raw = results[0] if results else ""
    score = parse_judge_score(raw)
    norm = score / 10.0 if score is not None else None
    payload = {
        "uuid": doc["uuid"],
        "side": doc["side"],
        "gold": float(doc["human_score"]),
        "score": norm,
        "parsed": norm is not None,
    }
    return {metric: payload for metric in _JUDGE_METRICS}


def _direct_pairs(items: list[dict[str, Any]]) -> list[tuple[float, float]]:
    pairs = []
    for item in items:
        pred = item.get("pred")
        if item.get("parsed") and pred is not None:
            pairs.append((float(pred), float(item["gold"])))
    return pairs


def _judge_pairs(items: list[dict[str, Any]]) -> list[tuple[float, float]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        entry = grouped.setdefault(
            item["uuid"], {"gold": float(item["gold"]), "supporting": None, "opposing": None}
        )
        side = str(item.get("side", "")).lower()
        if item.get("parsed"):
            if side.startswith("support"):
                entry["supporting"] = float(item["score"])
            elif side.startswith("oppos"):
                entry["opposing"] = float(item["score"])

    pairs = []
    for entry in grouped.values():
        if entry["supporting"] is None or entry["opposing"] is None:
            continue
        pred = calculate_final_bayes_score(entry["supporting"], entry["opposing"])
        pairs.append((pred, entry["gold"]))
    return pairs


def _prediction_pairs(items: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not items:
        return []
    if "side" in items[0]:
        return _judge_pairs(items)
    return _direct_pairs(items)


def _pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xs = [p for p, _ in pairs]
    ys = [g for _, g in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def agg_pearson(items: list[dict[str, Any]]) -> float:
    return _pearson(_prediction_pairs(items))


def agg_mae(items: list[dict[str, Any]]) -> float:
    pairs = _prediction_pairs(items)
    if not pairs:
        return float("nan")
    return sum(abs(pred - gold) for pred, gold in pairs) / len(pairs)


def agg_parse_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return float("nan")
    return sum(1 for item in items if item.get("parsed")) / len(items)


def agg_paired_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return float("nan")
    if "side" not in items[0]:
        return agg_parse_rate(items)
    grouped: dict[str, set[str]] = {}
    for item in items:
        side = str(item.get("side", "")).lower()
        if not item.get("parsed"):
            continue
        if side.startswith("support"):
            grouped.setdefault(item["uuid"], set()).add("supporting")
        elif side.startswith("oppos"):
            grouped.setdefault(item["uuid"], set()).add("opposing")
    total = len({item["uuid"] for item in items})
    paired = sum(1 for sides in grouped.values() if {"supporting", "opposing"} <= sides)
    return paired / total if total else float("nan")

