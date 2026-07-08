"""Offline helpers for the SoNLI multi-stage counterfactual pipeline.

Typical full-pipeline flow:

1. Run `sonli_supporting` and `sonli_opposing` with `--predict_only --log_samples`.
2. Use `prepare` to build a judge-input JSONL with two rows per SoNLI item.
3. Run `sonli_judge` with `SONLI_JUDGE_DATA=<jsonl>` and `--log_samples`.
4. Use `score` to recompute Bayes posterior Pearson/MAE from the judge sample log
   if you want an offline artifact outside the lm-eval results table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import utils


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return ""


def _generation_from_sample(sample: dict[str, Any]) -> str:
    text = _first_text(sample.get("filtered_resps", []))
    if text:
        return text
    return _first_text(sample.get("resps", []))


def prepare_judge_inputs(
    support_samples: Path,
    oppose_samples: Path,
    output: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    rows_by_uuid: dict[str, dict[str, Any]] = {}

    for side, sample_path in (
        ("supporting", support_samples),
        ("opposing", oppose_samples),
    ):
        samples = _read_jsonl(sample_path)
        if limit is not None:
            samples = samples[:limit]
        for sample in samples:
            doc = sample["doc"]
            uuid = doc["uuid"]
            base = rows_by_uuid.setdefault(
                uuid,
                {
                    "uuid": uuid,
                    "inference": doc["inference"],
                    "human_score": float(doc["human_score"]),
                },
            )
            base[f"{side}_explanation"] = _generation_from_sample(sample)

    judge_rows = []
    for uuid in sorted(rows_by_uuid):
        item = rows_by_uuid[uuid]
        for side in ("supporting", "opposing"):
            explanation = item.get(f"{side}_explanation", "")
            judge_rows.append(
                {
                    "uuid": uuid,
                    "side": side,
                    "inference": item["inference"],
                    "explanation": explanation,
                    "human_score": item["human_score"],
                    "target": "",
                }
            )

    _write_jsonl(output, judge_rows)
    return {
        "output": str(output),
        "items": len(rows_by_uuid),
        "judge_rows": len(judge_rows),
    }


def score_judge_samples(judge_samples: Path, output: Path | None = None) -> dict[str, Any]:
    samples = _read_jsonl(judge_samples)
    payloads = []
    for sample in samples:
        doc = sample["doc"]
        raw = _generation_from_sample(sample)
        score = utils.parse_judge_score(raw)
        payloads.append(
            {
                "uuid": doc["uuid"],
                "side": doc["side"],
                "gold": float(doc["human_score"]),
                "score": score / 10.0 if score is not None else None,
                "parsed": score is not None,
            }
        )

    pairs = utils._judge_pairs(payloads)
    result = {
        "judge_docs": len(payloads),
        "judge_parse_rate": utils.agg_parse_rate(payloads),
        "items": len({p["uuid"] for p in payloads}),
        "paired_rate": utils.agg_paired_rate(payloads),
        "paired_items": len(pairs),
        "pearson": utils.agg_pearson(payloads),
        "mae": utils.agg_mae(payloads),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def released_baseline(output: Path | None = None) -> dict[str, Any]:
    payloads = []
    for doc in utils._base_docs():
        payloads.append(
            {
                "uuid": doc["uuid"],
                "gold": float(doc["human_score"]),
                "pred": float(doc["counterfactual_score"]),
                "parsed": True,
            }
        )

    result = {
        "items": len(payloads),
        "pearson": utils.agg_pearson(payloads),
        "mae": utils.agg_mae(payloads),
        "parse_rate": utils.agg_parse_rate(payloads),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SoNLI offline scoring helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="build sonli_judge JSONL from stage-1 sample logs")
    prep.add_argument("--support-samples", required=True, type=Path)
    prep.add_argument("--oppose-samples", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--limit", type=int, default=None)

    score = sub.add_parser("score", help="score sonli_judge sample log offline")
    score.add_argument("--judge-samples", required=True, type=Path)
    score.add_argument("--output", type=Path, default=None)

    rel = sub.add_parser("released-baseline", help="score released counterfactual_score vs humans")
    rel.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_judge_inputs(
            args.support_samples, args.oppose_samples, args.output, args.limit
        )
    elif args.command == "score":
        result = score_judge_samples(args.judge_samples, args.output)
    else:
        result = released_baseline(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
