#!/usr/bin/env python3
"""Compare XToM lm-eval result files with the paper's Qwen-2.5-7B rows.

The targets below are transcribed from XToM Tables 8--10.  Values are
percentages, matching the adapter's custom aggregations.  When a task has been
run more than once, the newest result (by lm-eval's top-level ``date`` field,
then file modification time) is used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path


def _targets(columns: tuple[str, ...], rows: dict[str, tuple[float, ...]]):
    return {
        f"{language}_{metric}": value
        for language, values in rows.items()
        for metric, value in zip(columns, values, strict=True)
    }


PAPER_TARGETS = {
    "xtom_xfantom": _targets(
        (
            "belief_acc",
            "belief_first_acc",
            "belief_second_acc",
            "belief_acyclic_acc",
            "belief_cyclic_acc",
            "fact_acc",
        ),
        {
            "en": (26.4, 25.3, 28.7, 31.4, 26.0, 83.0),
            "zh": (25.2, 20.7, 34.70, 31.4, 38.0, 81.3),
            "de": (29.9, 29.5, 30.7, 19.6, 42.0, 86.3),
            "fr": (27.4, 25.8, 30.7, 23.5, 38.0, 82.7),
            "ja": (43.4, 37.3, 56.4, 51.0, 62.0, 80.3),
        },
    ),
    "xtom_xfantom_cot": _targets(
        (
            "belief_acc",
            "belief_first_acc",
            "belief_second_acc",
            "belief_acyclic_acc",
            "belief_cyclic_acc",
            "fact_acc",
        ),
        {
            "en": (31.4, 26.3, 42.6, 47.1, 38.0, 85.3),
            "zh": (85.8, 82.9, 92.1, 88.2, 96.0, 94.3),
            "de": (28.0, 24.9, 34.7, 29.4, 40.0, 80.7),
            "fr": (27.4, 23.5, 35.6, 29.4, 42.0, 81.7),
            "ja": (45.9, 39.2, 60.4, 56.9, 64.0, 76.3),
        },
    ),
    "xtom_xtomi": _targets(
        ("first_order_acc", "second_order_acc", "belief_acc", "reality_acc"),
        {
            "en": (46.34, 40.63, 42.65, 99.33),
            "zh": (17.89, 54.91, 41.79, 97.00),
            "de": (65.85, 47.32, 53.89, 87.00),
            "fr": (34.96, 25.45, 28.82, 99.00),
            "ja": (45.53, 70.09, 61.38, 74.00),
        },
    ),
    "xtom_xtomi_cot": _targets(
        ("first_order_acc", "second_order_acc", "belief_acc", "reality_acc"),
        {
            "en": (60.98, 55.36, 57.35, 96.67),
            "zh": (19.51, 59.82, 45.53, 87.00),
            "de": (69.11, 52.23, 58.21, 71.00),
            "fr": (49.59, 34.82, 40.06, 97.00),
            "ja": (55.28, 77.68, 69.74, 61.67),
        },
    ),
    "xtom_xnegtom": _targets(
        (
            "belief_exact_match",
            "desire_exact_match",
            "intention_micro_f1",
            "intention_macro_f1",
        ),
        {
            "en": (7.34, 13.33, 44.00, 35.00),
            "zh": (18.28, 18.97, 46.00, 34.00),
            "de": (16.50, 17.84, 44.00, 31.00),
            "fr": (11.34, 13.67, 43.00, 32.00),
            "ja": (12.17, 18.50, 42.00, 34.00),
        },
    ),
    "xtom_xnegtom_cot": _targets(
        (
            "belief_exact_match",
            "desire_exact_match",
            "intention_micro_f1",
            "intention_macro_f1",
        ),
        {
            "en": (8.34, 14.00, 48.00, 38.00),
            "zh": (15.00, 21.03, 50.00, 36.00),
            "de": (14.00, 12.00, 43.00, 29.00),
            "fr": (7.50, 11.34, 44.00, 32.00),
            "ja": (16.50, 19.67, 42.00, 34.00),
        },
    ),
}


def _result_files(paths: list[Path]) -> list[Path]:
    files = set()
    for path in paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("results_*.json"))
        else:
            raise FileNotFoundError(path)
    return sorted(files)


def _metric_value(result: dict, metric: str):
    for key, value in result.items():
        if key.split(",", 1)[0] == metric:
            return float(value)
    return None


def _newest_results(files: list[Path]):
    newest = {}
    failures = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{path}: {error}")
            continue
        sort_key = (str(payload.get("date", "")), path.stat().st_mtime_ns)
        for task, result in payload.get("results", {}).items():
            if task not in PAPER_TARGETS:
                continue
            if task not in newest or sort_key > newest[task][0]:
                newest[task] = (sort_key, path, result)
    return newest, failures


def _rows(newest):
    rows = []
    missing_metrics = []
    for task in PAPER_TARGETS:
        if task not in newest:
            continue
        _, path, result = newest[task]
        for metric, paper in PAPER_TARGETS[task].items():
            actual = _metric_value(result, metric)
            if actual is None:
                missing_metrics.append(f"{task}:{metric}")
                continue
            language, short_metric = metric.split("_", 1)
            rows.append(
                {
                    "task": task,
                    "language": language,
                    "metric": short_metric,
                    "paper": paper,
                    "adapter": actual,
                    "delta": actual - paper,
                    "result_file": str(path),
                }
            )
    return rows, missing_metrics


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.2f}"


def _markdown(rows, newest) -> str:
    lines = [
        "| Task | Lang | Metric | Paper | Adapter | Delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['language']} | {row['metric']} | "
            f"{_fmt(row['paper'])} | {_fmt(row['adapter'])} | {_fmt(row['delta'])} |"
        )

    lines.extend(["", "Summary (percentage-point deltas):", ""])
    lines.extend(["| Task | Cells | MAE | Max abs. delta | Result file |", "|---|---:|---:|---:|---|"])
    for task in PAPER_TARGETS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        finite = [abs(row["delta"]) for row in task_rows if math.isfinite(row["delta"])]
        mae = sum(finite) / len(finite) if finite else math.nan
        maximum = max(finite, default=math.nan)
        path = newest[task][1]
        lines.append(
            f"| {task} | {len(task_rows)} | {_fmt(mae)} | {_fmt(maximum)} | `{path}` |"
        )
    return "\n".join(lines) + "\n"


def _csv(rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]) if rows else (
        "task", "language", "metric", "paper", "adapter", "delta", "result_file"
    ))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="result JSON file(s) or directories")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--output", type=Path, help="write the report instead of stdout")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any of the six tasks or paper metrics is missing",
    )
    args = parser.parse_args()

    try:
        files = _result_files(args.paths)
    except FileNotFoundError as error:
        parser.error(f"path does not exist: {error}")
    if not files:
        parser.error("no results_*.json files found")

    newest, failures = _newest_results(files)
    rows, missing_metrics = _rows(newest)
    missing_tasks = [task for task in PAPER_TARGETS if task not in newest]
    report = _csv(rows) if args.format == "csv" else _markdown(rows, newest)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    for failure in failures:
        print(f"warning: could not read {failure}", file=sys.stderr)
    if missing_tasks:
        print(f"warning: missing tasks: {', '.join(missing_tasks)}", file=sys.stderr)
    if missing_metrics:
        print(f"warning: missing metrics: {', '.join(missing_metrics)}", file=sys.stderr)
    return int(args.strict and bool(failures or missing_tasks or missing_metrics))


if __name__ == "__main__":
    raise SystemExit(main())
