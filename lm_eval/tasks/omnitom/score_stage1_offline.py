#!/usr/bin/env python
"""Offline Stage-1 scorer for the OmniToM lm-eval adapter.

Stage 1 (belief extraction) is scored by a LIVE GPT-5 semantic judge, which cannot run
inside lm-eval.  So Stage 1 is run generate-only in the harness:

    lm-eval --model hf --model_args pretrained=<model> \\
        --config lm_eval/tasks/omnitom/eval_config_extract.yaml \\
        --tasks omnitom_extract --predict_only --log_samples \\
        --output_path out/omnitom_extract

and this script bridges the harness output to the benchmark's OWN judge+metrics so the
canonical GPT-5 judge and macro-over-stories P/R/F1 are preserved byte-for-byte:

    OPENAI_API_KEY=... python lm_eval/tasks/omnitom/score_stage1_offline.py \\
        --samples out/omnitom_extract \\
        --output-dir runs/omnitom_stage1

It (1) reads the harness --log_samples file for omnitom_extract, (2) parses each
generation into (actor, belief, order) rows with the benchmark's OWN parser and writes
them into the repo's extraction/csv/ layout, then (3) shells out to
`run_replication.py --stages judge metrics` (which runs the GPT-5 judge and writes
metrics/stage1_overall.csv + metrics/stage1_by_category.csv).

Nothing in benchmarks/ is modified; run_replication is only imported for its pure parser
/ CSV writer / path helpers (so the extraction CSVs are identical to a native run) and
invoked as a subprocess for the judge pass.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def _find_benchmark_dir():
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "benchmarks" / "omnitom-benchmark"
        if (cand / "run_replication.py").exists():
            return cand
    raise SystemExit("Could not locate benchmarks/omnitom-benchmark/run_replication.py")


BENCH_DIR = _find_benchmark_dir()
sys.path.insert(0, str(BENCH_DIR))
import run_replication as rr  # noqa: E402  (imported after sys.path is set)


def _resolve_samples_file(samples):
    """Accept either a samples_*.jsonl file or the --output_path directory/tree."""
    p = Path(samples)
    if p.is_file():
        return p
    # lm-eval writes samples_<task>_<timestamp>.jsonl under <output_path>/<model>/
    patterns = [
        str(p / "**" / "samples_omnitom_extract_*.jsonl"),
        str(p / "**" / "samples_omnitom_extract_*.json"),
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))
    if not hits:
        raise SystemExit(f"No omnitom_extract samples file found under {samples}")
    # newest by mtime
    return Path(max(hits, key=lambda h: os.path.getmtime(h)))


def _iter_samples(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        content = handle.read()
    stripped = content.lstrip()
    if stripped.startswith("["):  # a JSON array
        for obj in json.loads(content):
            yield obj
        return
    for line in content.splitlines():  # JSONL
        line = line.strip()
        if line:
            yield json.loads(line)


def _generation_of(sample):
    """Pull the model's generated string out of an lm-eval sample record."""
    if sample.get("filtered_resps"):
        r = sample["filtered_resps"][0]
        return r[0] if isinstance(r, list) else r
    if sample.get("resps"):
        r = sample["resps"][0]
        while isinstance(r, list):
            r = r[0]
        return r
    raise KeyError("sample has neither filtered_resps nor resps")


def _story_id_of(sample):
    doc = sample.get("doc", {})
    if "story_id" in doc:
        return int(doc["story_id"])
    raise KeyError("sample doc has no story_id")


def main():
    parser = argparse.ArgumentParser(description="Offline GPT-5 judge scorer for OmniToM Stage 1.")
    parser.add_argument("--samples", required=True,
                        help="lm-eval --log_samples file (samples_omnitom_extract_*.jsonl) OR the "
                             "--output_path directory to search.")
    parser.add_argument("--output-dir", required=True,
                        help="Where the repo's extraction/, judge/, metrics/ trees are written.")
    parser.add_argument("--dataset-path", default=str(BENCH_DIR / "benchmark_story_belief_labels.jsonl"),
                        help="benchmark_story_belief_labels.jsonl (default: the submodule copy).")
    parser.add_argument("--judge-model", default="gpt-5", help="Semantic judge model id (default gpt-5).")
    parser.add_argument("--judge-fewshots", type=int, default=3, help="Judge few-shots (paper default 3).")
    parser.add_argument("--api-provider", default="openai", help="run_replication --api-provider.")
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Materialize extraction CSVs but skip the GPT-5 judge subprocess.")
    args = parser.parse_args()

    samples_file = _resolve_samples_file(args.samples)
    output_dir = Path(args.output_dir)
    print(f"[omnitom] reading generations from: {samples_file}")

    story_ids = []
    for sample in _iter_samples(samples_file):
        sid = _story_id_of(sample)
        rows = rr.parse_extraction_rows(_generation_of(sample))
        rr.write_rows_csv(rr.stage_paths(output_dir, sid)["extract_csv"], rr.EXTRACTION_COLUMNS, rows)
        story_ids.append(sid)

    story_ids = sorted(set(story_ids))
    if not story_ids:
        raise SystemExit("No samples parsed -- nothing to score.")
    print(f"[omnitom] wrote extraction CSVs for {len(story_ids)} stories -> {output_dir / 'extraction' / 'csv'}")

    if args.dry_run:
        print("[omnitom] --dry-run: skipping GPT-5 judge. Extraction CSVs are ready.")
        return 0

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")):
        raise SystemExit("OPENAI_API_KEY is not set -- the GPT-5 judge cannot run. "
                         "Set it (or use --dry-run to only materialize the extraction CSVs).")

    cmd = [
        sys.executable, str(BENCH_DIR / "run_replication.py"),
        "--dataset-path", args.dataset_path,
        "--backend", "api",
        "--api-provider", args.api_provider,
        "--judge-model", args.judge_model,
        "--judge-fewshots", str(args.judge_fewshots),
        "--stages", "judge", "metrics",
        "--output-dir", str(output_dir),
        "--story-ids", ",".join(str(s) for s in story_ids),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
    ]
    print("[omnitom] running canonical judge+metrics:\n  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BENCH_DIR))
    if result.returncode != 0:
        raise SystemExit(f"run_replication.py exited {result.returncode}")

    metrics_dir = output_dir / "metrics"
    print(f"[omnitom] Stage-1 overall:     {metrics_dir / 'stage1_overall.csv'}")
    print(f"[omnitom] Stage-1 by category: {metrics_dir / 'stage1_by_category.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
