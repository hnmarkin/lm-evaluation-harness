"""Check and shrink a ToMBench `--log_samples` file.

ToMBench scores cross-item (majority vote over shuffled repeats, then a nested macro
roll-up), so no per-doc scalar exists for any of the 57 metrics. `process_results` therefore
emits the *same* payload dict under all 57 metric names -- each aggregation needs the whole
payload set. lm-eval's `evaluator.py` then does `example.update(metrics)`, so every logged
sample carries 57 identical copies of that payload. Measured on `tombench_zh` (14,300 docs):
~143 MB of payload copies against ~14 MB of actual docs, i.e. ~90% of the file.

This script does two things:

  CHECK   assert that all 57 metric values in a sample really are identical. That is an
          invariant of the FANToM pattern, not a coincidence -- if a future
          `process_results` ever emits per-metric values, the aggregations silently start
          reading different payload sets. This is the only place that would notice.

  DEDUPE  rewrite each sample with a single `tombench_payload` key instead of 57, leaving
          `metrics` (the name list) intact so the file stays self-describing.

Usage (from anywhere):

    python dedupe_samples.py results/**/samples_tombench_en_*.jsonl
    python dedupe_samples.py samples_tombench_zh.jsonl --in-place
    python dedupe_samples.py samples_tombench_zh.jsonl --check-only
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402

PAYLOAD_KEY = "tombench_payload"


def _process_line(line, lineno, path, check_only):
    """Return (sample, status) where status is "rewritten", "checked", or "already"."""
    sample = json.loads(line)

    present = [m for m in U.METRIC_NAMES if m in sample]
    if not present:
        if PAYLOAD_KEY in sample:
            return sample, "already"
        raise ValueError(f"{path}:{lineno}: no ToMBench metric keys -- not a tombench samples file?")
    if len(present) != len(U.METRIC_NAMES):
        missing = set(U.METRIC_NAMES) - set(present)
        raise ValueError(f"{path}:{lineno}: {len(missing)} metric keys missing, e.g. {sorted(missing)[:3]}")

    payload = sample[present[0]]
    for name in present[1:]:
        if sample[name] != payload:
            raise ValueError(
                f"{path}:{lineno}: metric {name!r} carries a different payload than "
                f"{present[0]!r}. process_results is no longer emitting one shared payload; "
                f"the 57 aggregations are now reading different data."
            )

    if check_only:
        return sample, "checked"
    for name in present:
        del sample[name]
    sample[PAYLOAD_KEY] = payload
    return sample, "rewritten"


def process_file(path, in_place, check_only):
    path = Path(path)
    before = path.stat().st_size
    out_lines = []
    counts = {"rewritten": 0, "checked": 0, "already": 0}

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            sample, status = _process_line(line, lineno, path, check_only)
            counts[status] += 1
            out_lines.append(json.dumps(sample, ensure_ascii=False))

    n = len(out_lines)
    if check_only:
        if counts["already"] == n:
            print(f"[check] {path.name}: {n} samples, already deduped (nothing to verify)")
        else:
            print(f"[check] {path.name}: {counts['checked']} samples verified, all 57 payloads identical"
                  + (f"; {counts['already']} already deduped" if counts["already"] else ""))
        return

    if counts["already"] == n:
        print(f"[dedupe] {path.name}: {n} samples, already deduped -- nothing written")
        return

    target = path if in_place else path.with_suffix(".deduped.jsonl")
    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    after = target.stat().st_size
    shrink = f"{before / after:.1f}x" if after else "inf"
    print(
        f"[dedupe] {path.name}: {counts['rewritten']} samples rewritten -> "
        f"{target.name} ({before / 1e6:.1f} MB -> {after / 1e6:.1f} MB, {shrink} smaller)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path, help="samples_*.jsonl file(s)")
    ap.add_argument("--in-place", action="store_true", help="overwrite instead of writing .deduped.jsonl")
    ap.add_argument("--check-only", action="store_true", help="verify the payloads match; write nothing")
    args = ap.parse_args()

    for path in args.paths:
        try:
            process_file(path, args.in_place, args.check_only)
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
