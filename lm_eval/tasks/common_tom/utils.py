import csv
import gzip
import re
from pathlib import Path

import datasets


_TEST_CID = "4431"
_WINDOW_SIZE = 5
_STOP_SIGN = "\U0001f6d1"
_YES_NO = {"yes": "Yes", "no": "No"}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "benchmarks" / "common-tom").exists():
            return parent
    raise FileNotFoundError("Could not find benchmarks/common-tom from task path")


def _benchmark_dir() -> Path:
    return _repo_root() / "benchmarks" / "common-tom"


def _prompt_template() -> str:
    path = _benchmark_dir() / "data" / "prompts" / "gpt-zero-shot"
    return path.read_text(encoding="utf-8").strip()


def _question_files(cid: str | None) -> list[Path]:
    qdir = _benchmark_dir() / "data" / "questions"
    if cid is None:
        files = sorted(qdir.glob("*.csv.gz"))
    else:
        files = sorted(qdir.glob(f"{cid}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No Common-ToM question files found for cid={cid!r}")
    return files


def _context_window(context: str, sno: int, window_size: int) -> str:
    lines = context.split("\n")
    marker_line = lines[sno - 1]
    if not marker_line.endswith(_STOP_SIGN):
        raise ValueError(f"Expected stop-sign marker at sno={sno}, got {marker_line!r}")
    start = max(0, sno - 1 - window_size)
    end = sno + window_size
    window = lines[start:end]
    if len(window) > (window_size * 2) + 1:
        raise ValueError("Common-ToM context window exceeded expected size")
    return "\n".join(window)


def _read_rows(cid: str | None) -> list[dict]:
    rows = []
    for path in _question_files(cid):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def load(cid=_TEST_CID, order=None, window_size=_WINDOW_SIZE, **kwargs):
    """Load the paper's zero-shot evaluation split.

    Common-ToM's paper uses CID 4431 as the held-out test conversation for prompting.
    The prompt window is reconstructed exactly as bin/openai_zero_shot.py does:
    five utterances before and after the stop-sign-marked utterance.
    """
    if cid == "all":
        cid = None
    else:
        cid = str(cid)
    order = None if order is None else str(order)
    window_size = int(window_size)
    template = _prompt_template()

    docs = []
    for idx, row in enumerate(_read_rows(cid)):
        if order is not None and row["order"] != order:
            continue
        context_window = _context_window(row["context"], int(row["sno"]), window_size)
        doc = dict(row)
        doc["context_window"] = context_window
        doc["prompt"] = template.format(context=context_window, question=row["question"])
        doc["doc_id"] = f"{row['cid']}:{idx}"
        docs.append(doc)

    if not docs:
        raise ValueError(f"Common-ToM loader produced no docs for cid={cid!r}, order={order!r}")
    return {"train": datasets.Dataset.from_list(docs)}


def _extract_yes_no(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return _YES_NO[match.group(1).lower()]


def process_results(doc, results):
    pred = _extract_yes_no(results[0])
    return {"acc": 1.0 if pred == doc["answer"] else 0.0}
