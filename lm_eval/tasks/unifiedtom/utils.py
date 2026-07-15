"""UniToMBench direct-baseline lm-eval adapter utilities.

This adapter targets the static, single-call direct question-answering condition
from `benchmarks/unifiedtombenchmark`. The SimToM condition in the source repo is
a three-call LLM pipeline and is documented as out-of-harness in README.md.
"""

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import datasets


_XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


TOMBENCH_SHEETS = {
    "uot": ("Unexpected Outcome Test", "UOT: Unexpected Outcome Test"),
    "sit": ("Scalar Implicature Test", "SIT: Scalar Implicature Task"),
    "pst": ("Persuasion Story Task", "PST: Persuasion Story Task"),
    "fbt": ("False Belief Task", "FBT: False Belief Task"),
    "ast": ("Ambiguous Story Task", "AST: Ambiguous Story Task"),
    "ht": ("Hinting Task Test", "HT: Hinting Test"),
    "sst": ("Strange Story Task", "SST: Strange Story Task"),
    "frt": ("Faux-pas Recognition Test", "FRT: Faux-pas Recognition Test"),
}

CUSTOM_FILES = {
    "evolving_stories": ("evolving_stories_250.xlsx", "Evolving Stories"),
    "multi_interaction": ("multi_interaction_100.xlsx", "Multi-Interaction Tasks"),
}

SUBSET_ORDER = tuple(TOMBENCH_SHEETS) + tuple(CUSTOM_FILES)

_ANSWER_KEYS = ("ANSWER", "\u7b54\u6848\nANSWER", "Answer", "Answers")
_ABILITY_KEYS = ("ABILITY", "\u80fd\u529b\nABILITY")
_INDEX_KEYS = ("INDEX", "\u5e8f\u53f7\nINDEX")
_CACHE = {}


def _benchmark_dir():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks" / "unifiedtombenchmark"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("benchmarks/unifiedtombenchmark not found under repo root")


def _clean(value):
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def _col_index(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return idx - 1


def _shared_strings(zf):
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for si in root.findall("a:si", _XLSX_NS):
        values.append("".join(t.text or "" for t in si.findall(".//a:t", _XLSX_NS)))
    return values


def _sheet_targets(zf):
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", _REL_NS)
    }
    targets = {}
    for sheet in workbook.findall("a:sheets/a:sheet", _XLSX_NS):
        rid = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rid_to_target[rid]
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        targets[sheet.attrib["name"]] = target
    return targets


def _cell_value(cell, shared):
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", _XLSX_NS))

    value = cell.find("a:v", _XLSX_NS)
    if value is None:
        return ""

    raw = value.text or ""
    if ctype == "s":
        return shared[int(raw)] if raw else ""
    if ctype == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def _read_xlsx_sheet(path, sheet_name):
    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        targets = _sheet_targets(zf)
        if sheet_name not in targets:
            raise KeyError(f"{path.name} has no sheet {sheet_name!r}")
        root = ET.fromstring(zf.read(targets[sheet_name]))

        rows = []
        for row in root.findall(".//a:sheetData/a:row", _XLSX_NS):
            values = []
            for cell in row.findall("a:c", _XLSX_NS):
                idx = _col_index(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                values[idx] = _clean(_cell_value(cell, shared))
            rows.append(values)

    if not rows:
        return []

    headers = [_clean(h) for h in rows[0]]
    records = []
    for row in rows[1:]:
        record = {}
        for idx, header in enumerate(headers):
            if not header:
                continue
            record[header] = row[idx] if idx < len(row) else ""
        if any(record.values()):
            records.append(record)
    return records


def _first(record, *keys):
    for key in keys:
        if key in record and _clean(record[key]):
            return _clean(record[key])
    raise KeyError(f"none of {keys!r} present in row keys {sorted(record)}")


def _optional(record, key):
    value = _clean(record.get(key, ""))
    if not value or value.lower() == "nan":
        return ""
    return value


def _gold_letter(raw):
    value = _clean(raw).upper().replace(".", "")
    if value in "ABCD" and len(value) == 1:
        return value
    for ch in value:
        if ch in "ABCD":
            return ch
    raise ValueError(f"cannot recover A-D gold from {raw!r}")


def _format_tombench_prompt(story, question, choices):
    parts = [f"Option {letter}: {choice}" for letter, choice in zip("ABCD", choices)]
    options = ", ".join(parts)
    return (
        f"Story: {story}.  Question: {question}. {options}. "
        "reply only with the option. for ex: D"
    )


def _parse_custom_options(raw_options):
    try:
        parsed = ast.literal_eval(raw_options)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        choices = []
        for letter in "ABCD":
            if letter in parsed and _clean(parsed[letter]):
                choices.append(_clean(parsed[letter]))
        return choices

    # The README-canonical custom files are Python-dict-ish, but some values
    # contain unescaped apostrophes ("doesn't"), so ast.literal_eval can fail.
    # Split on the explicit quoted option keys instead of on value quotes.
    markers = list(re.finditer(r"'([A-D])'\s*:\s*'", raw_options))
    if markers:
        choices = []
        for idx, marker in enumerate(markers):
            start = marker.end()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(raw_options)
            value = raw_options[start:end].strip()
            value = value.rstrip("}").rstrip().rstrip(",").rstrip()
            value = value[:-1] if value.endswith("'") else value
            choices.append(_clean(value))
        return choices

    # Some non-canonical shipped snapshots use "A) ... B) ..." strings. The
    # public leaves do not target those files, but this keeps load("all") honest.
    matches = list(re.finditer(r"([A-D])\)\s*", raw_options))
    if matches:
        choices = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_options)
            choices.append(_clean(raw_options[start:end]))
        return choices

    raise ValueError(f"cannot parse custom Options field: {raw_options!r}")


def _format_custom_prompt(scenario, question, raw_options):
    return (
        f"Story: {scenario}.  Question: {question}. Options: {raw_options} "
        "reply only with the option. for ex: D"
    )


def _collapse_custom_prompt_duplicates(docs):
    """Match the custom source evaluators' ``questions[prompt] = answer`` maps.

    Normal dictionary insertion preserves the first row's prompt position while a
    later duplicate overwrites its gold answer.  We retain the first row's public
    ``id``/``index`` metadata and record both source row numbers for auditability;
    row metadata never enters prompting or scoring.
    """
    by_prompt = {}
    for doc in docs:
        prompt = doc["prompt"]
        if prompt in by_prompt:
            existing = by_prompt[prompt]
            existing["target"] = doc["target"]
            existing["source_row_last"] = doc["source_row_last"]
        else:
            by_prompt[prompt] = doc
    return list(by_prompt.values())


def _load_tombench_subset(subset):
    sheet_name, display_name = TOMBENCH_SHEETS[subset]
    path = _benchmark_dir() / "datasets" / "ToMBench_release_v1_0618.xlsx"
    docs = []
    for row_idx, row in enumerate(_read_xlsx_sheet(path, sheet_name), start=2):
        story = _first(row, "STORY")
        question = _first(row, "QUESTION")
        raw_choices = [_optional(row, f"OPTION-{letter}") for letter in "ABCD"]
        present = [choice for choice in raw_choices if choice]
        if len(present) < 2:
            raise ValueError(f"{sheet_name} row {row_idx} has fewer than two choices")
        gold = _gold_letter(_first(row, *_ANSWER_KEYS))
        if "ABCD".index(gold) >= len(present):
            raise ValueError(f"{sheet_name} row {row_idx} gold {gold} is outside choices")
        prompt = _format_tombench_prompt(story, question, present)
        arity = len(present)
        docs.append(
            {
                "id": f"{subset}:{row_idx}",
                "source_file": path.name,
                "sheet": sheet_name,
                "subset": subset,
                "subset_name": display_name,
                "ability": _first(row, *_ABILITY_KEYS) if any(k in row for k in _ABILITY_KEYS) else "",
                "index": _first(row, *_INDEX_KEYS) if any(k in row for k in _INDEX_KEYS) else "",
                "prompt": prompt,
                "target": gold,
                "arity": arity,
            }
        )
    return docs


def _load_custom_subset(subset):
    filename, display_name = CUSTOM_FILES[subset]
    path = _benchmark_dir() / "datasets" / filename
    docs = []
    for row_idx, row in enumerate(_read_xlsx_sheet(path, "Sheet1"), start=2):
        scenario = _first(row, "Scenario")
        question = _first(row, "Question")
        raw_options = _first(row, "Options")
        choices = _parse_custom_options(raw_options)
        if len(choices) < 2:
            raise ValueError(f"{filename} row {row_idx} has fewer than two choices")
        gold = _gold_letter(_first(row, *_ANSWER_KEYS))
        if "ABCD".index(gold) >= len(choices):
            raise ValueError(f"{filename} row {row_idx} gold {gold} is outside choices")
        docs.append(
            {
                "id": f"{subset}:{row_idx}",
                "source_file": path.name,
                "sheet": "Sheet1",
                "subset": subset,
                "subset_name": display_name,
                "ability": "",
                "index": str(row_idx - 1),
                "prompt": _format_custom_prompt(scenario, question, raw_options),
                "target": gold,
                "arity": len(choices),
                "source_row_first": row_idx,
                "source_row_last": row_idx,
            }
        )
    return _collapse_custom_prompt_duplicates(docs)


def load(subset="all", **kwargs):
    if subset == "all":
        subsets = SUBSET_ORDER
    else:
        if subset not in SUBSET_ORDER:
            raise ValueError(f"unknown UnifiedToM subset {subset!r}")
        subsets = (subset,)

    cache_key = tuple(subsets)
    if cache_key not in _CACHE:
        docs = []
        for name in subsets:
            if name in TOMBENCH_SHEETS:
                docs.extend(_load_tombench_subset(name))
            else:
                docs.extend(_load_custom_subset(name))
        _CACHE[cache_key] = datasets.Dataset.from_list(docs)
    return {"train": _CACHE[cache_key]}


def process_results(doc, results):
    # The source scripts use exact string equality with only light strip/case/dot
    # variants. Do not regex-extract letters out of verbose answers here.
    response = str(results[0]).strip().replace(".", "").upper()
    return {"acc": 1.0 if response == doc["target"] else 0.0}
