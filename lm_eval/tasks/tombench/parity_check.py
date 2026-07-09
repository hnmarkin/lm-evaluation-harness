"""Parity check: this adapter vs ToMBench's OWN code.

Imports `benchmarks/ToMBench/{prompts,run_api,get_results}.py` with their unused heavy deps
stubbed (openai / tqdm), then asserts that

  1. every rendered prompt is byte-identical to `run_api.format_prompt_{2,4}`,
  2. every shuffled->canonical letter map is identical (incl. the duplicate-option
     first-match-wins quirk),
  3. `extract_answer` / `most_common_element` are behaviourally identical,
  4. each task YAML's `description:` is byte-identical to the matching system prompt.

`benchmarks/` is never modified. Run from the repo root:

    ~/miniconda3/envs/eval_env/python.exe \
        lm-evaluation-harness/lm_eval/tasks/tombench/parity_check.py
"""

import importlib.util
import json
import math
import random
import re
import sys
import types
from pathlib import Path

# `benchmarks/` is strictly read-only: importing run_api.py would otherwise drop a
# __pycache__/ into the submodule and dirty it.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "benchmarks/ToMBench").is_dir())
BENCH = REPO / "benchmarks/ToMBench"

sys.path.insert(0, str(HERE))
import utils as U  # noqa: E402

# --- stub the deps run_api.py imports but does not need for prompt formatting ---
for name in ("openai", "tqdm"):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if name == "tqdm":
            mod.tqdm = lambda x, **k: x
        else:
            mod.api_key = mod.api_base = ""
            mod.ChatCompletion = types.SimpleNamespace(create=lambda **k: None)
        sys.modules[name] = mod

sys.path.insert(0, str(BENCH))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"tb_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prompts = _load("prompts")
run_api = _load("run_api")
get_results = _load("get_results")


class Args:
    def __init__(self, language):
        self.language = language


def _fixed_shuffle(perm):
    """Replace random.shuffle with a deterministic permutation so both sides agree."""

    def _sh(lst):
        vals = list(lst)
        lst[:] = [vals[i] for i in perm]

    return _sh


class _FakeRng:
    def __init__(self, perm):
        self._sh = _fixed_shuffle(perm)

    def shuffle(self, lst):
        self._sh(lst)


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def main():
    rows_by_file = {}
    for path in sorted((BENCH / "data").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows_by_file[path.stem] = [json.loads(l) for l in fh if l.strip()]

    perms = {2: [[0, 1], [1, 0]], 4: [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]]}
    checked = skipped = 0

    for language in ("zh", "en"):
        args = Args(language)
        c_key = "选项C" if language == "zh" else "OPTION-C"
        zh_c = "选项C"
        story_k, question_k, *opt_ks = U._ZH if language == "zh" else U._EN
        t4 = U.UserEvaluatePrompt4Choices_zh if language == "zh" else U.UserEvaluatePrompt4Choices_en
        t2 = U.UserEvaluatePrompt2Choices_zh if language == "zh" else U.UserEvaluatePrompt2Choices_en

        for stem, rows in rows_by_file.items():
            for row_idx, row in enumerate(rows):
                n = 2 if _missing(row[c_key]) else 4
                # run_api gates arity on the CHINESE column regardless of --language.
                # That disagrees with the English column on exactly one corrupt row
                # (Strange Story #292); we gate per-language, so skip it here.
                if _missing(row[zh_c]) != _missing(row[c_key]):
                    skipped += 1
                    continue

                for perm in perms[n]:
                    random.shuffle = _fixed_shuffle(perm)  # patch what run_api sees
                    run_api.random.shuffle = _fixed_shuffle(perm)
                    fn = run_api.format_prompt_4 if n == 4 else run_api.format_prompt_2
                    want_map, want_prompt = fn(row, args)

                    values = [row[k] for k in opt_ks]
                    canon = [U._strip_label(v, L) for v, L in zip(values[:n], "ABCD")]
                    shuffled, got_map = U._shuffle_and_map(canon, _FakeRng(perm))
                    fields = dict(story=row[story_k], question=row[question_k])
                    for L, choice in zip("abcd", shuffled):
                        fields[f"choice_{L}"] = choice
                    got_prompt = (t4 if n == 4 else t2).format(**fields)

                    assert got_prompt == want_prompt, f"PROMPT {language} {stem}#{row_idx} {perm}"
                    assert got_map == want_map, f"MAP {language} {stem}#{row_idx} {perm}: {got_map} != {want_map}"
                    checked += 1

    print(f"[1/2] prompts + de-maps: {checked} renderings byte-identical to run_api.py")
    print(f"      ({skipped} skipped: the 1 zh/en arity-disagreement row, per-language gate)")

    # --- extract_answer / most_common_element ---
    rnd = random.Random(0)
    alphabet = list("ABCD[]() .xyz")
    for _ in range(20000):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 18)))
        assert U.extract_answer(s) == get_results.extract_answer(s), repr(s)
    for _ in range(5000):
        lst = [rnd.choice("ABCD") for _ in range(rnd.randint(1, 7))]
        assert U.most_common_element(lst) == get_results.most_common_element(lst), lst
    print("[2/2] extract_answer (20k strings) + most_common_element (5k lists): identical")

    # --- YAML description == system prompt ---
    sysmap = {
        "tombench_zh": prompts.SystemEvaluatePrompt_zh,
        "tombench_zh_cot": prompts.SystemEvaluatePrompt_zh_cot,
        "tombench_en": prompts.SystemEvaluatePrompt_en,
        "tombench_en_cot": prompts.SystemEvaluatePrompt_en_cot,
    }
    for task, want in sysmap.items():
        for suffix in ("", "_rawgold"):
            text = (HERE / f"{task}{suffix}.yaml").read_text(encoding="utf-8")
            block = re.search(r"^description: \|-\n((?:  .*\n|\n)+)", text, re.M).group(1)
            got = "\n".join(l[2:] for l in block.rstrip("\n").split("\n"))
            assert got == want, f"description mismatch in {task}{suffix}.yaml"
    print("[+]   all 8 task YAML `description:` blocks == prompts.py system prompts")
    print("\nPARITY OK")


if __name__ == "__main__":
    main()
