# MindGames Faithfulness Audit

Audit date: 2026-07-02

Scope: local lm-evaluation-harness task at `lm_eval/tasks/mindgames`, compared against the MindGames paper, the upstream `llm-theory-of-mind` repository, and the public Hugging Face dataset.

This document is intentionally adversarial. It records faithfulness deviations, reproducibility risks, and non-issues that were checked so the same questions do not need to be re-litigated later.

## Source References

- Local task config: `lm_eval/tasks/mindgames/mindgames.yaml`
- Local task utilities: `lm_eval/tasks/mindgames/utils.py`
- Paper: [MindGames: Targeting Theory of Mind in Large Language Models with Dynamic Epistemic Modal Logic](https://aclanthology.org/2023.findings-emnlp.303/) by Sileo and Lernould, Findings of EMNLP 2023.
- Paper PDF: <https://aclanthology.org/2023.findings-emnlp.303.pdf>
- Upstream code: <https://github.com/sileod/llm-theory-of-mind>
- Public dataset: <https://huggingface.co/datasets/sileod/mindgames>
- Hugging Face dataset-server split metadata queried during audit:
  - `https://datasets-server.huggingface.co/splits?dataset=sileod%2Fmindgames`
  - `https://datasets-server.huggingface.co/statistics?dataset=sileod%2Fmindgames&config=default&split=test`
  - `https://datasets-server.huggingface.co/statistics?dataset=sileod%2Fmindgames&config=default&split=validation`
  - `https://datasets-server.huggingface.co/statistics?dataset=sileod%2Fmindgames&config=default&split=train`

## Local Implementation Snapshot

The local adaptation is very small:

```yaml
task: mindgames
dataset_path: sileod/mindgames
dataset_name: null
test_split: test
output_type: multiple_choice
doc_to_text: |
  {{premise}}
  Hypothesis: {{hypothesis}}
  Question: Does the premise entail the hypothesis? Answer True or False.
  Answer:
doc_to_choice: ["False", "True"]
doc_to_target: label
process_docs: !function utils.process_label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

The label mapping is:

```python
label_map = {"not_entailment": 0, "entailment": 1}
```

With `doc_to_choice: ["False", "True"]`, this means:

- `not_entailment` is scored as `False`
- `entailment` is scored as `True`

That mapping is internally consistent.

## Paper Reference Behavior

The relevant paper claims are in sections 4.2 and 4.3:

- The authors trained a DeBERTa-small shortcut detector on the generated corpus and included its predictions/confidence as metadata.
- They used approximately 11.2k training examples and approximately 3.73k validation/test examples in the public dataset.
- For the reported MindGames benchmark, they did not simply evaluate the entire public `test` split.
- They limited the number of agents to 3, deduplicated, and undersampled to produce 400 test cases.
- The 400-case benchmark is balanced by True/False labels per setup.
- Reported figures are setup-level, not just one global average.
- Experiments include 0-shot and 5-shot settings.
- The paper says lm-eval-harness measured multiple-choice perplexity with an NLI-style prompt equivalent to:

```text
<PREMISE>
Question: <HYPOTHESIS> True or False?
```

with two continuations: `True` and `False`.

The paper figures split results by four setups:

- Forehead-mud-mirror
- Thirst
- Forehead-mud
- Explicit

The upstream generator also defines the relevant setup families in `src/dataset_generator.py`:

- `forehead`
- `forehead_mirror`
- `internal`
- `explicit`

The public HF dataset exposes these setup labels as `forehead`, `forehead_mirror`, `internal`, and `explicit`.

## Public Dataset Snapshot

Dataset-server statistics collected on 2026-07-02:

| Split | Rows |
| --- | ---: |
| train | 11,174 |
| validation | 3,725 |
| test | 3,725 |

Current public `test` split distribution:

| Field | Distribution |
| --- | --- |
| rows | 3,725 |
| label | `entailment`: 1,823; `not_entailment`: 1,902 |
| setup | `forehead`: 932; `forehead_mirror`: 921; `internal`: 910; `explicit`: 962 |
| setup-label (`s-l`) | `forehead-0`: 490; `forehead-1`: 442; `forehead_mirror-0`: 453; `forehead_mirror-1`: 468; `internal-0`: 458; `internal-1`: 452; `explicit-0`: 501; `explicit-1`: 461 |
| n_agents | 2 agents: 586; 3 agents: 1,358; 4 agents: 1,781 |
| DeBERTa confidence | mean 0.95978; median 0.99813 |
| difficulty | mean 0.152; median 0.00187 |

This public `test` split is the full generated/test corpus, not the paper's 400-case reported benchmark subset.

## Findings

### 1. Critical: The local task evaluates the raw HF `test` split, not the paper's 400-case MindGames benchmark.

Local evidence:

- `mindgames.yaml` sets `dataset_path: sileod/mindgames`.
- `mindgames.yaml` sets `test_split: test`.
- There is no `process_docs` filtering beyond label conversion.

Reference behavior:

- The paper's named MindGames benchmark is a 400-case evaluation set.
- It is produced by limiting to at most 3 agents, deduplicating, and undersampling.
- It is balanced per setup and per True/False label.

Observed mismatch:

- Local evaluation uses 3,725 test examples.
- 1,781 of those examples have 4 agents.
- The paper's reported benchmark explicitly limits to 3 agents.
- Local evaluation is not balanced per setup-label:
  - `forehead`: 490 false / 442 true
  - `forehead_mirror`: 453 false / 468 true
  - `internal`: 458 false / 452 true
  - `explicit`: 501 false / 461 true
- The paper's 400-case set should imply 100 examples per setup and, if perfectly balanced per setup, 50 true and 50 false per setup.

Why it matters:

- Any score from this local task is not comparable to the paper's figures.
- The raw public split changes both sample size and difficulty distribution.
- The raw public split includes many 4-agent problems, which the paper's reported benchmark excludes.
- The raw public split has a large number of examples that DeBERTa-small predicts with very high confidence, which changes the shortcut/difficulty profile.

Severity: Critical. This is the central faithfulness failure.

### 2. Critical: The local task cannot reproduce setup-level paper results.

Local evidence:

- The local YAML defines a single task, `mindgames`.
- The only metric is global `acc`.
- There are no subtasks or filters for `setup`.
- There is no `process_results` that emits setup-specific metrics.

Reference behavior:

- The paper figures report separate accuracy curves for each setup:
  - Forehead-mud-mirror
  - Thirst
  - Forehead-mud
  - Explicit
- The paper compares 0-shot, 5-shot, and human performance at setup granularity.

Observed mismatch:

- Local runs only produce one aggregate accuracy number.
- Even if the local task used the correct 400 rows, the result table would still not match the paper's reporting structure.
- On the raw HF test split, the aggregate is also weighted by the raw setup frequencies, not by an intended balanced 100-per-setup design.

Why it matters:

- Aggregating hides setup-specific weaknesses, which are the main analysis unit in the paper.
- Model comparisons against Figure 1 or Figure 2 cannot be reconstructed from one global `acc`.
- A model can improve on one setup and regress on another while the local metric hides it.

Severity: Critical for reproduction of the paper's reported experiments.

### 3. Critical for 5-shot: The task has no few-shot split, so few-shot evaluation samples from the test set.

Local evidence:

- `mindgames.yaml` does not define `training_split`.
- `mindgames.yaml` does not define `validation_split`.
- `mindgames.yaml` does not define `fewshot_split`.

Harness behavior:

- In the base task fallback, if a task has no training or validation docs, `fewshot_docs()` falls back to `test_docs()`.
- `ConfigurableTask.fewshot_docs()` delegates to that fallback when `fewshot_split` is `None`.
- The sampler is initialized from those fallback docs.
- During few-shot construction, the evaluated document is only passed for exclusion when `fewshot_cfg.split == config.test_split`.
- Here `fewshot_cfg.split` is `None`, so that exclusion path is not activated.

Observed mismatch:

- Running this task with `--num_fewshot 5` samples labeled examples from the evaluation test split.
- The current evaluated item can also appear in its own few-shot context because the exclusion condition is not triggered.
- With a 3,725-row test split and 5 sampled examples, the expected number of self-inclusions across a complete evaluation is about 5. That is not the largest issue; the larger issue is that all few-shot examples are drawn from the evaluation distribution.

Reference behavior:

- The paper reports 5-shot experiments, but those must not leak test labels into the prompt.
- The paper describes separate train, validation, and test resources.

Why it matters:

- 5-shot local results are contaminated.
- Any 5-shot score is inflated or at least not comparable.
- This is a silent failure mode. The YAML looks like a 0-shot task, but harness users commonly run `--num_fewshot 5` for paper reproduction.

Severity: Critical for 5-shot reproduction.

Note:

- Few-shot answer rendering itself appears likely correct. Because `doc_to_target` becomes an integer label and `doc_to_choice` is available, harness few-shot rendering should convert labels to `False`/`True`, not literal `0`/`1`.
- The problem is the source split and self-exclusion behavior, not answer text rendering.

### 4. Major: The prompt is not the paper's NLI prompt.

Local prompt:

```text
{{premise}}
Hypothesis: {{hypothesis}}
Question: Does the premise entail the hypothesis? Answer True or False.
Answer:
```

Paper prompt shape:

```text
<PREMISE>
Question: <HYPOTHESIS> True or False?
```

Nearby local precedent:

- `lm_eval/tasks/super_glue/rte/default.yaml` uses the closer template:

```yaml
doc_to_text: "{{premise}}\nQuestion: {{hypothesis}} True or False?\nAnswer:"
doc_to_choice: ['True', 'False']
```

Observed mismatch:

- The local prompt introduces a separate `Hypothesis:` line.
- The local prompt asks an explicit metalinguistic entailment question.
- The paper prompt places the hypothesis directly after `Question:`.
- The local prompt includes the instruction sentence `Does the premise entail the hypothesis? Answer True or False.`

Why it matters:

- MindGames is evaluated through language-model continuation likelihood, so prompt wording is part of the benchmark.
- The local prompt can change model priors and token likelihoods independently of reasoning ability.
- The paper explicitly names an NLI-style prompt from Brown et al. and describes its shape. The local prompt is semantically close but not faithful.

Severity: Major.

### 5. Major: The local task does not reproduce the paper's 0-shot/5-shot comparison contract.

Local evidence:

- No `num_fewshot` metadata is set.
- No few-shot split is defined.
- No task aliases distinguish 0-shot and 5-shot variants.

Reference behavior:

- The paper reports both 0-shot and 5-shot results for Pythia and GPT-3 families.
- The paper notes that increasing the number of examples beyond the chosen few-shot setting did not improve validation accuracy.

Observed mismatch:

- A 0-shot local run is possible, but it uses the wrong dataset and prompt.
- A 5-shot local run is possible only through generic harness flags, but it leaks from the test split.
- There is no task-level guardrail to stop invalid 5-shot runs.

Why it matters:

- Users trying to reproduce Figure 1 or Figure 2 will likely run `--num_fewshot 5` and get contaminated results.
- The local task name `mindgames` gives no warning that it is only a rough 0-shot raw-split adapter.

Severity: Major.

### 6. Medium: Choice order differs from the paper wording and nearby lm-eval RTE precedent.

Local implementation:

```yaml
doc_to_choice: ["False", "True"]
```

Paper wording:

- The paper describes the two possible continuations as `True` and `False`.

Nearby lm-eval RTE precedent:

```yaml
doc_to_choice: ['True', 'False']
```

Assessment:

- The local order is internally consistent with the local label mapping.
- In normal loglikelihood multiple-choice scoring, ordering should not affect the predicted class except in exact ties.
- Exact ties are rare, but `np.argmax` breaks ties by choosing the first choice. Therefore, the local task would break exact ties toward `False`, while a `['True', 'False']` task would break ties toward `True`.

Why it matters:

- This is not a main source of score drift.
- It is still a small faithfulness difference against the described continuation order and existing RTE template.

Severity: Medium to low.

### 7. Medium: The task is not pinned to a dataset revision.

Local evidence:

- `dataset_path: sileod/mindgames`
- No `dataset_kwargs` revision is specified.

Observed risk:

- Hugging Face datasets can change.
- The local task always resolves whatever `sileod/mindgames` means at evaluation time.

Why it matters:

- A benchmark adapter meant for reproducibility should ideally pin the dataset revision or commit hash.
- This is especially important here because the paper's exact 400-case subset is already not encoded locally.

Severity: Medium for long-term reproducibility. It is not the primary present-day mismatch.

### 8. Medium: The local task lacks benchmark metadata and provenance warnings.

Local evidence:

- No README in `lm_eval/tasks/mindgames`.
- No `metadata.version`.
- No comments warning that the task is raw HF `test`, not the paper's 400-case benchmark.

Why it matters:

- Future users will reasonably assume `--tasks mindgames` means the paper's MindGames benchmark.
- The task name does not signal "raw split" or "unfaithful/simple adapter".
- A paper reproduction task should document subset construction and prompt choices.

Severity: Medium.

### 9. Low: Label conversion mutates labels from strings to ints, but this is appropriate for the local multiple-choice setup.

Local evidence:

```python
label_map = {"not_entailment": 0, "entailment": 1}
```

Assessment:

- This is correct given `doc_to_choice: ["False", "True"]`.
- Harness multiple-choice scoring accepts integer targets indexing into `doc_to_choice`.
- There is no evidence that this reverses labels.

Why it matters:

- This should not be treated as a bug.
- If the choice order is changed later to `["True", "False"]`, this mapping must also be changed.

Severity: Low / non-issue in current config.

## Non-Issues Checked

### Multiple-choice scoring mode

The local task uses:

```yaml
output_type: multiple_choice
metric_list:
  - metric: acc
```

This is broadly consistent with the paper's description of lm-eval-harness measuring which continuation the model favors by likelihood. The problem is not the use of multiple choice itself. The problem is the dataset selection, prompt mismatch, setup aggregation, and few-shot split behavior.

### Entailment to True/False conversion

The paper frames the task as deciding whether the premise entails the hypothesis. The public dataset labels are `entailment` and `not_entailment`. The local conversion to `True` and `False` is conceptually correct.

### Generated public dataset fields

The public dataset has enough metadata to support a more faithful adapter:

- `setup`
- `s-l`
- `n_agents`
- `label`
- `deberta_pred`
- `deberta_confidence`
- `difficulty`
- `index`

The issue is that the local task does not use that metadata to reconstruct the paper benchmark.

## Expected Shape of a Faithful Adapter

This is not an implementation plan, but it records what a faithful adapter would need to do.

### Dataset selection

A faithful reproduction must identify or reconstruct the paper's 400-case benchmark subset:

- Keep only examples with `n_agents <= 3`.
- Deduplicate in the same way the paper did.
- Undersample to 400 examples.
- Balance by setup and True/False label.
- Preserve deterministic row selection, ideally by using exact row IDs or `index` values rather than a fresh random sample.

Open question:

- The exact 400 row IDs used in the paper are not present in the local adapter. If they are not available upstream, any reconstruction will be only an approximation unless a deterministic selection rule can be recovered.

### Task granularity

A faithful adapter should expose setup-level metrics or setup-level subtasks:

- `mindgames_forehead`
- `mindgames_forehead_mirror`
- `mindgames_internal` or `mindgames_thirst`
- `mindgames_explicit`
- an aggregate group only after setup-level metrics are available

The aggregate should be a macro average over setup subtasks if the paper's balanced design is intended.

### Prompt

The prompt should match the paper's NLI-style format as closely as possible:

```text
{{premise}}
Question: {{hypothesis}} True or False?
Answer:
```

The choice list and label mapping must be kept consistent. If choices are ordered as `["True", "False"]`, labels should map `entailment -> 0` and `not_entailment -> 1`.

### Few-shot behavior

A faithful 5-shot task must explicitly define a non-test few-shot source:

- likely `train`, or a paper-specified few-shot pool if recoverable
- never implicit fallback to `test`

If the task is intended to be 0-shot only, it should either document that clearly or guard against accidental few-shot runs.

### Provenance

A faithful adapter should include:

- paper citation
- dataset revision
- subset construction details
- expected row counts
- expected setup-label counts
- a warning if exact paper subset IDs are unavailable

## Severity Summary

| Severity | Finding |
| --- | --- |
| Critical | Uses raw 3,725-row HF `test` split instead of the 400-case paper benchmark |
| Critical | Cannot reproduce setup-level paper results |
| Critical | 5-shot runs sample from the test set and may self-include |
| Major | Prompt differs from the paper's NLI prompt |
| Major | Does not reproduce the paper's 0-shot/5-shot comparison contract |
| Medium | Choice order differs from paper wording and RTE precedent; affects exact ties |
| Medium | Dataset revision is not pinned |
| Medium | Missing metadata/provenance warnings |
| Low | Label mapping is internally correct and should not be treated as a bug |

## Bottom Line

The current `mindgames` task is a simple, usable multiple-choice adapter over the public `sileod/mindgames` raw test split. It is not a faithful reproduction of the MindGames benchmark as reported in the paper.

The most serious deviation is dataset selection: the local task evaluates 3,725 raw test rows, including 4-agent examples, while the paper's reported benchmark is a balanced 400-case subset limited to at most 3 agents. The second most serious deviation is few-shot behavior: any 5-shot local run is contaminated because examples are drawn from the test split. The third major deviation is prompt wording, which differs from the paper's stated NLI-style lm-eval prompt.

