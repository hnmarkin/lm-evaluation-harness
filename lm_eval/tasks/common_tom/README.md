# Common-ToM (lm-eval adapter)

Faithful lm-eval port of **Common-ToM** from Soubki et al., ACL Findings 2024,
"Views Are My Own, but Also Yours." The benchmark asks Yes/No questions over
natural spoken-dialogue transcripts, with the queried utterance marked by a
stop-sign symbol.

Phase-0 verification corrected the recon dossier's first-pass mapping: the
paper reports **accuracy**, not macro-F1, and the zero-shot protocol is
generative. Appendix C uses the `gpt-zero-shot` prompt, a five-utterance
window before and after the marked utterance, and `temperature=1.0`.

## Tasks

`common_tom` is a group over the three paper-reported belief orders:

| task | order | n |
|---|---:|---:|
| `common_tom_order_1` | 1 | 676 |
| `common_tom_order_2` | 2 | 707 |
| `common_tom_order_3` | 3 | 721 |

The group aggregate is a size-weighted mean `acc`, matching the paper's total
accuracy over the held-out test conversation. The held-out conversation is CID
`4431`, whose No/Yes counts are 1139/965, matching Table 2's test split.

## Architecture

- **Doc unit:** one Common-ToM question from CID `4431`.
- **Prompt:** read verbatim from `benchmarks/common-tom/data/prompts/gpt-zero-shot`
  and filled after reconstructing the same +/-5 utterance window used by
  `bin/openai_zero_shot.py`.
- **Output type:** `generate_until`, because the paper generates Yes/No answers
  rather than ranking fixed strings by loglikelihood.
- **Metric:** `acc`; the extractor maps the first standalone `yes` or `no` in
  the generation to the corresponding label. Missing or unparsable answers are
  scored wrong.
- **Decoding:** sampled with `do_sample: true`, `temperature: 1.0`, and a small
  `max_gen_toks` budget for direct Yes/No answers.

## Usage

```bash
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks common_tom --include_path lm-evaluation-harness/lm_eval/tasks \
  --batch_size auto --output_path outputs/common_tom --log_samples

# one order:
PYTHONIOENCODING=utf-8 lm-eval run --model hf --model_args pretrained=<model> \
  --tasks common_tom_order_1 --include_path lm-evaluation-harness/lm_eval/tasks \
  --limit 5 --log_samples
```

`--limit` is only a plumbing check; sampled decoding makes scores non-deterministic.

## Faithfulness Deviations

1. **Generation cap and stop set.** The original OpenAI call did not set
   `max_tokens` or stop strings. The adapter caps direct answers at 32 tokens and
   stops only on EOS/chat-end markers so harness runs terminate predictably
   without truncating on newline.
2. **Extractor reconstructed.** The repo's zero-shot runner dumps generations,
   and the paper reports accuracy but does not ship an accuracy scorer. The
   extractor is intentionally narrow: the first standalone `yes` or `no`,
   case-insensitive, is mapped to the corresponding label; missing or unparsable
   answers are scored wrong. This matches the prompt's direct-answer instruction
   without trying to infer semantics from explanations.

## Faithful Scope Choices

1. **Non-CoT only.** Appendix C says CoT was tried, performed worse, and was not
   reported, so this adapter implements the reported non-CoT protocol only.
2. **Test split only.** The shipped question files contain train and test
   conversations. The paper evaluates zero-shot prompting only on the held-out
   test conversation, so the default loader excludes CIDs 4245, 4248, and 4310.

## Deferred Parts

- **Diagnostic CoT variant.** A CoT prompt would be useful for model analysis,
  but it is not a paper-reported headline condition and should be labeled
  diagnostic if added.
- **All-CID diagnostic run.** The loader supports `cid: all` for local analysis,
  but no YAML task exposes it because the paper's zero-shot evaluation is the
  held-out CID `4431` split.
- **Stricter extractor comparison.** A future diagnostic could compare the
  current first-standalone-token extractor against a stricter starts-with-Yes/No
  parser to quantify sensitivity to rambling generations.

## Provenance

- Data: `benchmarks/common-tom/data/questions/*.csv.gz`
- Prompt/window code: `benchmarks/common-tom/bin/openai_zero_shot.py`
- Prompt text: `benchmarks/common-tom/data/prompts/gpt-zero-shot`
- Paper protocol: Appendix C and Table 3 of Soubki et al. 2024
  (https://aclanthology.org/2024.findings-acl.880/)
