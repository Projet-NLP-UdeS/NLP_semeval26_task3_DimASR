# Contextual Augmentation Script Reference

This file documents behavior of:

```text
scripts/augment_dimabsa_contextual.py
```

For conceptual background, see
[dimabsa_contextual_augmentation_concepts.md](dimabsa_contextual_augmentation_concepts.md).

## 1. Core Components

- Augmenter class: `nlpaug.augmenter.word.ContextualWordEmbsAug`
- Typical model: `distilbert-base-uncased`
- Input format: JSONL with `Text` and `Quadruplet`

## 2. Safety Logic In The Script

The script uses two safeguards to preserve label alignment.

1. It marks protected phrases as stopwords for the augmenter.
2. It verifies each generated candidate still contains all protected phrases.

Protected phrases come from each quadruplet's non-`NULL`:

- `Aspect`
- `Opinion`

## 3. Acceptance Conditions Per Candidate

A generated candidate is accepted only if:

1. it is not identical to previously seen text for that record
2. all protected phrases are still present

If no candidate passes within `--retries`, the record is skipped.

## 4. Key Arguments

- `--action`: `substitute` or `insert`
- `--num-aug`: target number of accepted augmentations per record
- `--retries`: attempts per augmentation target
- `--aug-p`: approximate fraction of eligible tokens to modify
- `--aug-min`: minimum token changes
- `--aug-max`: maximum token changes
- `--top-k`: candidate pool size from model logits
- `--no-original`: write only augmented records

## 5. Why Kept Count Can Be Lower Than Input Count

With strict phrase preservation, many candidates are rejected.

So this is expected:

- all records are processed
- fewer records produce accepted augmentations

## 6. Practical Tuning

To increase accepted outputs while staying conservative:

- raise `--retries` (for example `10`)
- lower aggressiveness (`--aug-p 0.10`, `--aug-max 1`)
- reduce diversity (`--top-k 20`)

To increase coverage (with more risk to naturalness):

- try `--action insert`

## 7. Example Commands

Conservative substitution:

```bash
python scripts/augment_dimabsa_contextual.py \
  path/to/input.jsonl \
  path/to/output.jsonl \
  --num-aug 1 \
  --model-path distilbert-base-uncased \
  --batch-size 1 \
  --aug-p 0.10 \
  --aug-max 1 \
  --top-k 20 \
  --retries 10 \
  --no-original
```

Insertion variant:

```bash
python scripts/augment_dimabsa_contextual.py \
  path/to/input.jsonl \
  path/to/output_insert.jsonl \
  --action insert \
  --num-aug 1 \
  --model-path distilbert-base-uncased \
  --batch-size 1 \
  --aug-p 0.10 \
  --aug-max 1 \
  --top-k 20 \
  --retries 10 \
  --no-original
```

## 8. End-To-End Pipeline Summary

For each input record:

1. extract protected phrases from quadruplets
2. keep only phrases present in original text
3. configure augmenter stopwords for those phrases
4. generate candidates with contextual augmentation
5. reject duplicates or phrase-breaking outputs
6. keep accepted candidate(s) and assign new ID suffix
7. write output with or without original records
