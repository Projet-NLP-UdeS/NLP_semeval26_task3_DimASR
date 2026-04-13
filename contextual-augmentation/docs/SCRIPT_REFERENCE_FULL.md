# Contextual Augmentation Script Reference (Full)

This document explains the implementation details behind:

```text
scripts/augment_dimabsa_contextual.py
```

## Table of Contents

- [1. Why Protect Aspect And Opinion Phrases?](#1-why-protect-aspect-and-opinion-phrases)
- [2. How Protected Phrases Are Extracted](#2-how-protected-phrases-are-extracted)
- [3. Why We Keep Only Phrases Present In The Original Text](#3-why-we-keep-only-phrases-present-in-the-original-text)
- [4. How The Script Prevents Editing Protected Phrases](#4-how-the-script-prevents-editing-protected-phrases)
- [5. Second Safety Check On Final Candidates](#5-second-safety-check-on-final-candidates)
- [6. What --retries Means](#6-what---retries-means)
- [7. How Many Tokens Are Changed](#7-how-many-tokens-are-changed)
- [8. How Token Positions And Replacements Are Chosen](#8-how-token-positions-and-replacements-are-chosen)
- [9. What --top-k And --seed Control](#9-what---top-k-and---seed-control)
- [10. What --batch-size And --no-original Control](#10-what---batch-size-and---no-original-control)
- [11. Practical Tuning To Keep More Safe Augmentations](#11-practical-tuning-to-keep-more-safe-augmentations)
- [12. Recommended Commands](#12-recommended-commands)
- [13. Full Pipeline Summary](#13-full-pipeline-summary)
- [14. Why This Design Is Conservative](#14-why-this-design-is-conservative)

## 1. Why Protect Aspect And Opinion Phrases?

For ABSA and DimABSA, labels refer to specific spans in the sentence.

Example:

```json
{
  "Text": "- biggest disappointment is the track pad .",
  "Quadruplet": [
    {
      "Aspect": "track pad",
      "Category": "HARDWARE#GENERAL",
      "Opinion": "disappointment",
      "VA": "2.50#6.00"
    }
  ]
}
```

If augmentation changes a labeled phrase but labels stay unchanged, the training
example becomes inconsistent.

- `disappointment` changed to `problem` while label still says `Opinion = disappointment`
- `track pad` changed to `touchpad` while label still says `Aspect = track pad`

For VA labels, changing the opinion phrase can also invalidate valence/arousal.

Rule used in this script:

```text
Keep original labels only if labeled Aspect and Opinion phrases
are still present in augmented text.
```

## 2. How Protected Phrases Are Extracted

```python
def protected_phrases(record):
    phrases = []
    for quad in record.get("Quadruplet", []):
        for key in ("Aspect", "Opinion"):
            value = quad.get(key)
            if value and value != NULL_VALUE and value not in phrases:
                phrases.append(value)
    return phrases
```

Protected:

- `Aspect`
- `Opinion`

Not protected:

- `Category`
- `VA`
- `ID`
- `NULL`

## 3. Why We Keep Only Phrases Present In The Original Text

```python
def present_phrases(text, phrases):
    lowered = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in lowered]
```

This avoids protecting labels that do not appear as exact text spans in the
sentence, for example due to `NULL` or tokenization mismatch.

## 4. How The Script Prevents Editing Protected Phrases

```python
def set_record_stopwords(aug, phrases):
    stopwords = [phrase.lower() for phrase in phrases] if "uncased" in aug.model_path else phrases
    aug.stopwords = stopwords
    aug.stopword_reg = None
    aug.reserve_word_reg = None
    aug._build_stop_words(stopwords)
```

For `distilbert-base-uncased`, protected phrases are lowercased before being set
as stopwords.

Here, `stopwords` means "phrases to avoid modifying," not linguistic stopwords.

## 5. Second Safety Check On Final Candidates

```python
def preserves_phrases(text, phrases):
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)
```

Candidate acceptance logic:

```python
candidate = aug.augment(text, n=1)[0]
if candidate in seen_texts:
    continue
if not preserves_phrases(candidate, phrases_to_keep):
    continue
augmented_text = candidate
break
```

A candidate is kept only if:

1. it is not duplicated
2. it keeps all protected phrases

## 6. What --retries Means

The script tries multiple times to get a valid candidate:

```python
parser.add_argument("--retries", type=int, default=5)
```

With `--num-aug 1`, each record gets up to `retries` attempts to find one safe
augmentation.

## 7. How Many Tokens Are Changed

Controlled by:

```bash
--aug-p 0.15
--aug-min 1
--aug-max 2
```

Interpretation:

- approximately 15% of eligible tokens are selected
- at least 1 token if possible
- at most 2 tokens

Protected aspect/opinion phrases should not be eligible.

## 8. How Token Positions And Replacements Are Chosen

Position selection in augmenter internals:

```python
aug_idxes = self._get_aug_idxes(head_tokens)
aug_idxes.sort(reverse=True)
```

Replacement flow for `--action substitute`:

1. selected token is masked with `[MASK]`
2. model predicts candidates for the masked position
3. candidates are filtered and sampled

Prediction call:

```python
outputs = self.model.predict(masked_texts, target_words=original_tokens, n=2)
```

## 9. What --top-k And --seed Control

`--top-k` controls candidate pool size from model logits:

```python
parser.add_argument("--top-k", type=int, default=100)
```

- smaller `top_k` (for example 20): safer/more conservative
- larger `top_k` (for example 200): more diversity/more risk

`--seed` affects reproducibility:

```python
parser.add_argument("--seed", type=int, default=42)
random.seed(args.seed)
np.random.seed(args.seed)
```

Some variation can still remain due to backend/model behavior.

## 10. What --batch-size And --no-original Control

`--batch-size`:

- `1` means one masked sentence at a time
- higher values can be faster but use more memory

`--no-original`:

- enabled: write only accepted augmented records
- disabled: write original + accepted augmented records

## 11. Practical Tuning To Keep More Safe Augmentations

Conservative ways to increase kept outputs:

- increase `--retries` (for example `10`)
- reduce edit strength (`--aug-p 0.10`, `--aug-max 1`)
- reduce candidate diversity (`--top-k 20`)

Coverage-oriented option:

- try `--action insert` (often keeps spans better, but can reduce naturalness)

## 12. Recommended Commands

Conservative substitution:

```bash
~/nlpaug/venv/bin/python scripts/augment_dimabsa_contextual.py \
  ~/DimABSA2026-main/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl \
  ~/DimABSA2026-main/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks_contextual_aug.jsonl \
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
~/nlpaug/venv/bin/python scripts/augment_dimabsa_contextual.py \
  ~/DimABSA2026-main/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl \
  ~/DimABSA2026-main/task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks_contextual_insert_aug.jsonl \
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

## 13. Full Pipeline Summary

For each input record:

1. read JSONL record
2. extract non-`NULL` Aspect/Opinion phrases
3. keep only phrases present in original text
4. set protected phrases as stopwords
5. generate candidate(s)
6. reject duplicates and unsafe candidates
7. keep accepted candidates and assign new ID suffix
8. write output according to `--no-original`

## 14. Why This Design Is Conservative

The script does not rewrite labels.

Because labels remain fixed, augmentation must preserve label-critical phrases.
This yields fewer samples than aggressive rewriting, but usually gives more
trustworthy supervision for ABSA/VA training.
