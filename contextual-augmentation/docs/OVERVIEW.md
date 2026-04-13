# Contextual Data Augmentation For DimABSA

## Reading Guide

- General concepts (data augmentation, contextual augmentation, masked language
  models, and why label protection matters):
  [dimabsa_contextual_augmentation_concepts.md](dimabsa_contextual_augmentation_concepts.md)
- Script and implementation details (arguments, safety checks, retries, and
  commands):
  [dimabsa_contextual_augmentation_script_reference.md](dimabsa_contextual_augmentation_script_reference.md)
- Full deep-dive reference (extended explanations and examples):
  [script_reference_full.md](script_reference_full.md)

## Quick Start Command

```bash
python scripts/augment_dimabsa_contextual.py \
    path/to/input.jsonl \
    path/to/output.jsonl \
    --num-aug 1 \
    --model-path distilbert-base-uncased \
    --batch-size 1 \
    --aug-p 0.15 \
    --aug-max 2 \
    --no-original
```

Main script:

```text
scripts/augment_dimabsa_contextual.py
```
