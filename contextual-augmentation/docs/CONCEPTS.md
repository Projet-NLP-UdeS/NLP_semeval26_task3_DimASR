# Contextual Augmentation: General Concepts

This note covers the conceptual background only. For script-level behavior, see
[dimabsa_contextual_augmentation_script_reference.md](dimabsa_contextual_augmentation_script_reference.md).

## 1. What Is Data Augmentation?

Data augmentation creates additional training examples from existing examples.

In NLP, common approaches include:

- synonym substitution
- contextual substitution
- insertion or deletion
- paraphrasing
- back-translation

Goal: improve generalization by exposing the model to more surface forms while
preserving task labels.

## 2. What Is Contextual Augmentation?

Contextual augmentation replaces or inserts tokens based on sentence context,
rather than using a fixed synonym list.

Example:

```text
the battery life is very [MASK] .
```

A masked language model predicts plausible words for `[MASK]` based on
surrounding context.

## 3. What Is A Masked Language Model?

A masked language model is trained to predict hidden tokens in a sentence.

Example:

```text
the laptop has a [MASK] screen .
```

Possible predictions may include:

```text
bright
large
touch
small
```

BERT and DistilBERT are standard masked language models.

## 4. Why ABSA/DimABSA Needs Extra Care

For ABSA/DimABSA, labels are tied to specific text spans, not only sentence
meaning. A typical record includes:

```json
{
  "Aspect": "track pad",
  "Category": "HARDWARE#GENERAL",
  "Opinion": "disappointment",
  "VA": "2.50#6.00"
}
```

If augmentation changes labeled aspect/opinion phrases but labels stay unchanged,
training data becomes inconsistent.

## 5. Label-Consistency Rule

When labels are not regenerated, keep the sample only if label-critical phrases
are still present in the augmented text.

For this project, the protected phrases are:

- Aspect
- Opinion

This conservative policy typically yields fewer augmented samples, but better
label fidelity.

## 6. Why Conservative Augmentation Is Often Better Here

Aggressive augmentation increases variety but can introduce label noise.

Conservative augmentation trades quantity for reliability:

- lower risk of corrupting span-level labels
- more stable supervision for valence/arousal targets
- easier error analysis
