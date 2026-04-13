"""Dummy replacement for `import nlpaug.augmenter.word as naw`.

Use this only for portability/testing when real `nlpaug` is unavailable.
"""

from __future__ import annotations


class ContextualWordEmbsAug:
    """Minimal stand-in for nlpaug's ContextualWordEmbsAug.

    This class preserves the interface used by
    scripts/augment_dimabsa_contextual.py.
    It does not perform real NLP augmentation.
    """

    def __init__(
        self,
        model_path="distilbert-base-uncased",
        action="substitute",
        device="cpu",
        batch_size=8,
        top_k=100,
        aug_p=0.15,
        aug_min=1,
        aug_max=3,
        silence=True,
        **kwargs,
    ):
        self.model_path = model_path
        self.action = action
        self.device = device
        self.batch_size = batch_size
        self.top_k = top_k
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.silence = silence

        self.stopwords = []
        self.stopword_reg = None
        self.reserve_word_reg = None

        self._counter = 0

    def _build_stop_words(self, stopwords):
        # Keep compatibility with caller; no regex/index build in stub.
        self.stopwords = stopwords or []

    def augment(self, text, n=1):
        """Return deterministic fake augmentations while preserving original text.

        The caller expects a list and accesses index 0.
        """
        outputs = []
        for _ in range(max(1, int(n))):
            self._counter += 1
            outputs.append(f"{text} [dummy-ctxaug-{self._counter}]")
        return outputs
