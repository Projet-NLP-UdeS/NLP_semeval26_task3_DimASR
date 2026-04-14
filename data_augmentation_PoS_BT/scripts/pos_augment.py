"""
pos_augment.py
==============
Task 1 – Data Augmentation via Part-of-Speech (PoS) substitution.

Strategy
--------
* Use spaCy to tag every token in the sentence.
* Classify each token as SAFE or PROTECTED.
* Apply deterministic, meaning-preserving substitution rules to SAFE tokens.
* Keep the Quadruplet (including VA values) unchanged.

Safe-to-replace tags
--------------------
    AUX   – auxiliary verbs          e.g.  is → was, are → were
    DET   – determiners              e.g.  the → this/a
    ADP   – prepositions             e.g.  in → at (context-aware)
    NOUN  – neutral nouns ONLY when NOT part of Aspect/Opinion spans

Protected (never modified)
--------------------------
    ADJ, ADV, any token overlapping an Aspect or Opinion span
"""

from __future__ import annotations

import random
import re
import uuid
from pathlib import Path
from typing import Callable

import spacy

from data_utils import (
    collect_texts,
    deduplicate,
    deep_copy_record,
    get_protected_spans,
    load_jsonl,
    save_jsonl,
    validate_record,
)

# ---------------------------------------------------------------------------
# Substitution rule tables
# ---------------------------------------------------------------------------

AUX_MAP: dict[str, list[str]] = {
    "is": ["was"],
    "are": ["were"],
    "was": ["is"],
    "were": ["are"],
    "has": ["had"],
    "have": ["had"],
    "had": ["have"],
    "will": ["would"],
    "would": ["will"],
    "can": ["could"],
    "could": ["can"],
    "do": ["did"],
    "does": ["did"],
    "did": ["does"],
}

DET_MAP: dict[str, list[str]] = {
    "the": ["this", "that"],
    "a": ["one"],
    "an": ["one"],
    "this": ["the"],
    "that": ["the"],
    "these": ["the"],
    "those": ["the"],
    "some": ["a few"],
    "any": ["some"],
}

# Preposition swaps: only contextually safe pairs
ADP_MAP: dict[str, list[str]] = {
    "in": ["at", "inside"],
    "at": ["in"],
    "on": ["upon"],
    "upon": ["on"],
    "for": ["for"],   # no-op placeholder – rarely safe to swap
    "with": ["with"],
    "by": ["by"],
    "from": ["from"],
    "to": ["toward"],
    "toward": ["to"],
    "about": ["regarding", "concerning"],
    "regarding": ["about"],
    "concerning": ["about"],
    "during": ["throughout"],
    "throughout": ["during"],
    "near": ["close to", "by"],
    "inside": ["within"],
    "within": ["inside"],
}

# Neutral noun replacements (domain-level, manually curated)
NEUTRAL_NOUN_MAP: dict[str, list[str]] = {
    "place": ["spot", "venue", "location"],
    "spot": ["place", "venue"],
    "venue": ["place", "spot"],
    "restaurant": ["eatery", "establishment"],
    "eatery": ["restaurant"],
    "establishment": ["restaurant", "place"],
    "evening": ["night"],
    "night": ["evening"],
    "meal": ["dish"],
    "dish": ["meal"],
    "table": ["seat"],
    "seat": ["table"],
    "waiter": ["server"],
    "waitress": ["server"],
    "server": ["waiter"],
    "staff": ["team"],
    "team": ["staff"],
    "experience": ["visit"],
    "visit": ["experience"],
    "atmosphere": ["ambiance", "vibe"],
    "ambiance": ["atmosphere"],
    "vibe": ["atmosphere"],
    "neighborhood": ["area", "district"],
    "area": ["neighborhood"],
    "district": ["neighborhood"],
    "portion": ["serving", "amount"],
    "serving": ["portion"],
    "taste": ["flavor", "flavour"],
    "flavor": ["taste"],
    "flavour": ["taste", "flavor"],
    "selection": ["variety", "range"],
    "variety": ["selection"],
    "range": ["selection"],
    "time": ["moment"],
    "moment": ["time"],
    "money": ["cash"],
    "cash": ["money"],
}


# ---------------------------------------------------------------------------
# Core augmenter class
# ---------------------------------------------------------------------------

class PosAugmenter:
    """
    PoS-based sentence augmenter that preserves quadruplet annotations.

    Parameters
    ----------
    spacy_model : str
        spaCy model name to load (default: 'en_core_web_sm').
    rng_seed : int | None
        Seed for the random number generator for reproducibility.
    max_substitutions_per_sentence : int
        Upper bound on how many tokens we replace per sentence.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        rng_seed: int | None = 42,
        max_substitutions_per_sentence: int = 3,
    ) -> None:
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Run:  python -m spacy download {spacy_model}"
            )
        self.rng = random.Random(rng_seed)
        self.max_subs = max_substitutions_per_sentence

        # Build combined lookup (lowercase token → list of replacements)
        self._lookup: dict[str, list[str]] = {}
        for src, tgts in {
            **AUX_MAP,
            **DET_MAP,
            **ADP_MAP,
            **NEUTRAL_NOUN_MAP,
        }.items():
            self._lookup.setdefault(src.lower(), []).extend(tgts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_sentence(self, record: dict) -> dict | None:
        """
        Generate one augmented variant of *record*.

        Returns a new record dict, or None if no safe substitution
        could be found (e.g. all tokens are protected).
        """
        text = record["Text"]
        protected = get_protected_spans(record)
        doc = self.nlp(text)

        # Build a list of candidate (token_index, replacement) pairs
        candidates: list[tuple[int, str]] = []
        for i, tok in enumerate(doc):
            if self._is_protected(tok, protected):
                continue
            replacements = self._get_replacements(tok)
            if replacements:
                replacement = self.rng.choice(replacements)
                if replacement.lower() != tok.text.lower():  # skip no-ops
                    candidates.append((i, replacement))

        if not candidates:
            return None  # nothing safe to change

        # Shuffle and cap at max_substitutions_per_sentence
        self.rng.shuffle(candidates)
        selected = candidates[: self.max_subs]

        # Apply substitutions while preserving original capitalisation style
        tokens = [tok.text_with_ws for tok in doc]
        for idx, repl in selected:
            orig_tok = doc[idx]
            repl_with_ws = self._match_case(orig_tok.text, repl) + orig_tok.whitespace_
            tokens[idx] = repl_with_ws

        new_text = "".join(tokens).strip()
        if new_text == text:
            return None  # transformation was a no-op overall

        new_record = deep_copy_record(record)
        new_record["Text"] = new_text
        # VA and Quadruplet are preserved exactly (deep-copied above)
        return new_record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_protected(self, tok, protected: set[str]) -> bool:
        """Return True if this token must not be modified."""
        lower = tok.text.lower()

        # Never touch adjectives or adverbs (they carry sentiment/VA)
        if tok.pos_ in ("ADJ", "ADV"):
            return True

        # Skip punctuation, space, numbers
        if tok.pos_ in ("PUNCT", "SPACE", "NUM", "SYM", "X"):
            return True

        # Skip if the lowercase token appears in the protected span set
        if lower in protected:
            return True

        # Skip if the token is part of a named entity
        if tok.ent_type_:
            return True

        return False

    def _get_replacements(self, tok) -> list[str]:
        """
        Return a list of valid replacements for a token, filtered by POS.
        Returns an empty list if no replacement is appropriate.
        """
        lower = tok.text.lower()

        if tok.pos_ == "AUX" and lower in AUX_MAP:
            return AUX_MAP[lower]
        if tok.pos_ == "DET" and lower in DET_MAP:
            return DET_MAP[lower]
        if tok.pos_ == "ADP" and lower in ADP_MAP:
            # Filter out no-op entries (lists where only item == source)
            candidates = [r for r in ADP_MAP[lower] if r != lower]
            return candidates if candidates else []
        if tok.pos_ == "NOUN" and lower in NEUTRAL_NOUN_MAP:
            return NEUTRAL_NOUN_MAP[lower]

        return []

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """
        Apply the capitalisation of *original* to *replacement*.

        Rules:
        - All-caps  → ALL-CAPS replacement
        - Title case → Title case replacement
        - Otherwise  → lowercase replacement
        """
        if original.isupper():
            return replacement.upper()
        if original.istitle() or (original and original[0].isupper()):
            return replacement[0].upper() + replacement[1:] if replacement else replacement
        return replacement.lower()


# ---------------------------------------------------------------------------
# Main pipeline function (required interface)
# ---------------------------------------------------------------------------

def augment_dataset(
    input_path: str | Path,
    output_path: str | Path,
    n_samples: int = 750,
    spacy_model: str = "en_core_web_sm",
    rng_seed: int = 42,
    max_substitutions: int = 3,
    max_attempts_per_record: int = 5,
) -> list[dict]:
    """
    Generate *n_samples* augmented examples using PoS substitution.

    Parameters
    ----------
    input_path : path to the source .jsonl file
    output_path : path where augmented_pos.jsonl will be written
    n_samples : desired number of augmented records (500–1000 recommended)
    spacy_model : spaCy model name (must be downloaded first)
    rng_seed : random seed for reproducibility
    max_substitutions : max token replacements per sentence
    max_attempts_per_record : how many times to retry a record before skipping

    Returns
    -------
    List of augmented record dicts (also written to output_path).
    """
    print(f"[pos_augment] Loading dataset from {input_path} …")
    original_records = load_jsonl(input_path)
    valid_records = [r for r in original_records if validate_record(r)]
    print(f"[pos_augment] {len(valid_records)} valid records loaded.")

    # Pre-collect original texts so we never duplicate them
    existing_texts = collect_texts(valid_records)

    augmenter = PosAugmenter(
        spacy_model=spacy_model,
        rng_seed=rng_seed,
        max_substitutions_per_sentence=max_substitutions,
    )

    rng = random.Random(rng_seed)
    augmented: list[dict] = []
    new_texts: set[str] = set()

    # We cycle through the dataset, generating variants until we hit n_samples
    pool = valid_records[:]
    rng.shuffle(pool)
    idx = 0
    total_attempts = 0
    max_total_attempts = n_samples * 20  # safety stop

    while len(augmented) < n_samples and total_attempts < max_total_attempts:
        record = pool[idx % len(pool)]
        idx += 1
        total_attempts += 1

        for _ in range(max_attempts_per_record):
            new_rec = augmenter.augment_sentence(record)
            if new_rec is None:
                break

            norm_text = " ".join(new_rec["Text"].lower().split())
            if norm_text in existing_texts or norm_text in new_texts:
                # Try again with a different random choice
                augmenter.rng = random.Random(rng.randint(0, 10_000_000))
                continue

            # Assign a fresh ID
            new_rec["ID"] = f"pos_aug_{len(augmented):05d}"
            augmented.append(new_rec)
            new_texts.add(norm_text)
            break

    print(
        f"[pos_augment] Generated {len(augmented)} / {n_samples} requested samples "
        f"({total_attempts} attempts)."
    )

    # Deduplicate once more as a safety net
    augmented = deduplicate(augmented, existing_texts=existing_texts)

    save_jsonl(augmented, output_path)
    return augmented


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PoS-based data augmentation for DimABSA")
    parser.add_argument("--input",  required=True, help="Input .jsonl path")
    parser.add_argument("--output", default="augmented_pos.jsonl", help="Output .jsonl path")
    parser.add_argument("--n_samples", type=int, default=750, help="Number of samples to generate")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_subs", type=int, default=3, help="Max token substitutions per sentence")
    args = parser.parse_args()

    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        spacy_model=args.spacy_model,
        rng_seed=args.seed,
        max_substitutions=args.max_subs,
    )
