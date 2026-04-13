"""
data_utils.py
=============
Shared I/O helpers for the DimABSA augmentation pipeline.

All other modules import from here so that the .jsonl schema is
defined in one place.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    """Load every line of a .jsonl file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {lineno} of {path}: {exc}") from exc
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a .jsonl file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[data_utils] Saved {len(records)} records → {path}")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Memory-efficient line-by-line generator (useful for large files)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Record manipulation helpers
# ---------------------------------------------------------------------------

def deep_copy_record(record: dict) -> dict:
    """Return a fully independent deep copy of a record dict."""
    return copy.deepcopy(record)


def parse_va(va_str: str) -> tuple[float, float]:
    """
    Parse a VA string like '6.83#6.33' into (valence, arousal) floats.

    Raises ValueError if the format is unexpected.
    """
    parts = va_str.strip().split("#")
    if len(parts) != 2:
        raise ValueError(f"Expected 'V#A' format, got: {va_str!r}")
    return float(parts[0]), float(parts[1])


def format_va(valence: float, arousal: float) -> str:
    """Format (valence, arousal) back to the canonical '6.83#6.33' string."""
    return f"{valence:.2f}#{arousal:.2f}"


def get_protected_spans(record: dict) -> set[str]:
    """
    Return the set of lowercased strings that must not be modified:
    - all Aspect terms
    - all Opinion expressions

    Used by the PoS augmenter to build a per-example blacklist.
    """
    protected: set[str] = set()
    for quad in record.get("Quadruplet", []):
        aspect = quad.get("Aspect", "").strip().lower()
        opinion = quad.get("Opinion", "").strip().lower()
        if aspect:
            protected.add(aspect)
        if opinion:
            protected.add(opinion)
        # Also add individual tokens from multi-word spans
        for tok in aspect.split():
            protected.add(tok)
        for tok in opinion.split():
            protected.add(tok)
    return protected


def validate_record(record: dict) -> bool:
    """
    Basic schema validation.  Returns True if the record looks well-formed.
    Does NOT modify the record.
    """
    if not isinstance(record.get("ID"), str):
        return False
    if not isinstance(record.get("Text"), str) or not record["Text"].strip():
        return False
    if not isinstance(record.get("Quadruplet"), list):
        return False
    for quad in record["Quadruplet"]:
        for key in ("Aspect", "Category", "Opinion", "VA"):
            if key not in quad:
                return False
        try:
            parse_va(quad["VA"])
        except ValueError:
            return False
    return True


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def deduplicate(
    records: list[dict],
    existing_texts: set[str] | None = None,
) -> list[dict]:
    """
    Remove records whose normalised Text already appears in *existing_texts*
    or appears more than once within *records* itself.

    Returns a deduplicated list and leaves *records* unchanged.
    The comparison is case-insensitive after collapsing whitespace.
    """
    seen: set[str] = set(existing_texts) if existing_texts else set()
    unique = []
    for rec in records:
        key = " ".join(rec.get("Text", "").lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def collect_texts(records: list[dict]) -> set[str]:
    """Return the set of normalised texts from a list of records."""
    return {" ".join(r.get("Text", "").lower().split()) for r in records}
