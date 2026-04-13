"""
combine.py
==========
Fusion finale : original + PoS-augmenté + Backtranslation-augmenté
→ final_augmented_dataset.jsonl
"""

from __future__ import annotations

from pathlib import Path

from data_utils import (
    collect_texts,
    deduplicate,
    load_jsonl,
    save_jsonl,
    validate_record,
)


def combine_datasets(
    original_path: str | Path,
    pos_aug_path:  str | Path | None,
    bt_aug_path:   str | Path | None,
    output_path:   str | Path,
    validate: bool = True,
) -> list[dict]:
    """
    Fusionne le dataset original avec les deux fichiers augmentés.

    Parameters
    ----------
    original_path : fichier source .jsonl
    pos_aug_path  : augmented_pos.jsonl (ou None pour ignorer)
    bt_aug_path   : augmented_backtrans.jsonl (ou None pour ignorer)
    output_path   : destination final_augmented_dataset.jsonl
    validate      : si True, supprime les records mal formés

    Returns
    -------
    Liste fusionnée et dédupliquée.
    """

    def _load(path, label):
        if path is None:
            print(f"[combine] {label} ignoré (aucun chemin fourni).")
            return []
        p = Path(path)
        if not p.exists():
            print(f"[combine] AVERTISSEMENT – fichier introuvable : {path}")
            return []
        recs = load_jsonl(p)
        print(f"[combine] {len(recs)} exemples chargés depuis {label} ({p.name})")
        return recs

    original = _load(original_path, "original")
    pos_aug  = _load(pos_aug_path,  "PoS-augmenté")
    bt_aug   = _load(bt_aug_path,   "Backtrans-augmenté")

    if validate:
        original = [r for r in original if validate_record(r)]
        pos_aug  = [r for r in pos_aug  if validate_record(r)]
        bt_aug   = [r for r in bt_aug   if validate_record(r)]

    # Fusion : original en premier (priorité pour la déduplication)
    all_records = original + pos_aug + bt_aug
    merged = deduplicate(all_records)

    print(
        f"\n[combine] Résumé :\n"
        f"          Original      : {len(original)}\n"
        f"          PoS-augmenté  : {len(pos_aug)}\n"
        f"          Backtrans-aug : {len(bt_aug)}\n"
        f"          Total avant dédup : {len(all_records)}\n"
        f"          Total après dédup : {len(merged)}"
    )

    save_jsonl(merged, output_path)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fusion des fichiers augmentés DimABSA")
    p.add_argument("--original",   required=True)
    p.add_argument("--pos_aug",    default=None)
    p.add_argument("--bt_aug",     default=None)
    p.add_argument("--output",     default="final_augmented_dataset.jsonl")
    p.add_argument("--no_validate", action="store_true")
    args = p.parse_args()

    combine_datasets(
        original_path = args.original,
        pos_aug_path  = args.pos_aug,
        bt_aug_path   = args.bt_aug,
        output_path   = args.output,
        validate      = not args.no_validate,
    )
