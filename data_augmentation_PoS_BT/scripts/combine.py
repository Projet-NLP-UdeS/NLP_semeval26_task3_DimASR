"""
combine.py  – Fusion des trois sources d'augmentation DimABSA.
"""
from __future__ import annotations
from pathlib import Path
from data_utils import deduplicate, load_jsonl, save_jsonl, validate_record


def combine_datasets(
    original_path: str | Path,
    pos_aug_path:  str | Path | None,
    bt_aug_path:   str | Path | None,
    adj_aug_path:  str | Path | None,
    output_path:   str | Path,
    validate: bool = True,
) -> list[dict]:
    def _load(path, label):
        if path is None:
            print(f"[combine] {label} ignoré.")
            return []
        p = Path(path)
        if not p.exists():
            print(f"[combine] AVERTISSEMENT – introuvable : {path}")
            return []
        recs = load_jsonl(p)
        print(f"[combine] {len(recs):5d} exemples ← {label} ({p.name})")
        return recs

    original = _load(original_path, "original")
    pos_aug  = _load(pos_aug_path,  "PoS-augmenté")
    bt_aug   = _load(bt_aug_path,   "Backtrans-augmenté")
    adj_aug  = _load(adj_aug_path,  "ADJ-augmenté")

    if validate:
        original = [r for r in original if validate_record(r)]
        pos_aug  = [r for r in pos_aug  if validate_record(r)]
        bt_aug   = [r for r in bt_aug   if validate_record(r)]
        adj_aug  = [r for r in adj_aug  if validate_record(r)]

    all_records = original + pos_aug + bt_aug + adj_aug
    merged = deduplicate(all_records)

    print(
        f"\n[combine] Résumé :\n"
        f"          Original       : {len(original)}\n"
        f"          PoS-augmenté   : {len(pos_aug)}\n"
        f"          Backtrans-aug  : {len(bt_aug)}\n"
        f"          ADJ-augmenté   : {len(adj_aug)}\n"
        f"          ─────────────────────\n"
        f"          Total avant dédup : {len(all_records)}\n"
        f"          Total après dédup : {len(merged)}"
    )
    save_jsonl(merged, output_path)
    return merged


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--original",  required=True)
    p.add_argument("--pos_aug",   default=None)
    p.add_argument("--bt_aug",    default=None)
    p.add_argument("--adj_aug",   default=None)
    p.add_argument("--output",    default="final_augmented_dataset.jsonl")
    p.add_argument("--no_validate", action="store_true")
    args = p.parse_args()
    combine_datasets(args.original, args.pos_aug, args.bt_aug,
                     args.adj_aug, args.output, not args.no_validate)
