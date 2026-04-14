"""
main.py
=======
Orchestrateur du pipeline d'augmentation DimABSA.

Stratégies disponibles
----------------------
  pos       : substitution PoS (auxiliaires, déterminants, prépositions, noms neutres)
  backtrans : backtranslation EN → langue pivot → EN
  adj       : substitution de synonymes ADJ/ADV de même polarité (VA préservée)
  combine   : fusion de tous les fichiers produits

Exemples d'utilisation
----------------------
# PoS seulement
python main.py --input data.jsonl --tasks pos

# Pipeline complet
python main.py --input data.jsonl --tasks pos backtrans adj combine

# Paramètres personnalisés
python main.py \
    --input data.jsonl \
    --tasks pos backtrans adj combine \
    --pos_n_samples 750  --pos_max_subs 3 \
    --bt_n_samples 500   --pivot_langs fr es \
    --adj_n_samples 600  --adj_max_subs 1 \
    --output_dir ./output --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Sous-fonctions
# ---------------------------------------------------------------------------

def run_pos(args) -> Path:
    from pos_augment import augment_dataset
    output = Path(args.output_dir) / "augmented_pos.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE : Augmentation PoS")
    print("=" * 60)
    augment_dataset(
        input_path        = args.input,
        output_path       = output,
        n_samples         = args.pos_n_samples,
        spacy_model       = args.spacy_model,
        rng_seed          = args.seed,
        max_substitutions = args.pos_max_subs,
    )
    return output


def run_backtrans(args) -> Path:
    from backtrans_augment import augment_dataset
    output = Path(args.output_dir) / "augmented_backtrans.jsonl"
    print("\n" + "=" * 60)
    print(f"ÉTAPE : Backtranslation  [{', '.join(args.pivot_langs)}]")
    print("=" * 60)
    augment_dataset(
        input_path  = args.input,
        output_path = output,
        n_samples   = args.bt_n_samples,
        pivot_langs = args.pivot_langs,
        device      = args.device,
        rng_seed    = args.seed,
        verbose     = args.verbose,
    )
    return output


def run_adj(args) -> Path:
    from pos_adj_augment import augment_dataset
    output = Path(args.output_dir) / "augmented_adj.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE : Substitution synonymes ADJ/ADV (même polarité)")
    print("=" * 60)
    augment_dataset(
        input_path        = args.input,
        output_path       = output,
        n_samples         = args.adj_n_samples,
        rng_seed          = args.seed,
        max_substitutions = args.adj_max_subs,
    )
    return output


def run_combine(args, pos_path, bt_path, adj_path) -> Path:
    from combine import combine_datasets
    output = Path(args.output_dir) / "final_augmented_dataset.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE FINALE : Fusion des datasets")
    print("=" * 60)
    combine_datasets(
        original_path = args.input,
        pos_aug_path  = pos_path,
        bt_aug_path   = bt_path,
        adj_aug_path  = adj_path,
        output_path   = output,
    )
    return output


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DimABSA Augmentation Pipeline (PoS + Backtranslation + ADJ synonymes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input",      required=True,
                        help="Fichier source .jsonl")
    parser.add_argument("--output_dir", default="output",
                        help="Dossier de sortie (défaut : ./output)")
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["pos", "backtrans", "adj", "combine"],
        default=["pos", "backtrans", "adj", "combine"],
        help="Étapes à exécuter",
    )

    # PoS
    parser.add_argument("--pos_n_samples", type=int, default=750)
    parser.add_argument("--pos_max_subs",  type=int, default=3)
    parser.add_argument("--spacy_model",   default="en_core_web_sm")

    # Backtranslation
    parser.add_argument("--bt_n_samples",  type=int, default=500)
    parser.add_argument("--pivot_langs",   nargs="+", default=["fr"],
                        choices=["fr","es","de","it","nl"])
    parser.add_argument("--device",        default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--verbose",       action="store_true")

    # ADJ synonymes
    parser.add_argument("--adj_n_samples", type=int, default=600,
                        help="Nb d'exemples ADJ à générer (défaut : 600)")
    parser.add_argument("--adj_max_subs",  type=int, default=1,
                        help="Max de remplacements par phrase (défaut : 1)")

    # Divers
    parser.add_argument("--seed", type=int, default=42)

    return parser


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tasks    = set(args.tasks)
    pos_path = bt_path = adj_path = None

    # Récupérer les fichiers existants si on ne les regénère pas
    for key, fname in [
        ("pos",       "augmented_pos.jsonl"),
        ("backtrans", "augmented_backtrans.jsonl"),
        ("adj",       "augmented_adj.jsonl"),
    ]:
        if key not in tasks:
            c = Path(args.output_dir) / fname
            if c.exists():
                print(f"[main] Fichier existant détecté : {c}")
                if key == "pos":       pos_path = c
                elif key == "backtrans": bt_path  = c
                elif key == "adj":     adj_path = c

    if "pos"       in tasks: pos_path  = run_pos(args)
    if "backtrans" in tasks: bt_path   = run_backtrans(args)
    if "adj"       in tasks: adj_path  = run_adj(args)
    if "combine"   in tasks: run_combine(args, pos_path, bt_path, adj_path)

    # Résumé
    print("\n" + "=" * 60)
    print("[main] Pipeline terminé. Fichiers produits :")
    print("=" * 60)
    for fname in [
        "augmented_pos.jsonl",
        "augmented_backtrans.jsonl",
        "augmented_adj.jsonl",
        "final_augmented_dataset.jsonl",
    ]:
        p = Path(args.output_dir) / fname
        if p.exists():
            n = sum(1 for _ in open(p, encoding="utf-8"))
            print(f"  {p}  ({n} lignes)")


if __name__ == "__main__":
    main()
