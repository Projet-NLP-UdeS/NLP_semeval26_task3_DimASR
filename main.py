"""
main.py
=======
Orchestrateur du pipeline d'augmentation DimABSA.
Deux stratégies disponibles : PoS (pos) et Backtranslation (backtrans).

Exemples d'utilisation
----------------------
# PoS seulement
python main.py --input eng_restaurant_test_task3.jsonl --tasks pos

# Backtranslation seulement (français par défaut)
python main.py --input eng_restaurant_test_task3.jsonl --tasks backtrans

# Backtranslation avec plusieurs langues pivot
python main.py --input eng_restaurant_test_task3.jsonl --tasks backtrans \
    --pivot_langs fr es de

# Pipeline complet (PoS + backtrans + fusion)
python main.py --input eng_restaurant_test_task3.jsonl --tasks pos backtrans combine

# Paramètres personnalisés
python main.py \
    --input eng_restaurant_test_task3.jsonl \
    --tasks pos backtrans combine \
    --pos_n_samples 750 \
    --pos_max_subs 3 \
    --bt_n_samples 500 \
    --pivot_langs fr es \
    --output_dir ./output \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Sous-fonctions
# ---------------------------------------------------------------------------

def run_pos(args) -> Path:
    from pos_augment import augment_dataset as pos_augment

    output = Path(args.output_dir) / "augmented_pos.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : Augmentation PoS")
    print("=" * 60)
    pos_augment(
        input_path        = args.input,
        output_path       = output,
        n_samples         = args.pos_n_samples,
        spacy_model       = args.spacy_model,
        rng_seed          = args.seed,
        max_substitutions = args.pos_max_subs,
    )
    return output


def run_backtrans(args) -> Path:
    from backtrans_augment import augment_dataset as bt_augment

    output = Path(args.output_dir) / "augmented_backtrans.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Augmentation Backtranslation")
    print(f"          Langues pivot : {args.pivot_langs}")
    print("=" * 60)
    bt_augment(
        input_path  = args.input,
        output_path = output,
        n_samples   = args.bt_n_samples,
        pivot_langs = args.pivot_langs,
        device      = args.device,
        rng_seed    = args.seed,
        verbose     = args.verbose,
    )
    return output


def run_combine(args, pos_path: Path | None, bt_path: Path | None) -> Path:
    from combine import combine_datasets

    output = Path(args.output_dir) / "final_augmented_dataset.jsonl"
    print("\n" + "=" * 60)
    print("ÉTAPE FINALE : Fusion des datasets")
    print("=" * 60)
    combine_datasets(
        original_path = args.input,
        pos_aug_path  = pos_path,
        bt_aug_path   = bt_path,
        output_path   = output,
    )
    return output


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DimABSA Augmentation Pipeline (PoS + Backtranslation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    parser.add_argument("--input", required=True,
                        help="Chemin vers le fichier source .jsonl")
    parser.add_argument("--output_dir", default="output",
                        help="Dossier de sortie (défaut : ./output)")

    # Tâches
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["pos", "backtrans", "combine"],
        default=["pos", "backtrans", "combine"],
        help="Étapes à exécuter",
    )

    # PoS
    parser.add_argument("--pos_n_samples", type=int, default=750,
                        help="Nb d'exemples PoS à générer (défaut : 750)")
    parser.add_argument("--pos_max_subs",  type=int, default=3,
                        help="Substitutions max par phrase, PoS (défaut : 3)")
    parser.add_argument("--spacy_model", default="en_core_web_sm",
                        help="Modèle spaCy (défaut : en_core_web_sm)")

    # Backtranslation
    parser.add_argument("--bt_n_samples", type=int, default=500,
                        help="Nb d'exemples backtrans à générer (défaut : 500)")
    parser.add_argument(
        "--pivot_langs", nargs="+", default=["fr"],
        choices=["fr", "es", "de", "it", "nl"],
        help="Langue(s) pivot (défaut : fr)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device pour la traduction (défaut : cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les traductions intermédiaires")

    # Divers
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire (défaut : 42)")

    return parser


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tasks    = set(args.tasks)
    pos_path = None
    bt_path  = None

    # Récupérer les fichiers existants si on ne les regénère pas
    if "pos" not in tasks:
        c = Path(args.output_dir) / "augmented_pos.jsonl"
        if c.exists():
            pos_path = c
            print(f"[main] Fichier PoS existant détecté : {c}")

    if "backtrans" not in tasks:
        c = Path(args.output_dir) / "augmented_backtrans.jsonl"
        if c.exists():
            bt_path = c
            print(f"[main] Fichier backtrans existant détecté : {c}")

    # Exécution
    if "pos"       in tasks: pos_path = run_pos(args)
    if "backtrans" in tasks: bt_path  = run_backtrans(args)
    if "combine"   in tasks: run_combine(args, pos_path, bt_path)

    # Résumé
    print("\n" + "=" * 60)
    print("[main] Pipeline terminé. Fichiers produits :")
    print("=" * 60)
    for fname in [
        "augmented_pos.jsonl",
        "augmented_backtrans.jsonl",
        "final_augmented_dataset.jsonl",
    ]:
        p = Path(args.output_dir) / fname
        if p.exists():
            n = sum(1 for _ in open(p, encoding="utf-8"))
            print(f"  {p}  ({n} lignes)")


if __name__ == "__main__":
    main()
