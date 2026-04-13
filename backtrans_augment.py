"""
backtrans_augment.py
====================
Augmentation par backtranslation (EN → langue pivot → EN).


Référence
---------
Sennrich, R., Haddow, B., & Birch, A. (2016).
  Improving Neural Machine Translation Models with Monolingual Data.
  ACL 2016.  ← origine de la backtranslation comme technique d'augmentation

Wei, J., & Zou, K. (2019).
  EDA: Easy Data Augmentation Techniques for Boosting Performance on
  Text Classification Tasks. EMNLP 2019.  ← contexte augmentation NLP
"""

from __future__ import annotations

import random
from pathlib import Path

from data_utils import (
    collect_texts,
    deduplicate,
    deep_copy_record,
    load_jsonl,
    save_jsonl,
    validate_record,
)

# ---------------------------------------------------------------------------
# Paires de langues supportées
# ---------------------------------------------------------------------------

SUPPORTED_PIVOTS: dict[str, dict[str, str]] = {
    "fr": {
        "fwd": "Helsinki-NLP/opus-mt-en-fr",
        "bwd": "Helsinki-NLP/opus-mt-fr-en",
        "name": "French",
    },
    "es": {
        "fwd": "Helsinki-NLP/opus-mt-en-es",
        "bwd": "Helsinki-NLP/opus-mt-es-en",
        "name": "Spanish",
    },
    "de": {
        "fwd": "Helsinki-NLP/opus-mt-en-de",
        "bwd": "Helsinki-NLP/opus-mt-de-en",
        "name": "German",
    },
    "it": {
        "fwd": "Helsinki-NLP/opus-mt-en-it",
        "bwd": "Helsinki-NLP/opus-mt-it-en",
        "name": "Italian",
    },
    "nl": {
        "fwd": "Helsinki-NLP/opus-mt-en-nl",
        "bwd": "Helsinki-NLP/opus-mt-nl-en",
        "name": "Dutch",
    },
}


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class BacktranslationAugmenter:
    """
    Augmente un dataset DimABSA via traduction aller-retour.

    Parameters
    ----------
    pivot_langs : list[str]
        Langues pivot à utiliser, ex. ["fr", "es"].
        Plusieurs langues → plus de diversité (le script alterne).
        Valeur par défaut : ["fr"] (français seulement).
    device : str
        "cpu" ou "cuda". Sur CPU, compter ~5–15 sec/phrase selon le modèle.
    max_length : int
        Longueur max (tokens) pour la traduction. 128 suffit pour des avis
        courts de restaurant.
    rng_seed : int
        Graine aléatoire pour la reproductibilité.
    verbose : bool
        Affiche la traduction intermédiaire si True (utile pour débugger).
    """

    def __init__(
        self,
        pivot_langs: list[str] | None = None,
        device: str = "cuda",
        max_length: int = 128,
        rng_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.pivot_langs = pivot_langs or ["fr"]
        self.device      = device
        self.max_length  = max_length
        self.rng         = random.Random(rng_seed)
        self.verbose     = verbose

        # Validation des langues demandées
        for lang in self.pivot_langs:
            if lang not in SUPPORTED_PIVOTS:
                raise ValueError(
                    f"Langue pivot '{lang}' non supportée. "
                    f"Choix disponibles : {list(SUPPORTED_PIVOTS.keys())}"
                )

        # Cache des pipelines chargés (chargement paresseux)
        # clé : "fr_fwd", "fr_bwd", "es_fwd", etc.
        self._pipes: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Chargement des modèles (lazy)
    # ------------------------------------------------------------------

    def _get_pipe(self, lang: str, direction: str):
        """
        Retourne le pipeline de traduction pour la langue et direction données.
        Charge le modèle depuis HuggingFace si pas encore en cache.
        """
        key = f"{lang}_{direction}"
        if key not in self._pipes:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError:
                raise ImportError(
                    "Le module 'transformers' est requis pour la backtranslation.\n"
                    "Installe-le avec :  pip install transformers sentencepiece sacremoses"
                )

            model_id = SUPPORTED_PIVOTS[lang][direction]
            device_id = 0 if self.device == "cuda" else -1
            print(f"[backtrans] Chargement du modèle {model_id} …")
            self._pipes[key] = hf_pipeline(
                "translation",
                model=model_id,
                device=device_id,
            )
            print(f"[backtrans] Modèle chargé ✓")

        return self._pipes[key]

    # ------------------------------------------------------------------
    # Traduction d'une seule phrase
    # ------------------------------------------------------------------

    def translate(self, text: str, lang: str, direction: str) -> str | None:
        """
        Traduit *text* dans la direction indiquée ("fwd" ou "bwd").
        Retourne None en cas d'échec.
        """
        pipe = self._get_pipe(lang, direction)
        try:
            result = pipe(text, max_length=self.max_length)
            return result[0]["translation_text"].strip()
        except Exception as exc:
            print(f"  [backtrans] Erreur de traduction ({lang} {direction}) : {exc}")
            return None

    # ------------------------------------------------------------------
    # Backtranslation d'un record
    # ------------------------------------------------------------------

    def augment_sentence(self, record: dict, lang: str) -> dict | None:
        """
        Génère une variante backtraduite de *record* via la langue *lang*.

        - La phrase est traduite EN → lang → EN.
        - Le Quadruplet (Aspect, Category, Opinion, VA) est copié tel quel.
        - Retourne None si la backtranslation est identique à l'original
          ou si une erreur survient.
        """
        original_text = record["Text"]

        # Étape 1 : EN → pivot
        pivot_text = self.translate(original_text, lang, "fwd")
        if pivot_text is None:
            return None

        # Étape 2 : pivot → EN
        back_text = self.translate(pivot_text, lang, "bwd")
        if back_text is None:
            return None

        if self.verbose:
            lang_name = SUPPORTED_PIVOTS[lang]["name"]
            print(f"  Original  : {original_text}")
            print(f"  → {lang_name:8s}: {pivot_text}")
            print(f"  → EN      : {back_text}")

        # Rejeter si la phrase est identique ou quasi-identique
        if back_text.strip().lower() == original_text.strip().lower():
            return None

        # Construire le nouveau record
        new_rec = deep_copy_record(record)
        new_rec["Text"] = back_text
        # Quadruplet (VA inclus) : copié sans modification
        return new_rec

    # ------------------------------------------------------------------
    # Pipeline complet sur un dataset
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        input_path: str | Path,
        output_path: str | Path,
        n_samples: int = 500,
    ) -> list[dict]:
        """
        Génère *n_samples* exemples augmentés par backtranslation.

        Si plusieurs langues pivot sont configurées, elles sont utilisées
        en alternance pour maximiser la diversité lexicale.

        Parameters
        ----------
        input_path  : fichier source .jsonl
        output_path : destination augmented_backtrans.jsonl
        n_samples   : nombre d'exemples cible

        Returns
        -------
        Liste des records augmentés (aussi écrits dans output_path).
        """
        print(f"[backtrans] Chargement depuis {input_path} …")
        records = [r for r in load_jsonl(input_path) if validate_record(r)]
        print(f"[backtrans] {len(records)} exemples valides chargés.")
        print(f"[backtrans] Langues pivot : {self.pivot_langs}")
        print(f"[backtrans] Cible : {n_samples} exemples augmentés.")

        existing_texts = collect_texts(records)
        augmented: list[dict]  = []
        new_texts: set[str]    = set()

        # On mélange le pool pour ne pas toujours partir des mêmes phrases
        pool = records[:]
        self.rng.shuffle(pool)

        pool_idx  = 0
        attempts  = 0
        max_attempts = n_samples * 8  # garde-fou contre la boucle infinie

        while len(augmented) < n_samples and attempts < max_attempts:
            record = pool[pool_idx % len(pool)]
            pool_idx += 1
            attempts += 1

            # Choisir la langue pivot en rotation
            lang = self.pivot_langs[(pool_idx - 1) % len(self.pivot_langs)]

            tag = f"[{attempts}|{len(augmented)}/{n_samples}]"
            print(f"{tag} {lang.upper()} ← {record['Text'][:60]}…")

            new_rec = self.augment_sentence(record, lang)

            if new_rec is None:
                print(f"{tag} → identique ou erreur, skip.")
                continue

            # Déduplication
            norm = " ".join(new_rec["Text"].lower().split())
            if norm in existing_texts or norm in new_texts:
                print(f"{tag} → doublon, skip.")
                continue

            new_rec["ID"] = f"backtrans_aug_{len(augmented):05d}"
            augmented.append(new_rec)
            new_texts.add(norm)
            print(f"{tag} ✓  → {new_rec['Text'][:60]}…")

        print(
            f"\n[backtrans] Terminé : {len(augmented)} / {n_samples} générés "
            f"({attempts} tentatives)."
        )

        # Déduplication finale (sécurité)
        augmented = deduplicate(augmented, existing_texts=existing_texts)
        save_jsonl(augmented, output_path)
        return augmented


# ---------------------------------------------------------------------------
# Fonction d'interface (même signature que pos_augment.augment_dataset)
# ---------------------------------------------------------------------------

def augment_dataset(
    input_path: str | Path,
    output_path: str | Path,
    n_samples: int = 500,
    pivot_langs: list[str] | None = None,
    device: str = "cuda",
    rng_seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    """
    Point d'entrée principal — même interface que pos_augment.augment_dataset.

    Parameters
    ----------
    input_path   : fichier source .jsonl
    output_path  : destination augmented_backtrans.jsonl
    n_samples    : nombre d'exemples cible (recommandé : 300–700)
    pivot_langs  : ex. ["fr", "es"] — None → ["fr"] seulement
    device       : "cpu" ou "cuda"
    rng_seed     : graine aléatoire
    verbose      : afficher les traductions intermédiaires

    Returns
    -------
    Liste des records augmentés.
    """
    aug = BacktranslationAugmenter(
        pivot_langs = pivot_langs or ["fr"],
        device      = device,
        rng_seed    = rng_seed,
        verbose     = verbose,
    )
    return aug.augment_dataset(
        input_path  = input_path,
        output_path = output_path,
        n_samples   = n_samples,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtranslation data augmentation for DimABSA"
    )
    parser.add_argument("--input",    required=True, help="Input .jsonl path")
    parser.add_argument("--output",   default="augmented_backtrans.jsonl")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument(
        "--pivot_langs", nargs="+", default=["fr"],
        choices=list(SUPPORTED_PIVOTS.keys()),
        help="Pivot language(s). Ex: --pivot_langs fr es de",
    )
    parser.add_argument("--device",  default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les traductions intermédiaires")
    args = parser.parse_args()

    augment_dataset(
        input_path  = args.input,
        output_path = args.output,
        n_samples   = args.n_samples,
        pivot_langs = args.pivot_langs,
        device      = args.device,
        rng_seed    = args.seed,
        verbose     = args.verbose,
    )
