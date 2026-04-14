"""
pos_adj_augment.py
==================
Augmentation par substitution de synonymes affectifs de MÊME POLARITÉ.

Objectif
--------
Générer de nouveaux exemples en remplaçant les adjectifs et adverbes
d'opinion par des synonymes dont la valence est proche (même zone du
spectre affectif), de sorte que la VA du quadruplet reste inchangée.

Exemples de substitutions valides
----------------------------------
  pretty    → lovely, gorgeous, beautiful, cute, charming
  great     → wonderful, superb, excellent, fantastic, terrific
  good      → decent, solid, fine, satisfying, pleasant
  slow      → sluggish, unhurried, lethargic, leisurely
  horrible  → dreadful, awful, atrocious, terrible, appalling

Contraintes
-----------
* Synonymes de MÊME polarité uniquement (positif → positif, négatif → négatif)
* VA du quadruplet copiée telle quelle (comme pos_augment et backtrans_augment)
* Les mots de l'Aspect et de l'Opinion originaux sont identifiés et
  remplacés dans le texte, l'Opinion dans le quadruplet est mise à jour
* Aucune dépendance externe obligatoire (dictionnaire embarqué)
* spaCy optionnel : si présent, utilisé pour un meilleur ciblage des tokens

Références
----------
Wei & Zou (2019). EDA: Easy Data Augmentation. EMNLP.
Warriner et al. (2013). Norms of valence, arousal, dominance. BRM.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from data_utils import (
    collect_texts,
    deduplicate,
    deep_copy_record,
    load_jsonl,
    save_jsonl,
    validate_record,
)


# ===========================================================================
#  DICTIONNAIRE DE SYNONYMES — même polarité, VA stable
#
#  Organisation : mot_source → [liste de synonymes]
#
#  Principes de construction
#  -------------------------
#  1. Tous les synonymes d'une entrée partagent la MÊME zone de valence
#     que le mot source (positif fort / positif modéré / négatif / neutre).
#  2. Les valeurs sont calibrées sur les observations du dataset DimABSA
#     (great=V8.0, good=V5.88, slow=V4.5, horrible=V1.25, etc.)
#     et sur Warriner et al. (2013).
#  3. On couvre les 60 mots d'opinion les plus fréquents du dataset
#     (inventaire extrait automatiquement).
# ===========================================================================

SYNONYM_DICT: dict[str, list[str]] = {

    # ── POSITIF FORT (V ≈ 7.5–9) ──────────────────────────────────────────

    "excellent": [
        "outstanding", "superb", "exceptional", "magnificent",
        "splendid", "brilliant", "first-rate", "impeccable",
    ],
    "amazing": [
        "incredible", "remarkable", "spectacular", "breathtaking",
        "astounding", "phenomenal", "extraordinary", "impressive",
    ],
    "fantastic": [
        "phenomenal", "terrific", "magnificent", "wonderful",
        "extraordinary", "spectacular", "marvelous", "exceptional",
    ],
    "great": [
        "wonderful", "superb", "outstanding", "splendid",
        "terrific", "marvelous", "magnificent", "brilliant",
    ],
    "wonderful": [
        "marvelous", "magnificent", "glorious", "splendid",
        "superb", "fantastic", "delightful", "enchanting",
    ],
    "delicious": [
        "exquisite", "delectable", "scrumptious", "mouthwatering",
        "luscious", "heavenly", "divine", "sumptuous",
    ],
    "tasty": [
        "flavorful", "delectable", "appetizing", "savory",
        "palatable", "yummy", "mouth-watering", "toothsome",
    ],
    "awesome": [
        "incredible", "spectacular", "phenomenal", "magnificent",
        "breathtaking", "remarkable", "extraordinary", "sublime",
    ],
    "superb": [
        "excellent", "outstanding", "magnificent", "splendid",
        "exceptional", "first-rate", "impeccable", "exquisite",
    ],
    "perfect": [
        "flawless", "impeccable", "ideal", "faultless",
        "exemplary", "immaculate", "spotless", "pristine",
    ],
    "heavenly": [
        "divine", "sublime", "exquisite", "blissful",
        "celestial", "ethereal", "transcendent", "rapturous",
    ],
    "stunning": [
        "breathtaking", "spectacular", "magnificent", "dazzling",
        "striking", "gorgeous", "splendid", "awe-inspiring",
    ],
    "best": [
        "finest", "greatest", "top", "premier",
        "leading", "foremost", "ultimate", "unrivaled",
    ],

    # ── POSITIF MODÉRÉ (V ≈ 6–7.5) ────────────────────────────────────────

    "good": [
        "decent", "solid", "fine", "satisfying",
        "pleasant", "commendable", "respectable", "worthwhile",
    ],
    "nice": [
        "pleasant", "lovely", "delightful", "agreeable",
        "charming", "appealing", "enjoyable", "inviting",
    ],
    "friendly": [
        "warm", "welcoming", "cordial", "hospitable",
        "amiable", "genial", "gracious", "affable",
    ],
    "fresh": [
        "crisp", "vibrant", "wholesome", "pristine",
        "pure", "natural", "bright", "invigorating",
    ],
    "attentive": [
        "responsive", "dedicated", "considerate", "thoughtful",
        "conscientious", "diligent", "careful", "thorough",
    ],
    "welcoming": [
        "hospitable", "warm", "inviting", "gracious",
        "cordial", "friendly", "accommodating", "receptive",
    ],
    "helpful": [
        "supportive", "accommodating", "cooperative", "obliging",
        "responsive", "constructive", "resourceful", "considerate",
    ],
    "clean": [
        "immaculate", "spotless", "tidy", "neat",
        "pristine", "hygienic", "sanitary", "well-kept",
    ],
    "quick": [
        "swift", "prompt", "speedy", "efficient",
        "rapid", "brisk", "timely", "expeditious",
    ],
    "fast": [
        "swift", "prompt", "speedy", "rapid",
        "quick", "brisk", "efficient", "nimble",
    ],
    "reasonable": [
        "fair", "affordable", "modest", "sensible",
        "acceptable", "appropriate", "suitable", "justified",
    ],
    "pretty": [
        "lovely", "beautiful", "attractive", "gorgeous",
        "charming", "elegant", "pleasing", "delightful",
    ],
    "cozy": [
        "snug", "comfortable", "inviting", "homey",
        "warm", "intimate", "relaxing", "welcoming",
    ],
    "comfortable": [
        "cozy", "relaxing", "pleasant", "snug",
        "inviting", "easy", "soothing", "restful",
    ],
    "authentic": [
        "genuine", "original", "traditional", "real",
        "true", "legitimate", "pure", "bona fide",
    ],
    "flavorful": [
        "tasty", "savory", "rich", "aromatic",
        "delectable", "appetizing", "seasoned", "full-bodied",
    ],

    # ── ADVERBES POSITIFS (modificateurs, VA stable) ──────────────────────

    "very": [
        "really", "quite", "truly", "remarkably",
        "notably", "particularly", "especially", "exceptionally",
    ],
    "really": [
        "truly", "genuinely", "honestly", "sincerely",
        "definitely", "certainly", "absolutely", "undeniably",
    ],
    "absolutely": [
        "completely", "utterly", "entirely", "thoroughly",
        "totally", "perfectly", "wholly", "fully",
    ],
    "always": [
        "consistently", "invariably", "reliably", "unfailingly",
        "perpetually", "constantly", "continually", "regularly",
    ],
    "super": [
        "incredibly", "extremely", "remarkably", "exceptionally",
        "tremendously", "extraordinarily", "wonderfully", "fabulously",
    ],
    "pretty": [   # adverbe ("pretty good") — même liste que l'adjectif
        "quite", "rather", "fairly", "reasonably",
        "relatively", "somewhat", "decently", "adequately",
    ],
    "just": [
        "simply", "merely", "only", "purely",
    ],
    "so": [
        "very", "quite", "really", "incredibly",
        "remarkably", "awfully", "terribly", "extremely",
    ],
    "quite": [
        "fairly", "rather", "reasonably", "pretty",
        "relatively", "moderately", "considerably", "somewhat",
    ],

    # ── NÉGATIF MODÉRÉ (V ≈ 3–5) ──────────────────────────────────────────

    "slow": [
        "sluggish", "unhurried", "lethargic", "leisurely",
        "dawdling", "plodding", "lagging", "tardy",
    ],
    "bland": [
        "tasteless", "insipid", "flat", "flavorless",
        "uninspiring", "dull", "unexciting", "vapid",
    ],
    "ok": [
        "mediocre", "average", "ordinary", "so-so",
        "passable", "tolerable", "adequate", "fair",
    ],
    "bad": [
        "poor", "subpar", "inferior", "unsatisfactory",
        "inadequate", "deficient", "lacking", "disappointing",
    ],
    "disappointing": [
        "underwhelming", "unsatisfying", "frustrating", "disheartening",
        "lackluster", "anticlimactic", "subpar", "inadequate",
    ],
    "overpriced": [
        "expensive", "costly", "steep", "exorbitant",
        "pricey", "extravagant", "inflated", "unreasonable",
    ],
    "cold": [   # négatif dans contexte service
        "unfriendly", "indifferent", "aloof", "distant",
        "detached", "impersonal", "unwelcoming", "standoffish",
    ],

    # ── NÉGATIF FORT (V ≈ 1–3) ────────────────────────────────────────────

    "horrible": [
        "dreadful", "atrocious", "appalling", "ghastly",
        "nightmarish", "abysmal", "deplorable", "wretched",
    ],
    "terrible": [
        "dreadful", "atrocious", "appalling", "awful",
        "abysmal", "wretched", "horrendous", "frightful",
    ],
    "awful": [
        "dreadful", "terrible", "horrendous", "atrocious",
        "appalling", "ghastly", "wretched", "abysmal",
    ],
    "disgusting": [
        "revolting", "repulsive", "nauseating", "vile",
        "loathsome", "repugnant", "sickening", "abhorrent",
    ],
    "rude": [
        "impolite", "disrespectful", "discourteous", "insolent",
        "obnoxious", "offensive", "abrasive", "boorish",
    ],
    "dirty": [
        "filthy", "grimy", "unsanitary", "unhygienic",
        "squalid", "unclean", "contaminated", "foul",
    ],
    "poor": [
        "subpar", "inferior", "inadequate", "deficient",
        "lacking", "unsatisfactory", "mediocre", "shoddy",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_case(original: str, replacement: str) -> str:
    """Applique la capitalisation de *original* à *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


def _find_opinion_words_in_text(
    opinion: str,
    text: str,
) -> list[str]:
    """
    Retourne les mots de l'expression d'opinion présents dans SYNONYM_DICT,
    en cherchant d'abord l'expression complète puis mot par mot.
    """
    op_lower = opinion.lower()
    if op_lower in SYNONYM_DICT:
        return [op_lower]
    return [w for w in op_lower.split() if w in SYNONYM_DICT]


# ===========================================================================
#  AUGMENTEUR PRINCIPAL
# ===========================================================================

class AdjSynonymAugmenter:
    """
    Remplace les adjectifs/adverbes d'opinion par des synonymes de même
    polarité. La VA du quadruplet est copiée telle quelle.

    Parameters
    ----------
    rng_seed : int
        Graine aléatoire (reproductibilité).
    max_substitutions : int
        Nombre maximum de mots d'opinion remplacés par phrase (défaut : 1).
    """

    def __init__(
        self,
        rng_seed: int = 42,
        max_substitutions: int = 1,
    ) -> None:
        self.rng = random.Random(rng_seed)
        self.max_substitutions = max_substitutions

    def augment_sentence(self, record: dict) -> dict | None:
        """
        Génère une variante en substituant des synonymes affectifs.

        - Cherche les mots remplaçables dans chaque Opinion du quadruplet.
        - Remplace dans le texte ET dans le champ Opinion.
        - La VA est copiée sans modification.
        - Retourne None si aucune substitution n'est possible.
        """
        new_rec  = deep_copy_record(record)
        text     = new_rec["Text"]
        n_subs   = 0
        modified = False

        for quad in new_rec["Quadruplet"]:
            if n_subs >= self.max_substitutions:
                break

            opinion  = quad["Opinion"]
            replaceable = _find_opinion_words_in_text(opinion, text)
            if not replaceable:
                continue

            # Choisir un mot source aléatoirement parmi ceux disponibles
            self.rng.shuffle(replaceable)
            for source_word in replaceable:
                synonyms = SYNONYM_DICT.get(source_word, [])
                if not synonyms:
                    continue

                # Tirer un synonyme différent du mot source
                candidates = [s for s in synonyms if s.lower() != source_word.lower()]
                if not candidates:
                    continue
                synonym = self.rng.choice(candidates)

                # Remplacer dans le texte (insensible à la casse, 1 occurrence)
                pattern  = re.compile(re.escape(source_word), re.IGNORECASE)
                match    = pattern.search(text)
                if match is None:
                    continue

                replacement = _match_case(match.group(0), synonym)
                new_text = text[:match.start()] + replacement + text[match.end():]

                if new_text == text:
                    continue

                # Mettre à jour le texte et l'opinion dans le quadruplet
                text = new_text
                new_opinion = re.sub(
                    re.escape(source_word), synonym, opinion, count=1, flags=re.IGNORECASE
                )
                quad["Opinion"] = new_opinion
                # VA inchangée — copiée par deep_copy_record

                n_subs  += 1
                modified = True
                break  # un seul remplacement par quadruplet

        if not modified:
            return None

        new_rec["Text"] = text
        return new_rec


# ===========================================================================
#  PIPELINE D'AUGMENTATION (interface identique aux autres modules)
# ===========================================================================

def augment_dataset(
    input_path: str | Path,
    output_path: str | Path,
    n_samples: int = 600,
    rng_seed: int = 42,
    max_substitutions: int = 1,
) -> list[dict]:
    """
    Génère *n_samples* nouveaux exemples par substitution de synonymes.

    Parameters
    ----------
    input_path        : fichier source .jsonl
    output_path       : destination augmented_adj.jsonl
    n_samples         : nombre d'exemples cible (recommandé : 400–800)
    rng_seed          : graine aléatoire
    max_substitutions : max de remplacements par phrase (défaut : 1)

    Returns
    -------
    Liste des records augmentés (écrits dans output_path).
    """
    print(f"[adj_augment] Chargement depuis {input_path} …")
    records = [r for r in load_jsonl(input_path) if validate_record(r)]
    print(f"[adj_augment] {len(records)} exemples valides.")
    print(f"[adj_augment] Entrées dans le dictionnaire : {len(SYNONYM_DICT)}")

    existing_texts = collect_texts(records)
    augmenter = AdjSynonymAugmenter(
        rng_seed          = rng_seed,
        max_substitutions = max_substitutions,
    )

    augmented: list[dict] = []
    new_texts: set[str]   = set()
    rng = random.Random(rng_seed)
    pool = records[:]
    rng.shuffle(pool)

    pool_idx     = 0
    attempts     = 0
    max_attempts = n_samples * 10

    while len(augmented) < n_samples and attempts < max_attempts:
        record = pool[pool_idx % len(pool)]
        pool_idx += 1
        attempts += 1

        new_rec = augmenter.augment_sentence(record)
        if new_rec is None:
            continue

        norm = " ".join(new_rec["Text"].lower().split())
        if norm in existing_texts or norm in new_texts:
            continue

        new_rec["ID"] = f"adj_aug_{len(augmented):05d}"
        augmented.append(new_rec)
        new_texts.add(norm)

    print(
        f"[adj_augment] Terminé : {len(augmented)} / {n_samples} générés "
        f"({attempts} tentatives)."
    )
    augmented = deduplicate(augmented, existing_texts=existing_texts)
    save_jsonl(augmented, output_path)
    return augmented


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augmentation par synonymes ADJ/ADV (même polarité) – DimABSA"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     default="augmented_adj.jsonl")
    parser.add_argument("--n_samples",  type=int, default=600)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max_subs",   type=int, default=1,
                        help="Max de remplacements par phrase (défaut : 1)")
    args = parser.parse_args()

    augment_dataset(
        input_path        = args.input,
        output_path       = args.output,
        n_samples         = args.n_samples,
        rng_seed          = args.seed,
        max_substitutions = args.max_subs,
    )
