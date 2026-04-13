#!/usr/bin/env python3
"""Contextual augmentation for DimABSA JSONL training files."""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Permet d'importer le package local nlpaug depuis la racine du repo.
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nlpaug-matplotlib")

NULL_VALUE = "NULL"


def read_jsonl(path):
    # Lit le fichier JSONL ligne par ligne pour limiter l'usage memoire.
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def write_jsonl(path, records):
    # Cree le dossier cible si besoin puis ecrit un objet JSON par ligne.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def protected_phrases(record):
    # Extrait les expressions (Aspect/Opinion) a ne pas perdre pendant l'augmentation.
    phrases = []
    for quad in record.get("Quadruplet", []):
        for key in ("Aspect", "Opinion"):
            value = quad.get(key)
            if value and value != NULL_VALUE and value not in phrases:
                phrases.append(value)
    return phrases


def present_phrases(text, phrases):
    # Ne garde que les expressions effectivement presentes dans le texte source.
    lowered = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in lowered]


def preserves_phrases(text, phrases):
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)


def set_record_stopwords(aug, phrases):
    # Configure les stopwords du modele pour eviter de modifier les expressions protegees.
    stopwords = [phrase.lower() for phrase in phrases] if "uncased" in aug.model_path else phrases
    aug.stopwords = stopwords
    aug.stopword_reg = None
    aug.reserve_word_reg = None
    aug._build_stop_words(stopwords)


def augment_record(record, aug, num_aug, retries):
    text = record["Text"]
    # Expressions critiques a conserver dans toutes les variantes generees.
    phrases_to_keep = present_phrases(text, protected_phrases(record))
    augmented_records = []
    # Evite les doublons entre texte original et textes augmentes.
    seen_texts = {text}

    set_record_stopwords(aug, phrases_to_keep)

    for aug_idx in range(1, num_aug + 1):
        augmented_text = None
        for _ in range(retries):

            candidate = aug.augment(text, n=1)[0]
            if candidate in seen_texts:
                continue
            if not preserves_phrases(candidate, phrases_to_keep):
                continue
            augmented_text = candidate
            break

        if augmented_text is None:
            continue

        seen_texts.add(augmented_text)
        # Copie complete pour conserver tous les champs du record original.
        augmented_record = copy.deepcopy(record)
        augmented_record["ID"] = f"{record['ID']}_ctxaug_{aug_idx}"
        augmented_record["Text"] = augmented_text
        augmented_records.append(augmented_record)

    return augmented_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create contextual-augmentation copies for DimABSA JSONL data."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument("output", type=Path, help="Output JSONL file.")
    parser.add_argument("--model-path", default="distilbert-base-uncased")
    parser.add_argument("--action", choices=["substitute", "insert"], default="substitute")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--aug-p", type=float, default=0.15)
    parser.add_argument("--aug-min", type=int, default=1)
    parser.add_argument("--aug-max", type=int, default=3)
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Write only augmented records instead of original plus augmented records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a quick smoke run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import numpy as np

        """
            NOTE PORTABILITE
            Ce script utilise la librairie nlpaug pour l'augmentation contextuelle.
            Si nlpaug n'est pas disponible dans un autre repo/environnement, on peut
            temporairement utiliser un stub local qui expose le meme chemin d'import
            et la classe ContextualWordEmbsAug pour eviter ModuleNotFoundError.
            IMPORTANT:
            - Le stub sert uniquement a la compatibilite d'import et aux tests.
            - Le stub ne fait pas de vraie augmentation NLP.
            Pour un usage reel, installer la librairie officielle:
            - Projet: nlpaug - Natural language processing augmentation library for deep neural networks.
            - Auteur: Edward Ma - makcedward@gmail.com
            - GitHub: https://github.com/makcedward/nlpaug
        """
        import nlpaug.augmenter.word as naw
    except ImportError as exc:
        raise SystemExit(
            "Missing augmentation dependencies. Install the base repo requirements and "
            "contextual augmenter extras first, for example: "
            "python3 -m pip install -r requirements.txt 'torch>=1.6.0' "
            "'transformers>=4.11.3' sentencepiece"
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)

    records = list(read_jsonl(args.input))
    if args.limit is not None:

        records = records[: args.limit]

    aug = naw.ContextualWordEmbsAug(
        model_path=args.model_path,
        action=args.action,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        aug_p=args.aug_p,
        aug_min=args.aug_min,
        aug_max=args.aug_max,
        silence=True,
    )

    output_records = [] if args.no_original else copy.deepcopy(records)
    augmented_count = 0

    for idx, record in enumerate(records, start=1):
        augmented = augment_record(record, aug, args.num_aug, args.retries)
        output_records.extend(augmented)
        augmented_count += len(augmented)

        if idx % 50 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} records; kept {augmented_count} augmentations.")

    write_jsonl(args.output, output_records)
    print(f"Wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()
