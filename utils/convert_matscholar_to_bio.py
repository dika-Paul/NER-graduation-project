import json
import random
from collections import OrderedDict
from pathlib import Path


SOURCE_PATH = Path("data/mat_scholar_ner.json")
TARGET_DIR = Path("data/matscholar")
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 42


def normalize_bio(labels):
    normalized = []
    prev_type = None
    prev_is_entity = False

    for label in labels:
        if label == "O":
            normalized.append(label)
            prev_type = None
            prev_is_entity = False
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            normalized.append(label)
            prev_type = entity_type
            prev_is_entity = True
            continue

        if prefix == "I" and prev_is_entity and prev_type == entity_type:
            normalized.append(label)
        else:
            normalized.append(f"B-{entity_type}")

        prev_type = entity_type
        prev_is_entity = True

    return normalized


def load_sentences(source_path):
    with source_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    sentences = OrderedDict()
    for record in records:
        sentence_id = record["sentence_id"]
        sentence = sentences.setdefault(sentence_id, {"tokens": [], "labels": []})
        sentence["tokens"].append(str(record["words"]))
        sentence["labels"].append(str(record["labels"]))

    results = []
    for sentence in sentences.values():
        labels = normalize_bio(sentence["labels"])
        results.append(list(zip(sentence["tokens"], labels)))
    return results


def write_bio(path, sentences):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")


def main():
    sentences = load_sentences(SOURCE_PATH)
    rng = random.Random(SEED)
    rng.shuffle(sentences)

    total = len(sentences)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_sentences = sentences[:train_end]
    valid_sentences = sentences[train_end:valid_end]
    test_sentences = sentences[valid_end:]

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    write_bio(TARGET_DIR / "train.txt", train_sentences)
    write_bio(TARGET_DIR / "valid.txt", valid_sentences)
    write_bio(TARGET_DIR / "test.txt", test_sentences)
    write_bio(TARGET_DIR / "all.txt", sentences)

    print(f"Total sentences: {total}")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Valid sentences: {len(valid_sentences)}")
    print(f"Test sentences: {len(test_sentences)}")
    print(f"Output directory: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
