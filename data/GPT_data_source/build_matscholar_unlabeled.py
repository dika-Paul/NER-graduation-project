import argparse
import json
from pathlib import Path


DEFAULT_SOURCE_PATH = Path("../matscholar/train.txt")
DEFAULT_UNLABELED_OUTPUT_PATH = Path(
    "matscholar_data/train_unlabeled.jsonl"
)
DEFAULT_MANUAL_OUTPUT_PATH = Path(
    "matscholar_data/train_labeled.txt"
)
DEFAULT_MANUAL_COUNT = 800


def read_bio_sentences(path: Path) -> list[list[tuple[str, str]]]:
    """Read a two-column BIO file and return sentence-level token-label pairs."""
    sentences: list[list[tuple[str, str]]] = []
    sentence: list[tuple[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            label = parts[-1]
            sentence.append((token, label))

    if sentence:
        sentences.append(sentence)

    return sentences


def write_unlabeled_jsonl(
    sentences: list[list[tuple[str, str]]],
    output_path: Path,
) -> None:
    """Write unlabeled sentence samples for GPT-assisted annotation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, sentence in enumerate(sentences):
            tokens = [token for token, _ in sentence]
            sample = {
                "sample_id": f"matscholar-train-{index:06d}",
                "text": " ".join(tokens),
                "tokens": tokens,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_manual_bio(
    sentences: list[list[tuple[str, str]]],
    output_path: Path,
) -> None:
    """Write selected sentences in the original BIO format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MatScholar BIO train data into unlabeled JSONL and "
            "export the last N sentences as manual BIO data."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Path to the source BIO file.",
    )
    parser.add_argument(
        "--unlabeled-output",
        type=Path,
        default=DEFAULT_UNLABELED_OUTPUT_PATH,
        help="Path to the output unlabeled JSONL file.",
    )
    parser.add_argument(
        "--manual-output",
        type=Path,
        default=DEFAULT_MANUAL_OUTPUT_PATH,
        help="Path to the output manual BIO file.",
    )
    parser.add_argument(
        "--manual-count",
        type=int,
        default=DEFAULT_MANUAL_COUNT,
        help="Number of tail sentences to export as manual BIO data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = read_bio_sentences(args.source)
    manual_count = min(args.manual_count, len(sentences))
    manual_sentences = sentences[-manual_count:]

    write_unlabeled_jsonl(sentences, args.unlabeled_output)
    write_manual_bio(manual_sentences, args.manual_output)

    print(f"Source: {args.source}")
    print(f"Unlabeled output: {args.unlabeled_output}")
    print(f"Manual BIO output: {args.manual_output}")
    print(f"Unlabeled samples: {len(sentences)}")
    print(f"Manual BIO samples: {len(manual_sentences)}")


if __name__ == "__main__":
    main()
