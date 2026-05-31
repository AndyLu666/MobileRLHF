import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


ANSWER_LETTERS = ("A", "B", "C", "D")
SPLIT_MAP = {
    "dev": "dev",
    "test": "test",
    "validation": "val",
    "auxiliary_train": "auxiliary_train",
}


def normalize_answer(value) -> str:
    if isinstance(value, int):
        return ANSWER_LETTERS[value]
    text = str(value).strip()
    if text.isdigit():
        return ANSWER_LETTERS[int(text)]
    return text[:1].upper()


def subject_filename(subject: str, split_dir: str) -> str:
    safe = subject.replace(" ", "_")
    return f"{safe}_{split_dir}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore MMLU CSV files from Hugging Face cais/mmlu")
    parser.add_argument("--output-root", default="data/mmlu/data")
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--config", default="all")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config)
    counts = {}

    for hf_split, split_dir in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        grouped = defaultdict(list)
        for row in ds[hf_split]:
            subject = row["subject"]
            grouped[subject].append(
                [
                    row["question"],
                    row["choices"][0],
                    row["choices"][1],
                    row["choices"][2],
                    row["choices"][3],
                    normalize_answer(row["answer"]),
                ]
            )

        split_root = output_root / split_dir
        split_root.mkdir(parents=True, exist_ok=True)
        for subject, rows in grouped.items():
            with (split_root / subject_filename(subject, split_dir)).open("w", encoding="utf-8", newline="") as fout:
                writer = csv.writer(fout)
                writer.writerows(rows)
        counts[split_dir] = sum(len(rows) for rows in grouped.values())

    print(json.dumps({"status": "ok", "output_root": str(output_root), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
