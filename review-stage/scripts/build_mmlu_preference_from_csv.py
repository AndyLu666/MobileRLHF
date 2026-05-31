import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List


LETTERS = ("A", "B", "C", "D")


def iter_rows(split_root: Path) -> Iterable[dict]:
    for path in sorted(split_root.glob("*.csv")):
        subject = path.stem
        with path.open("r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            for row in reader:
                if len(row) < 6:
                    continue
                answer = row[5].strip()[:1].upper()
                if answer not in LETTERS:
                    continue
                yield {
                    "subject": subject,
                    "question": row[0],
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                    "answer": answer,
                }


def format_prompt(row: dict) -> str:
    return (
        f"Question: {row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
        f"Answer:"
    )


def build_rows(split_root: Path, limit: int, seed: int) -> List[dict]:
    rows = list(iter_rows(split_root))
    random.Random(seed).shuffle(rows)
    rows = rows[:limit]
    out = []
    for idx, row in enumerate(rows):
        wrong = [letter for letter in LETTERS if letter != row["answer"]]
        rejected = random.Random(seed + idx).choice(wrong)
        out.append(
            {
                "prompt": format_prompt(row),
                "chosen": f" {row['answer']}",
                "rejected": f" {rejected}",
                "subject": row["subject"],
                "source_split": split_root.name,
                "generator_mode": "mcq_letter_real_mmlu",
            }
        )
    return out


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference JSONL from restored real MMLU CSV files")
    parser.add_argument("--mmlu-root", default="data/mmlu/data")
    parser.add_argument("--train-split", default="auxiliary_train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--eval-size", type=int, default=120)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.mmlu_root)
    train_rows = build_rows(root / args.train_split, args.train_size, args.seed)
    eval_rows = build_rows(root / args.eval_split, args.eval_size, args.seed + 10000)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "train_split": args.train_split,
                "eval_split": args.eval_split,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
