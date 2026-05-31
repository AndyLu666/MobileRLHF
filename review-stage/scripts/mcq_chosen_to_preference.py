import argparse
import json
import random
from pathlib import Path


LETTERS = ("A", "B", "C", "D")


def pick_rejected(chosen: str, seed: int) -> str:
    normalized = chosen.strip().upper()
    if normalized not in LETTERS:
        raise ValueError(f"Expected chosen answer in {LETTERS}, got: {chosen!r}")
    wrong = [letter for letter in LETTERS if letter != normalized]
    return " " + random.Random(seed).choice(wrong)


def convert(input_path: Path, output_path: Path, seed: int) -> None:
    rows = []
    for idx, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append(
            {
                "prompt": rec["prompt"],
                "chosen": rec["chosen"],
                "rejected": pick_rejected(rec["chosen"], seed + idx),
                "generator_mode": "mcq_letter",
            }
        )

    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "count": len(rows), "output": str(output_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert chosen-only MCQ data into preference pairs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output), args.seed)


if __name__ == "__main__":
    main()
