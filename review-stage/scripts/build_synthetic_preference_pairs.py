import argparse
import json
from pathlib import Path


NEAR_MISS = {
    "4": "5",
    "7": "8",
    "9": "8",
    "GOOD": "GREAT",
    "HELLO": "HALLO",
    "THANKS": "THANK",
    "YES": "YEP",
    "white": "light",
    "green": "gray",
    "cold": "cool",
    "down": "below",
    "Wednesday": "Tuesday",
    "6": "8",
    "C": "D",
    "5": "6",
    "Paris": "Lyon",
    "Tokyo": "Kyoto",
    "Rome": "Milan",
    "cat": "kitten",
    "dog": "puppy",
    "sun": "star",
    "moon": "satellite",
    "summer": "spring",
    "autumn": "fall",
    "February": "Febuary",
    "Saturday": "Sunday",
    "triangle": "pyramid",
    "square": "rectangle",
    "water": "juice",
    "oxygen": "air",
    "OPEN": "OPENS",
    "CLOSE": "CLOSED",
    "BYE": "BYEE",
    "black": "dark",
    "right": "straight",
    "15": "14",
    "Berlin": "Munich",
    "Madrid": "Barcelona",
    "rabbit": "bunny",
    "winter": "autumn",
    "START": "BEGIN",
    "STOP": "PAUSE",
    "pentagon": "polygon",
    "Earth": "world",
}


def build_rejected(chosen: str) -> str:
    key = chosen.strip()
    rejected = NEAR_MISS.get(key)
    if rejected is None:
        if key.isdigit():
            rejected = str(int(key) + 1)
        elif len(key) > 3:
            rejected = key[:-1]
        else:
            rejected = key + "X"
    prefix = " " if chosen.startswith(" ") else ""
    return prefix + rejected


def convert(input_path: Path, output_path: Path):
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append(
            {
                "prompt": rec["prompt"],
                "chosen": rec["chosen"],
                "rejected": build_rejected(rec["chosen"]),
            }
        )
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "count": len(rows), "output": str(output_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build synthetic preference pairs from chosen-only JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
