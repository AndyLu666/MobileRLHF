#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DESIRABLE_LABEL = "desirable"
UNDESIRABLE_LABEL = "undesirable"


def convert_record(rec: dict, idx: int) -> list[dict]:
    prompt = rec["prompt"]
    base_meta = {
        key: value
        for key, value in rec.items()
        if key not in {"prompt", "chosen", "rejected", "response", "completion", "label"}
    }
    preference_id = rec.get("preference_id", f"pref_{idx:08d}")
    return [
        {
            "prompt": prompt,
            "response": rec["chosen"],
            "completion": rec["chosen"],
            "label": DESIRABLE_LABEL,
            "desirable": True,
            "preference_id": preference_id,
            "feedback_source": "chosen_from_pair",
            **base_meta,
        },
        {
            "prompt": prompt,
            "response": rec["rejected"],
            "completion": rec["rejected"],
            "label": UNDESIRABLE_LABEL,
            "desirable": False,
            "preference_id": preference_id,
            "feedback_source": "rejected_from_pair",
            **base_meta,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pairwise preference JSONL into binary feedback JSONL for KTO-style training."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    row_count = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rows = convert_record(rec, idx)
            except Exception:
                skipped += 1
                continue
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
            pair_count += 1

    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(input_path),
                "output": str(output_path),
                "preference_pairs": pair_count,
                "binary_rows": row_count,
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
