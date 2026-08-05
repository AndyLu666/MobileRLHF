#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text.strip() else ""


def _write_record(out, prompt: str, response: str, label: bool, row: dict[str, Any]) -> None:
    record = {
        "prompt": prompt,
        "response": response,
        "label": label,
        "feedback_format": "kto_binary",
    }
    source = row.get("source")
    if source:
        record["source"] = source
    meta = row.get("meta")
    if isinstance(meta, dict):
        record["meta"] = meta
    out.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_dpo_pairs_to_kto_records(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "pairs": 0,
        "written": 0,
        "desirable": 0,
        "undesirable": 0,
        "skipped_invalid_json": 0,
        "skipped_missing_fields": 0,
        "skipped_identical": 0,
    }

    valid_rows: list[tuple[str, str, str, dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_invalid_json"] += 1
                continue

            prompt = _clean_text(row.get("prompt"))
            chosen = _clean_text(row.get("chosen"))
            rejected = _clean_text(row.get("rejected"))
            if not prompt or not chosen or not rejected:
                stats["skipped_missing_fields"] += 1
                continue
            if chosen == rejected:
                stats["skipped_identical"] += 1
                continue

            valid_rows.append((prompt, chosen, rejected, row))
            stats["pairs"] += 1
            stats["written"] += 2
            stats["desirable"] += 1
            stats["undesirable"] += 1

    # Match TRL's preference-unpairing layout: all desirable completions first,
    # followed by all undesirable completions. This prevents a two-row batch
    # from trivially rotating the two responses belonging to the same prompt.
    with output_path.open("w", encoding="utf-8") as out:
        for prompt, chosen, _, row in valid_rows:
            _write_record(out, prompt, chosen, True, row)
        for prompt, _, rejected, row in valid_rows:
            _write_record(out, prompt, rejected, False, row)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DPO prompt/chosen/rejected JSONL to KTO binary JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Input DPO JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output KTO binary JSONL.")
    parser.add_argument("--stats", type=Path, default=None, help="Optional stats JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = convert_dpo_pairs_to_kto_records(args.input, args.output)
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(stats_json + "\n", encoding="utf-8")
    print(stats_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
